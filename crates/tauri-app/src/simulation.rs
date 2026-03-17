//! Simulation commands for running agent-vs-agent competitions.
//!
//! Plays complete poker hands between two strategy sources (trained bundles
//! or rule-based agents) and reports performance in mbb/h.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;
use serde::Serialize;
use tauri::{AppHandle, Emitter, State};

use poker_solver_core::agent::AgentConfig;
use poker_solver_core::simulation::{
    RuleBasedAgentGenerator, SimResult, run_simulation,
};

use crate::exploration::list_blueprints_core;

use rs_poker::arena::agent::{
    AgentGenerator, AllInAgentGenerator, CallingAgentGenerator, FoldingAgentGenerator,
    RandomAgentGenerator,
};

/// Abstraction over event emission so simulation logic works with both
/// Tauri's `AppHandle` and alternative backends (e.g. broadcast channels).
pub trait SimEventSink: Send + 'static {
    fn emit_progress(&self, event: SimProgressEvent);
    fn emit_complete(&self, event: SimResultResponse);
    fn emit_error(&self, msg: String);
}

/// `SimEventSink` implementation for Tauri's `AppHandle`.
impl SimEventSink for AppHandle {
    fn emit_progress(&self, event: SimProgressEvent) {
        let _ = self.emit("simulation-progress", event);
    }
    fn emit_complete(&self, event: SimResultResponse) {
        let _ = self.emit("simulation-complete", event);
    }
    fn emit_error(&self, msg: String) {
        let _ = self.emit("simulation-error", msg);
    }
}

/// Managed state for the simulation view.
pub struct SimulationState {
    running: Arc<AtomicBool>,
    result: Arc<RwLock<Option<SimResult>>>,
}

impl Default for SimulationState {
    fn default() -> Self {
        Self {
            running: Arc::new(AtomicBool::new(false)),
            result: Arc::new(RwLock::new(None)),
        }
    }
}

/// Info about an available strategy source (agent or bundle).
#[derive(Debug, Clone, Serialize)]
pub struct StrategySourceInfo {
    pub name: String,
    pub source_type: String,
    pub path: String,
}

/// Progress event emitted during simulation.
#[derive(Debug, Clone, Serialize)]
pub struct SimProgressEvent {
    pub hands_played: u64,
    pub total_hands: u64,
    pub p1_profit_bb: f64,
    pub current_mbbh: f64,
}

/// Result returned to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct SimResultResponse {
    pub hands_played: u64,
    pub p1_profit_bb: f64,
    pub mbbh: f64,
    pub equity_curve: Vec<f64>,
    pub elapsed_ms: u64,
}

/// List all available strategy sources (agents and trained bundles).
///
/// If `dir` is provided, scans that directory for blueprint bundles using the
/// same logic as the Explorer tab. Otherwise, no bundles are returned.
///
/// Core variant: no Tauri dependency, usable from any runtime.
pub fn list_strategy_sources_core(dir: Option<String>) -> Result<Vec<StrategySourceInfo>, String> {
    let mut sources = Vec::new();

    // Built-in agents from rs_poker
    for (name, key) in [
        ("Calling Station", "builtin:calling"),
        ("Folding", "builtin:folding"),
        ("All-In", "builtin:allin"),
        ("Random", "builtin:random"),
    ] {
        sources.push(StrategySourceInfo {
            name: name.to_string(),
            source_type: "builtin".to_string(),
            path: key.to_string(),
        });
    }

    // Find agents
    if let Some(agents_dir) = find_agents_dir() {
        let entries = std::fs::read_dir(&agents_dir)
            .map_err(|e| format!("Failed to read agents directory: {e}"))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("toml") {
                continue;
            }
            let name = match AgentConfig::load(&path) {
                Ok(config) => config.game.name.unwrap_or_else(|| {
                    path.file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("Unknown")
                        .to_string()
                }),
                Err(_) => continue,
            };
            sources.push(StrategySourceInfo {
                name,
                source_type: "agent".to_string(),
                path: path.to_string_lossy().to_string(),
            });
        }
    }

    // Scan blueprint_dir for trained bundles (same logic as Explorer tab)
    if let Some(blueprint_dir) = dir {
        if let Ok(blueprints) = list_blueprints_core(blueprint_dir) {
            for bp in blueprints {
                sources.push(StrategySourceInfo {
                    name: bp.name,
                    source_type: "bundle".to_string(),
                    path: bp.path,
                });
            }
        }
    }

    sources.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(sources)
}

/// Tauri command wrapper for `list_strategy_sources_core`.
#[tauri::command]
pub fn list_strategy_sources(dir: Option<String>) -> Result<Vec<StrategySourceInfo>, String> {
    list_strategy_sources_core(dir)
}

/// Start a simulation between two strategy sources.
///
/// Core variant: accepts any `SimEventSink` instead of requiring `AppHandle`.
/// Runs in a background thread and emits progress/complete/error events via the sink.
pub fn start_simulation_core(
    sink: impl SimEventSink,
    state: &SimulationState,
    p1_path: String,
    p2_path: String,
    num_hands: u64,
    stack_depth: u32,
) -> Result<(), String> {
    // Stop any running simulation
    state.running.store(false, Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));

    let running = Arc::clone(&state.running);
    let result_store = Arc::clone(&state.result);

    running.store(true, Ordering::SeqCst);
    *result_store.write() = None;

    // Build generators inside the thread since AgentGenerator isn't Send
    std::thread::spawn(move || {
        let (p1_gen, p1_bs) = match build_agent_generator(&p1_path) {
            Ok(g) => g,
            Err(e) => {
                sink.emit_error(e);
                running.store(false, Ordering::SeqCst);
                return;
            }
        };
        let (p2_gen, p2_bs) = match build_agent_generator(&p2_path) {
            Ok(g) => g,
            Err(e) => {
                sink.emit_error(e);
                running.store(false, Ordering::SeqCst);
                return;
            }
        };

        // Use bet_sizes from whichever agent is a trained bundle
        let bet_sizes = if !p1_bs.is_empty() {
            p1_bs
        } else if !p2_bs.is_empty() {
            p2_bs
        } else {
            vec![0.5, 1.0]
        };

        let stop = AtomicBool::new(false);
        let running_ref = &running;
        let mut last_emit = std::time::Instant::now();

        let sim_result = run_simulation(
            p1_gen,
            p2_gen,
            num_hands,
            stack_depth,
            &stop,
            &bet_sizes,
            |progress| {
                if !running_ref.load(Ordering::SeqCst) {
                    stop.store(true, Ordering::Relaxed);
                    return;
                }

                let now = std::time::Instant::now();
                let is_final = progress.hands_played >= progress.total_hands;

                // Throttle: emit at most every 100ms, but always emit the final update
                if !is_final && now.duration_since(last_emit).as_millis() < 100 {
                    return;
                }
                last_emit = now;

                sink.emit_progress(SimProgressEvent {
                    hands_played: progress.hands_played,
                    total_hands: progress.total_hands,
                    p1_profit_bb: progress.p1_profit_bb,
                    current_mbbh: progress.current_mbbh,
                });

                // Yield so the main thread can deliver stop commands
                std::thread::yield_now();
            },
        );

        match sim_result {
            Ok(result) => {
                sink.emit_complete(SimResultResponse {
                    hands_played: result.hands_played,
                    p1_profit_bb: result.p1_profit_bb,
                    mbbh: result.mbbh,
                    equity_curve: result.equity_curve.clone(),
                    elapsed_ms: result.elapsed_ms,
                });
                *result_store.write() = Some(result);
            }
            Err(e) => {
                sink.emit_error(e);
            }
        }

        running_ref.store(false, Ordering::SeqCst);
    });

    Ok(())
}

/// Tauri command wrapper for `start_simulation_core`.
#[tauri::command]
pub async fn start_simulation(
    app: AppHandle,
    state: State<'_, SimulationState>,
    p1_path: String,
    p2_path: String,
    num_hands: u64,
    stack_depth: u32,
) -> Result<(), String> {
    start_simulation_core(app, &state, p1_path, p2_path, num_hands, stack_depth)
}

/// Stop the currently running simulation.
///
/// Core variant: no Tauri dependency.
pub fn stop_simulation_core(state: &SimulationState) {
    state.running.store(false, Ordering::SeqCst);
}

/// Tauri command wrapper for `stop_simulation_core`.
#[tauri::command]
pub async fn stop_simulation(state: State<'_, SimulationState>) -> Result<(), String> {
    stop_simulation_core(&state);
    Ok(())
}

/// Get the result of the last completed simulation.
///
/// Core variant: no Tauri dependency.
pub fn get_simulation_result_core(
    state: &SimulationState,
) -> Result<Option<SimResultResponse>, String> {
    let guard = state.result.read();
    Ok(guard.as_ref().map(|r| SimResultResponse {
        hands_played: r.hands_played,
        p1_profit_bb: r.p1_profit_bb,
        mbbh: r.mbbh,
        equity_curve: r.equity_curve.clone(),
        elapsed_ms: r.elapsed_ms,
    }))
}

/// Tauri command wrapper for `get_simulation_result_core`.
#[tauri::command]
pub fn get_simulation_result(
    state: State<'_, SimulationState>,
) -> Result<Option<SimResultResponse>, String> {
    get_simulation_result_core(&state)
}

// ============================================================================
// Helpers
// ============================================================================

/// Build an agent generator and its associated bet sizes.
///
/// Returns `(generator, bet_sizes)` where `bet_sizes` is non-empty for
/// trained bundle agents and empty for built-in/rule-based agents.
fn build_agent_generator(path: &str) -> Result<(Box<dyn AgentGenerator>, Vec<f32>), String> {
    if let Some(builtin) = path.strip_prefix("builtin:") {
        return match builtin {
            "calling" => Ok((Box::new(CallingAgentGenerator), vec![])),
            "folding" => Ok((Box::new(FoldingAgentGenerator), vec![])),
            "allin" => Ok((Box::new(AllInAgentGenerator), vec![])),
            "random" => Ok((Box::<RandomAgentGenerator>::default(), vec![])),
            _ => Err(format!("Unknown built-in agent: {builtin}")),
        };
    }

    let path_buf = PathBuf::from(path);

    if path.ends_with(".toml") {
        let config = AgentConfig::load(&path_buf)
            .map_err(|e| format!("Failed to load agent config: {e}"))?;
        Ok((Box::new(RuleBasedAgentGenerator::new(Arc::new(config))), vec![]))
    } else {
        Err(format!(
            "Unsupported strategy source: {path}. Only .toml agent configs and built-in agents are supported."
        ))
    }
}

/// Walk up from CWD looking for an `agents/` directory.
fn find_agents_dir() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..5 {
        let candidate = dir.join("agents");
        if candidate.is_dir() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn list_strategy_sources_core_returns_builtins() {
        let sources = list_strategy_sources_core(None).unwrap();
        let builtins: Vec<_> = sources
            .iter()
            .filter(|s| s.source_type == "builtin")
            .collect();
        assert_eq!(builtins.len(), 4);
        assert!(builtins.iter().any(|s| s.path == "builtin:calling"));
        assert!(builtins.iter().any(|s| s.path == "builtin:folding"));
        assert!(builtins.iter().any(|s| s.path == "builtin:allin"));
        assert!(builtins.iter().any(|s| s.path == "builtin:random"));
    }

    #[timed_test]
    fn list_strategy_sources_core_sorted_by_name() {
        let sources = list_strategy_sources_core(None).unwrap();
        for i in 1..sources.len() {
            assert!(sources[i - 1].name <= sources[i].name);
        }
    }

    #[timed_test]
    fn stop_simulation_core_sets_running_false() {
        let state = SimulationState::default();
        state.running.store(true, Ordering::SeqCst);
        stop_simulation_core(&state);
        assert!(!state.running.load(Ordering::SeqCst));
    }

    #[timed_test]
    fn get_simulation_result_core_returns_none_initially() {
        let state = SimulationState::default();
        let result = get_simulation_result_core(&state).unwrap();
        assert!(result.is_none());
    }

    #[timed_test]
    fn get_simulation_result_core_returns_stored_result() {
        let state = SimulationState::default();
        *state.result.write() = Some(SimResult {
            hands_played: 100,
            p1_profit_bb: 5.0,
            mbbh: 50.0,
            equity_curve: vec![1.0, 2.0, 3.0],
            elapsed_ms: 42,
        });
        let result = get_simulation_result_core(&state).unwrap().unwrap();
        assert_eq!(result.hands_played, 100);
        assert_eq!(result.mbbh, 50.0);
        assert_eq!(result.equity_curve, vec![1.0, 2.0, 3.0]);
        assert_eq!(result.elapsed_ms, 42);
    }

    #[timed_test]
    fn sim_event_sink_trait_exists_and_is_object_safe() {
        // Verify trait can be used as a trait object (Send + 'static)
        struct TestSink;
        impl SimEventSink for TestSink {
            fn emit_progress(&self, _event: SimProgressEvent) {}
            fn emit_complete(&self, _event: SimResultResponse) {}
            fn emit_error(&self, _msg: String) {}
        }
        let sink: Box<dyn SimEventSink> = Box::new(TestSink);
        sink.emit_progress(SimProgressEvent {
            hands_played: 0,
            total_hands: 10,
            p1_profit_bb: 0.0,
            current_mbbh: 0.0,
        });
        sink.emit_complete(SimResultResponse {
            hands_played: 10,
            p1_profit_bb: 1.0,
            mbbh: 100.0,
            equity_curve: vec![],
            elapsed_ms: 5,
        });
        sink.emit_error("test error".to_string());
    }

    #[timed_test]
    fn sim_progress_event_serializes() {
        let event = SimProgressEvent {
            hands_played: 50,
            total_hands: 100,
            p1_profit_bb: 2.5,
            current_mbbh: 25.0,
        };
        let json = serde_json::to_string(&event).expect("should serialize");
        assert!(json.contains("\"hands_played\":50"));
        assert!(json.contains("\"total_hands\":100"));
    }

    #[timed_test]
    fn sim_result_response_serializes() {
        let resp = SimResultResponse {
            hands_played: 100,
            p1_profit_bb: 5.0,
            mbbh: 50.0,
            equity_curve: vec![1.0],
            elapsed_ms: 42,
        };
        let json = serde_json::to_string(&resp).expect("should serialize");
        assert!(json.contains("\"mbbh\":50"));
    }
}

