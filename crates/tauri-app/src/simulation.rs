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
use poker_solver_core::blueprint::StrategyBundle;
use poker_solver_core::simulation::{
    BlueprintAgentGenerator, RuleBasedAgentGenerator, SimResult, run_simulation,
};

use rs_poker::arena::agent::{
    AgentGenerator, AllInAgentGenerator, CallingAgentGenerator, FoldingAgentGenerator,
    RandomAgentGenerator,
};

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
#[tauri::command]
pub fn list_strategy_sources() -> Result<Vec<StrategySourceInfo>, String> {
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

    // Find bundle directories (look for config.yaml in subdirectories)
    if let Some(root) = find_project_root() {
        let entries = std::fs::read_dir(&root)
            .map_err(|e| format!("Failed to read directory: {e}"))?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() && path.join("config.yaml").exists() && path.join("blueprint.bin").exists() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("Unknown")
                    .to_string();
                sources.push(StrategySourceInfo {
                    name,
                    source_type: "bundle".to_string(),
                    path: path.to_string_lossy().to_string(),
                });
            }
        }
    }

    sources.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(sources)
}

/// Start a simulation between two strategy sources.
///
/// Runs in a background thread and emits `"simulation-progress"` events.
#[tauri::command]
pub async fn start_simulation(
    app: AppHandle,
    state: State<'_, SimulationState>,
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
                let _ = app.emit("simulation-error", e);
                running.store(false, Ordering::SeqCst);
                return;
            }
        };
        let (p2_gen, p2_bs) = match build_agent_generator(&p2_path) {
            Ok(g) => g,
            Err(e) => {
                let _ = app.emit("simulation-error", e);
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

                let _ = app.emit(
                    "simulation-progress",
                    SimProgressEvent {
                        hands_played: progress.hands_played,
                        total_hands: progress.total_hands,
                        p1_profit_bb: progress.p1_profit_bb,
                        current_mbbh: progress.current_mbbh,
                    },
                );

                // Yield so the main thread can deliver stop commands
                std::thread::yield_now();
            },
        );

        match sim_result {
            Ok(result) => {
                let _ = app.emit(
                    "simulation-complete",
                    SimResultResponse {
                        hands_played: result.hands_played,
                        p1_profit_bb: result.p1_profit_bb,
                        mbbh: result.mbbh,
                        equity_curve: result.equity_curve.clone(),
                        elapsed_ms: result.elapsed_ms,
                    },
                );
                *result_store.write() = Some(result);
            }
            Err(e) => {
                let _ = app.emit("simulation-error", e);
            }
        }

        running_ref.store(false, Ordering::SeqCst);
    });

    Ok(())
}

/// Stop the currently running simulation.
#[tauri::command]
pub async fn stop_simulation(state: State<'_, SimulationState>) -> Result<(), String> {
    state.running.store(false, Ordering::SeqCst);
    Ok(())
}

/// Get the result of the last completed simulation.
#[tauri::command]
pub fn get_simulation_result(
    state: State<'_, SimulationState>,
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
        let bundle = StrategyBundle::load(&path_buf)
            .map_err(|e| format!("Failed to load bundle: {e}"))?;
        let bet_sizes = bundle.config.game.bet_sizes.clone();
        Ok((
            Box::new(BlueprintAgentGenerator::new(
                Arc::new(bundle.blueprint),
                bundle.config,
            )),
            bet_sizes,
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

/// Walk up from CWD to find the project root (where agents/ or Cargo.toml is).
fn find_project_root() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    for _ in 0..5 {
        if dir.join("Cargo.toml").exists() && dir.join("crates").is_dir() {
            return Some(dir);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}
