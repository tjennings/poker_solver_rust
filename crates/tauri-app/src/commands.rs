//! Tauri commands for the poker solver application.
//!
//! Provides async training with progress events and strategy persistence.

use poker_solver_core::cfr::{calculate_exploitability, MccfrSolver};
use poker_solver_core::game::KuhnPoker;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::{AppHandle, Emitter, State};
use tokio::sync::Mutex;

/// Training state shared across commands.
pub struct TrainingState {
    /// Flag to signal training should stop
    pub stop_flag: Arc<AtomicBool>,
    /// Current training task handle
    pub is_running: Arc<AtomicBool>,
    /// Last trained strategies (for saving)
    pub strategies: Arc<Mutex<Option<TrainedStrategy>>>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            stop_flag: Arc::new(AtomicBool::new(false)),
            is_running: Arc::new(AtomicBool::new(false)),
            strategies: Arc::new(Mutex::new(None)),
        }
    }
}

/// Trained strategy data for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainedStrategy {
    pub game_type: String,
    pub iterations: u64,
    pub exploitability: f64,
    pub strategies: HashMap<String, Vec<f64>>,
}

/// Progress checkpoint emitted during training.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingProgress {
    pub iteration: u64,
    pub total_iterations: u64,
    pub exploitability: f64,
    pub elapsed_ms: u64,
    pub running: bool,
}

/// Final training result.
#[derive(Debug, Clone, Serialize)]
pub struct TrainingResult {
    pub iterations: u64,
    pub strategies: HashMap<String, Vec<f64>>,
    pub exploitability: f64,
    pub elapsed_ms: u64,
    pub stopped_early: bool,
}

/// Legacy checkpoint type for compatibility.
#[derive(Serialize)]
pub struct Checkpoint {
    pub iteration: u64,
    pub exploitability: f64,
    pub elapsed_ms: u64,
}

/// Legacy result type for compatibility.
#[derive(Serialize)]
pub struct TrainingResultWithCheckpoints {
    pub checkpoints: Vec<Checkpoint>,
    pub strategies: HashMap<String, Vec<f64>>,
    pub total_iterations: u64,
    pub total_elapsed_ms: u64,
}

/// Start async training with progress events.
#[tauri::command]
pub async fn start_training(
    app: AppHandle,
    state: State<'_, TrainingState>,
    iterations: u64,
    checkpoint_interval: u64,
) -> Result<TrainingResult, String> {
    // Check if already running
    if state.is_running.load(Ordering::SeqCst) {
        return Err("Training already in progress".to_string());
    }

    // Set running flag and clear stop flag
    state.is_running.store(true, Ordering::SeqCst);
    state.stop_flag.store(false, Ordering::SeqCst);

    let stop_flag = state.stop_flag.clone();
    let is_running = state.is_running.clone();
    let strategies_store = state.strategies.clone();

    let start = std::time::Instant::now();
    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game.clone());

    let mut current_iteration = 0u64;
    let mut stopped_early = false;

    while current_iteration < iterations {
        // Check for stop signal
        if stop_flag.load(Ordering::SeqCst) {
            stopped_early = true;
            break;
        }

        let chunk = checkpoint_interval.min(iterations - current_iteration);
        solver.train_full(chunk);
        current_iteration += chunk;

        let strategies = solver.all_strategies();
        let exploitability = calculate_exploitability(&game, &strategies);

        // Emit progress event
        let progress = TrainingProgress {
            iteration: current_iteration,
            total_iterations: iterations,
            exploitability,
            elapsed_ms: start.elapsed().as_millis() as u64,
            running: true,
        };

        let _ = app.emit("training-progress", &progress);
    }

    let final_strategies = solver.all_strategies();
    let final_exploitability = calculate_exploitability(&game, &final_strategies);

    // Store strategies for potential save
    {
        let mut store = strategies_store.lock().await;
        *store = Some(TrainedStrategy {
            game_type: "kuhn".to_string(),
            iterations: current_iteration,
            exploitability: final_exploitability,
            strategies: final_strategies.clone(),
        });
    }

    // Clear running flag
    is_running.store(false, Ordering::SeqCst);

    // Emit final progress
    let _ = app.emit(
        "training-progress",
        &TrainingProgress {
            iteration: current_iteration,
            total_iterations: iterations,
            exploitability: final_exploitability,
            elapsed_ms: start.elapsed().as_millis() as u64,
            running: false,
        },
    );

    Ok(TrainingResult {
        iterations: current_iteration,
        strategies: final_strategies,
        exploitability: final_exploitability,
        elapsed_ms: start.elapsed().as_millis() as u64,
        stopped_early,
    })
}

/// Stop the current training.
#[tauri::command]
pub fn stop_training(state: State<'_, TrainingState>) -> Result<(), String> {
    if !state.is_running.load(Ordering::SeqCst) {
        return Err("No training in progress".to_string());
    }
    state.stop_flag.store(true, Ordering::SeqCst);
    Ok(())
}

/// Get current training status.
#[tauri::command]
pub fn get_training_status(state: State<'_, TrainingState>) -> bool {
    state.is_running.load(Ordering::SeqCst)
}

/// Save strategy to a file (MessagePack format).
#[tauri::command]
pub async fn save_strategy(state: State<'_, TrainingState>, path: String) -> Result<(), String> {
    let store = state.strategies.lock().await;
    let strategy = store
        .as_ref()
        .ok_or_else(|| "No strategy to save. Train first.".to_string())?;

    let bytes = rmp_serde::to_vec(strategy).map_err(|e| format!("Serialization error: {e}"))?;

    std::fs::write(&path, bytes).map_err(|e| format!("Failed to write file: {e}"))?;

    Ok(())
}

/// Load strategy from a file (MessagePack format).
#[tauri::command]
pub async fn load_strategy(
    state: State<'_, TrainingState>,
    path: String,
) -> Result<TrainedStrategy, String> {
    let bytes = std::fs::read(&path).map_err(|e| format!("Failed to read file: {e}"))?;

    let strategy: TrainedStrategy =
        rmp_serde::from_slice(&bytes).map_err(|e| format!("Deserialization error: {e}"))?;

    // Store loaded strategy
    {
        let mut store = state.strategies.lock().await;
        *store = Some(strategy.clone());
    }

    Ok(strategy)
}

// ============================================================================
// Legacy commands for backwards compatibility
// ============================================================================

#[tauri::command]
pub fn run_kuhn_training(iterations: u64) -> Result<TrainingResultWithCheckpoints, String> {
    let start = std::time::Instant::now();

    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game.clone());
    solver.train_full(iterations);

    let strategies = solver.all_strategies();
    let exploitability = calculate_exploitability(&game, &strategies);

    Ok(TrainingResultWithCheckpoints {
        checkpoints: vec![Checkpoint {
            iteration: iterations,
            exploitability,
            elapsed_ms: start.elapsed().as_millis() as u64,
        }],
        strategies,
        total_iterations: iterations,
        total_elapsed_ms: start.elapsed().as_millis() as u64,
    })
}

#[tauri::command]
pub fn train_with_checkpoints(
    total_iterations: u64,
    checkpoint_interval: u64,
) -> Result<TrainingResultWithCheckpoints, String> {
    let start = std::time::Instant::now();

    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game.clone());
    let mut checkpoints = Vec::new();

    let mut current_iteration = 0u64;
    while current_iteration < total_iterations {
        let iterations_this_chunk = checkpoint_interval.min(total_iterations - current_iteration);

        solver.train_full(iterations_this_chunk);
        current_iteration += iterations_this_chunk;

        let strategies = solver.all_strategies();
        let exploitability = calculate_exploitability(&game, &strategies);

        checkpoints.push(Checkpoint {
            iteration: current_iteration,
            exploitability,
            elapsed_ms: start.elapsed().as_millis() as u64,
        });
    }

    Ok(TrainingResultWithCheckpoints {
        checkpoints,
        strategies: solver.all_strategies(),
        total_iterations: current_iteration,
        total_elapsed_ms: start.elapsed().as_millis() as u64,
    })
}
