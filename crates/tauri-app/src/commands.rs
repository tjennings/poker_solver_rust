use poker_solver_core::cfr::{calculate_exploitability, MccfrSolver};
use poker_solver_core::game::KuhnPoker;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct TrainingResult {
    pub iterations: u64,
    pub strategies: HashMap<String, Vec<f64>>,
    pub elapsed_ms: u64,
}

#[derive(Serialize)]
pub struct Checkpoint {
    pub iteration: u64,
    pub exploitability: f64,
    pub elapsed_ms: u64,
}

#[derive(Serialize)]
pub struct TrainingResultWithCheckpoints {
    pub checkpoints: Vec<Checkpoint>,
    pub strategies: HashMap<String, Vec<f64>>,
    pub total_iterations: u64,
    pub total_elapsed_ms: u64,
}

#[tauri::command]
pub fn run_kuhn_training(iterations: u64) -> Result<TrainingResult, String> {
    let start = std::time::Instant::now();

    let game = KuhnPoker::new();
    let mut solver = MccfrSolver::new(game);
    solver.train_full(iterations);

    Ok(TrainingResult {
        iterations: solver.iterations(),
        strategies: solver.all_strategies(),
        elapsed_ms: start.elapsed().as_millis() as u64,
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
