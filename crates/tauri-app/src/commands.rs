use poker_solver_core::cfr::MccfrSolver;
use poker_solver_core::game::KuhnPoker;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize)]
pub struct TrainingResult {
    pub iterations: u64,
    pub strategies: HashMap<String, Vec<f64>>,
    pub elapsed_ms: u64,
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
