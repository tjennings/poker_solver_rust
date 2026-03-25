use range_solver::{compute_exploitability, finalize, solve_step, PostFlopGame};

use crate::solver_trait::{ComboEvMap, ConvergenceSolver, SolverMetrics, StrategyMap};

pub struct ExhaustiveSolver {
    game: PostFlopGame,
    iteration: u64,
}

impl ExhaustiveSolver {
    pub fn new(mut game: PostFlopGame) -> Self {
        game.allocate_memory(false);
        Self { game, iteration: 0 }
    }

    /// Compute exploitability without finalizing (can continue iterating).
    pub fn exploitability(&self) -> f32 {
        compute_exploitability(&self.game)
    }

    /// Access the underlying game for tree traversal and metric extraction.
    pub fn game(&self) -> &PostFlopGame {
        &self.game
    }

    /// Finalize: normalize strategy and compute EVs.
    /// WARNING: After calling this, no more iterations can run.
    pub fn finalize(&mut self) {
        finalize(&mut self.game);
    }

    /// Consume the solver and return the underlying PostFlopGame.
    pub fn into_game(self) -> PostFlopGame {
        self.game
    }
}

impl ConvergenceSolver for ExhaustiveSolver {
    fn name(&self) -> &str {
        "Exhaustive DCFR"
    }

    fn solve_step(&mut self) {
        solve_step(&self.game, self.iteration as u32);
        self.iteration += 1;
    }

    fn iterations(&self) -> u64 {
        self.iteration
    }

    fn average_strategy(&self) -> StrategyMap {
        // Tree traversal to collect strategy at every decision node.
        // Deferred to Task 5 (evaluator) where the tree walker is built.
        StrategyMap::new()
    }

    fn combo_evs(&self) -> ComboEvMap {
        // Deferred to Task 5 (evaluator) where the tree walker is built.
        ComboEvMap::new()
    }

    fn self_reported_metrics(&self) -> SolverMetrics {
        SolverMetrics::default()
    }
}

// Unit tests that require PostFlopGame / ExhaustiveSolver live in
// crates/convergence-harness/tests/integration.rs to avoid slowing
// down `cargo test --lib`.
