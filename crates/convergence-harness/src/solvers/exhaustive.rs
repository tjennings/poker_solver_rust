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

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str};
    use range_solver::CardConfig;

    /// Build a minimal river game for testing: narrow ranges, single river card,
    /// simple bet sizes. Fast to construct and solve.
    fn build_test_river_game() -> PostFlopGame {
        let oop_range = "AA,KK,QQ,AKs,AKo";
        let ip_range = "JJ,TT,99,AQs,AQo";

        let card_config = CardConfig {
            range: [oop_range.parse().unwrap(), ip_range.parse().unwrap()],
            flop: flop_from_str("Td9d6h").unwrap(),
            turn: card_from_str("2c").unwrap(),
            river: card_from_str("3s").unwrap(),
        };

        let bet_sizes = BetSizeOptions::try_from(("50%,a", "2.5x")).unwrap();

        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 200,
            rake_rate: 0.0,
            rake_cap: 0.0,
            flop_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            turn_bet_sizes: [bet_sizes.clone(), bet_sizes.clone()],
            river_bet_sizes: [bet_sizes.clone(), bet_sizes],
            turn_donk_sizes: None,
            river_donk_sizes: None,
            add_allin_threshold: 1.5,
            force_allin_threshold: 0.15,
            merging_threshold: 0.1,
            depth_limit: None,
        };

        let action_tree = ActionTree::new(tree_config).unwrap();
        PostFlopGame::with_config(card_config, action_tree).unwrap()
    }

    #[test]
    fn test_exhaustive_solver_name() {
        let game = build_test_river_game();
        let solver = ExhaustiveSolver::new(game);
        assert_eq!(solver.name(), "Exhaustive DCFR");
    }

    #[test]
    fn test_exhaustive_solver_starts_at_zero_iterations() {
        let game = build_test_river_game();
        let solver = ExhaustiveSolver::new(game);
        assert_eq!(solver.iterations(), 0);
    }

    #[test]
    fn test_exhaustive_solver_runs_iterations() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);

        assert_eq!(solver.iterations(), 0);
        let expl_before = solver.exploitability();

        // Run 10 iterations
        for _ in 0..10 {
            solver.solve_step();
        }

        assert_eq!(solver.iterations(), 10);
        let expl_after = solver.exploitability();

        // Exploitability should decrease after iterations
        assert!(
            expl_after < expl_before,
            "Exploitability should decrease: before={}, after={}",
            expl_before,
            expl_after
        );
    }

    #[test]
    fn test_exhaustive_solver_game_accessor() {
        let game = build_test_river_game();
        let combo_count = game.private_cards(0).len();
        let solver = ExhaustiveSolver::new(game);
        assert_eq!(solver.game().private_cards(0).len(), combo_count);
    }

    #[test]
    fn test_exhaustive_solver_into_game() {
        let game = build_test_river_game();
        let combo_count = game.private_cards(0).len();
        let solver = ExhaustiveSolver::new(game);
        let recovered = solver.into_game();
        assert_eq!(recovered.private_cards(0).len(), combo_count);
    }

    #[test]
    fn test_exhaustive_solver_finalize() {
        let game = build_test_river_game();
        let mut solver = ExhaustiveSolver::new(game);

        // Run a few iterations first
        for _ in 0..5 {
            solver.solve_step();
        }

        // Finalize should not panic
        solver.finalize();
    }

    #[test]
    fn test_exhaustive_solver_stub_strategy_returns_empty() {
        let game = build_test_river_game();
        let solver = ExhaustiveSolver::new(game);
        assert!(solver.average_strategy().is_empty());
    }

    #[test]
    fn test_exhaustive_solver_stub_combo_evs_returns_empty() {
        let game = build_test_river_game();
        let solver = ExhaustiveSolver::new(game);
        assert!(solver.combo_evs().is_empty());
    }

    #[test]
    fn test_exhaustive_solver_self_reported_metrics_default() {
        let game = build_test_river_game();
        let solver = ExhaustiveSolver::new(game);
        let metrics = solver.self_reported_metrics();
        assert!(metrics.values.is_empty());
    }
}
