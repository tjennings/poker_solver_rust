pub mod extract;
pub mod gpu;
pub mod kernels;
pub mod solver;
pub mod terminal;

/// Configuration for the GPU range solver.
pub struct GpuSolverConfig {
    pub max_iterations: u32,
    pub target_exploitability: f32,
    pub print_progress: bool,
}

/// Result of a GPU solve.
pub struct GpuSolveResult {
    /// Final exploitability achieved.
    pub exploitability: f32,
    /// Number of iterations run.
    pub iterations_run: u32,
    /// Average strategy at root: `[num_actions * num_hands]` in row-major order.
    pub root_strategy: Vec<f32>,
}

/// Solve a postflop game on GPU and return the result (legacy multi-launch).
pub fn gpu_solve_game(
    game: &range_solver::PostFlopGame,
    config: &GpuSolverConfig,
) -> GpuSolveResult {
    use range_solver::interface::Game;

    let topo = extract::extract_topology(game);
    let term = extract::extract_terminal_data(game, &topo);
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
    let initial_weights: [Vec<f32>; 2] = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];

    solver::gpu_solve_cudarc(&topo, &term, config, &initial_weights, num_hands)
        .expect("GPU solve failed")
}

/// Solve a postflop game on GPU using the single cooperative mega-kernel.
pub fn gpu_solve_mega(
    game: &range_solver::PostFlopGame,
    config: &GpuSolverConfig,
) -> GpuSolveResult {
    use range_solver::interface::Game;

    let topo = extract::extract_topology(game);
    let term = extract::extract_terminal_data(game, &topo);
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
    let initial_weights: [Vec<f32>; 2] = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];

    solver::gpu_solve_mega(&topo, &term, config, &initial_weights, num_hands)
        .expect("GPU mega-kernel solve failed")
}

/// Solve a postflop game on GPU using the hand-parallel kernel.
/// One thread block = one subgame, threads = hands, sequential tree traversal.
/// No cooperative groups or grid.sync() required.
pub fn gpu_solve_hand_parallel(
    game: &range_solver::PostFlopGame,
    config: &GpuSolverConfig,
) -> GpuSolveResult {
    use range_solver::interface::Game;

    let topo = extract::extract_topology(game);
    let term = extract::extract_terminal_data(game, &topo);
    let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
    let initial_weights: [Vec<f32>; 2] = [
        game.initial_weights(0).to_vec(),
        game.initial_weights(1).to_vec(),
    ];

    solver::gpu_solve_hand_parallel(&topo, &term, config, &initial_weights, num_hands)
        .expect("GPU hand-parallel solve failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
    use range_solver::bet_size::BetSizeOptions;
    use range_solver::card::{card_from_str, flop_from_str, CardConfig};

    fn make_river_game() -> range_solver::PostFlopGame {
        let oop_range = "AA,KK,QQ,AKs".parse().unwrap();
        let ip_range = "QQ-JJ,AQs,AJs".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: card_from_str("3s").unwrap(),
        };
        let sizes = BetSizeOptions::try_from(("100%", "")).unwrap();
        let tree_config = TreeConfig {
            initial_state: BoardState::River,
            starting_pot: 100,
            effective_stack: 100,
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = range_solver::PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn gpu_solve_river_game_reduces_exploitability() {
        let game = make_river_game();
        let config = GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.0,
            print_progress: false,
        };

        let result = gpu_solve_game(&game, &config);

        assert!(result.exploitability > -1e-4, "exploitability must be >= ~0, got {}", result.exploitability);
        assert!(result.exploitability.is_finite(), "exploitability must be finite");
        assert!(
            result.exploitability < 1.0,
            "exploitability should converge below 1.0 after 500 iterations, got {}",
            result.exploitability
        );
        assert!(result.iterations_run > 0, "must run at least 1 iteration");
        assert!(result.iterations_run <= 500, "must not exceed max_iterations");
        assert!(!result.root_strategy.is_empty(), "root_strategy must not be empty");
    }

    fn make_turn_game() -> range_solver::PostFlopGame {
        let oop_range = "AA".parse().unwrap();
        let ip_range = "KK".parse().unwrap();
        let card_config = CardConfig {
            range: [oop_range, ip_range],
            flop: flop_from_str("Qs Jh 2c").unwrap(),
            turn: card_from_str("8d").unwrap(),
            river: range_solver::card::NOT_DEALT,
        };
        let sizes = BetSizeOptions::try_from(("100%", "")).unwrap();
        let tree_config = range_solver::action_tree::TreeConfig {
            initial_state: BoardState::Turn,
            starting_pot: 100,
            effective_stack: 100,
            turn_bet_sizes: [sizes.clone(), sizes.clone()],
            river_bet_sizes: [sizes.clone(), sizes],
            ..Default::default()
        };
        let action_tree = ActionTree::new(tree_config).unwrap();
        let mut game = range_solver::PostFlopGame::with_config(card_config, action_tree).unwrap();
        game.allocate_memory(false);
        game
    }

    #[test]
    fn gpu_solve_turn_game_reduces_exploitability() {
        let game = make_turn_game();
        let config = GpuSolverConfig {
            max_iterations: 30,
            target_exploitability: 0.0,
            print_progress: false,
        };

        let result = gpu_solve_game(&game, &config);

        assert!(result.exploitability.is_finite(), "exploitability must be finite, got {}", result.exploitability);
        assert!(result.exploitability > -1.0, "exploitability must be >= ~0, got {}", result.exploitability);
        assert!(
            result.exploitability < 10.0,
            "exploitability should converge below 10.0 after 30 iterations on turn game, got {}",
            result.exploitability
        );
        assert!(result.iterations_run > 0, "must run at least 1 iteration");
        assert!(!result.root_strategy.is_empty(), "root_strategy must not be empty");
    }

    #[test]
    fn gpu_solve_turn_game_matches_cpu() {
        let game = make_turn_game();

        let mut cpu_game = make_turn_game();
        let cpu_expl = range_solver::solve(&mut cpu_game, 30, 0.0, false);

        let config = GpuSolverConfig {
            max_iterations: 30,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let gpu_result = gpu_solve_game(&game, &config);

        let ratio = if cpu_expl > 0.001 {
            gpu_result.exploitability / cpu_expl
        } else {
            1.0
        };
        assert!(
            ratio < 5.0,
            "GPU exploitability ({}) should be within 5x of CPU ({})",
            gpu_result.exploitability, cpu_expl
        );
    }

    #[test]
    fn gpu_solve_matches_cpu_convergence() {
        let game = make_river_game();

        let mut cpu_game = make_river_game();
        let cpu_expl = range_solver::solve(&mut cpu_game, 500, 0.0, false);

        let config = GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let gpu_result = gpu_solve_game(&game, &config);

        assert!(
            gpu_result.exploitability.abs() < 0.01,
            "GPU exploitability should converge near 0, got {}",
            gpu_result.exploitability
        );
        assert!(
            cpu_expl.abs() < 0.01,
            "CPU exploitability should converge near 0, got {}",
            cpu_expl
        );

        let diff = (gpu_result.exploitability - cpu_expl).abs();
        assert!(
            diff < 0.01,
            "GPU ({}) and CPU ({}) exploitability should be close, diff={}",
            gpu_result.exploitability, cpu_expl, diff
        );
    }

    #[test]
    fn mega_kernel_1iter_strategy_sum_nonzero() {
        let game = make_river_game();
        use range_solver::interface::Game;

        let topo = extract::extract_topology(&game);
        let term = extract::extract_terminal_data(&game, &topo);
        let num_hands = game.private_cards(0).len().max(game.private_cards(1).len());
        let initial_weights: [Vec<f32>; 2] = [
            game.initial_weights(0).to_vec(),
            game.initial_weights(1).to_vec(),
        ];

        // The root strategy should be non-empty after 1 iteration
        let config_1 = GpuSolverConfig { max_iterations: 1, target_exploitability: 0.0, print_progress: false };
        let r = solver::gpu_solve_mega(&topo, &term, &config_1, &initial_weights, num_hands).unwrap();
        assert!(!r.root_strategy.is_empty(), "mega root_strategy empty");
    }

    #[test]
    fn mega_kernel_turn_solve_converges() {
        let game = make_turn_game();

        let mut cpu_game = make_turn_game();
        let cpu_expl = range_solver::solve(&mut cpu_game, 100, 0.0, false);

        let config = GpuSolverConfig {
            max_iterations: 100,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let gpu_result = gpu_solve_mega(&game, &config);

        assert!(
            gpu_result.exploitability.is_finite(),
            "mega-kernel turn exploitability must be finite, got {}",
            gpu_result.exploitability
        );
        assert!(
            gpu_result.exploitability < 10.0,
            "mega-kernel turn solve should converge below 10.0, got {}",
            gpu_result.exploitability
        );
        let ratio = if cpu_expl > 0.001 {
            gpu_result.exploitability / cpu_expl
        } else {
            1.0
        };
        assert!(
            ratio < 5.0,
            "mega-kernel turn ({}) should be within 5x of CPU ({})",
            gpu_result.exploitability, cpu_expl
        );
        assert!(!gpu_result.root_strategy.is_empty(), "root_strategy must not be empty");
    }

    #[test]
    fn mega_kernel_solve_river_converges() {
        let game = make_river_game();

        let mut cpu_game = make_river_game();
        let cpu_expl = range_solver::solve(&mut cpu_game, 500, 0.0, false);

        let config = GpuSolverConfig {
            max_iterations: 500,
            target_exploitability: 0.0,
            print_progress: false,
        };
        let gpu_result = gpu_solve_mega(&game, &config);

        assert!(
            gpu_result.exploitability.is_finite(),
            "mega-kernel exploitability must be finite"
        );
        assert!(
            gpu_result.exploitability.abs() < 0.01,
            "mega-kernel exploitability should converge near 0, got {}",
            gpu_result.exploitability
        );
        let diff = (gpu_result.exploitability - cpu_expl).abs();
        assert!(
            diff < 0.01,
            "mega-kernel ({}) and CPU ({}) exploitability should be close, diff={}",
            gpu_result.exploitability, cpu_expl, diff
        );
        assert!(!gpu_result.root_strategy.is_empty(), "root_strategy must not be empty");
    }
}
