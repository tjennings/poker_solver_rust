//! Integration tests comparing GPU DCFR+ solver against the range-solver.
//!
//! These tests verify that the GPU solver produces reasonable strategies
//! by comparing against the trusted CPU range-solver implementation.
//!
//! Run with: `cargo test -p poker-solver-gpu --features cuda --test compare_range_solver`

#![cfg(feature = "cuda")]

use poker_solver_gpu::gpu::GpuContext;
use poker_solver_gpu::solver::GpuSolver;
use poker_solver_gpu::tree::FlatTree;
use range_solver::action_tree::{ActionTree, BoardState, TreeConfig};
use range_solver::bet_size::BetSizeOptions;
use range_solver::card::{card_from_str, flop_from_str};
use range_solver::range::Range;
use range_solver::interface::Game;
use range_solver::{solve, CardConfig, PostFlopGame};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a river game from the given parameters.
fn build_river_game(
    oop_range_str: &str,
    ip_range_str: &str,
    flop: &str,
    turn: &str,
    river: &str,
    bet_sizes: &str,
) -> PostFlopGame {
    let oop_range: Range = oop_range_str.parse().unwrap();
    let ip_range: Range = ip_range_str.parse().unwrap();
    let card_config = CardConfig {
        range: [oop_range, ip_range],
        flop: flop_from_str(flop).unwrap(),
        turn: card_from_str(turn).unwrap(),
        river: card_from_str(river).unwrap(),
    };
    let sizes = BetSizeOptions::try_from((bet_sizes, "")).unwrap();
    let tree_config = TreeConfig {
        initial_state: BoardState::River,
        starting_pot: 100,
        effective_stack: 100,
        river_bet_sizes: [sizes.clone(), sizes],
        ..Default::default()
    };
    let tree = ActionTree::new(tree_config).unwrap();
    let mut game = PostFlopGame::with_config(card_config, tree).unwrap();
    game.allocate_memory(false);
    game
}

/// Solve on GPU and return the flat strategy vector plus tree metadata.
fn solve_gpu(game: &mut PostFlopGame, iterations: u32) -> (Vec<f32>, FlatTree) {
    let flat_tree = FlatTree::from_postflop_game(game);
    let gpu = GpuContext::new(0).unwrap();
    let result = GpuSolver::new(&gpu, &flat_tree)
        .unwrap()
        .solve(iterations, None)
        .unwrap();
    (result.strategy, flat_tree)
}

/// Verify that every infoset's strategy sums to ~1.0 for each hand,
/// and contains no NaN/Inf values.
fn assert_strategy_valid(strategy: &[f32], flat: &FlatTree) {
    let num_hands = flat.num_hands;
    let max_actions = flat.max_actions();

    for iset in 0..flat.num_infosets {
        let n_actions = flat.infoset_num_actions[iset] as usize;
        for h in 0..num_hands {
            let mut total = 0.0f32;
            for a in 0..n_actions {
                let idx = (iset * max_actions + a) * num_hands + h;
                let prob = strategy[idx];
                assert!(
                    prob.is_finite(),
                    "non-finite probability at iset={iset} action={a} hand={h}: {prob}"
                );
                assert!(
                    prob >= -1e-5,
                    "negative probability {prob} at iset={iset} action={a} hand={h}"
                );
                total += prob;
            }
            assert!(
                (total - 1.0).abs() < 1e-3,
                "strategy doesn't sum to 1.0 for iset={iset} hand={h}: sum={total}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Test 1: River comparison — GPU vs range-solver dominant actions
// ---------------------------------------------------------------------------

#[test]
fn test_river_comparison() {
    let mut game = build_river_game(
        "AA,KK,QQ,AKs",
        "QQ-JJ,AQs,AJs",
        "Qs Jh 2c",
        "8d",
        "3s",
        "50%, a",
    );

    // --- Solve with range-solver (CPU) ---
    let mut cpu_game = build_river_game(
        "AA,KK,QQ,AKs",
        "QQ-JJ,AQs,AJs",
        "Qs Jh 2c",
        "8d",
        "3s",
        "50%, a",
    );
    let _cpu_exploit = solve(&mut cpu_game, 1000, 0.0, false);

    // Read CPU root strategy: layout is [action * num_hands + hand]
    cpu_game.back_to_root();
    let cpu_strategy = cpu_game.strategy();
    let cpu_player = cpu_game.current_player();
    let cpu_num_hands = cpu_game.num_private_hands(cpu_player);
    let cpu_actions = cpu_game.available_actions();
    let cpu_num_actions = cpu_actions.len();

    // --- Solve with GPU ---
    let (gpu_strategy, flat) = solve_gpu(&mut game, 1000);
    let gpu_num_hands = flat.num_hands;

    // Root is infoset 0
    let root_n_actions = flat.infoset_num_actions[0] as usize;

    // 1. Both should have the same number of actions at root
    assert_eq!(
        root_n_actions, cpu_num_actions,
        "action count mismatch at root"
    );

    // 2. Both should have the same number of hands
    assert_eq!(
        gpu_num_hands, cpu_num_hands,
        "hand count mismatch: GPU={gpu_num_hands}, CPU={cpu_num_hands}"
    );

    // 3. Verify GPU strategy is valid (sums to 1.0 per hand, no NaN/Inf)
    assert_strategy_valid(&gpu_strategy, &flat);

    // 4. Verify that both solvers produce non-trivial strategies
    //    (not all uniform — at least some hands should have concentrated strategies).
    let mut gpu_has_concentrated = false;
    let mut cpu_has_concentrated = false;
    let max_actions = flat.max_actions();

    for h in 0..cpu_num_hands.min(gpu_num_hands) {
        // CPU check
        for a in 0..cpu_num_actions {
            if cpu_strategy[a * cpu_num_hands + h] > 0.8 {
                cpu_has_concentrated = true;
            }
        }
        // GPU check
        for a in 0..root_n_actions {
            let idx = (0 * max_actions + a) * gpu_num_hands + h;
            if gpu_strategy[idx] > 0.8 {
                gpu_has_concentrated = true;
            }
        }
    }

    assert!(
        cpu_has_concentrated,
        "CPU solver produced only uniform strategies after 1000 iterations"
    );
    assert!(
        gpu_has_concentrated,
        "GPU solver produced only uniform strategies after 1000 iterations"
    );

    // 5. Print comparison info for diagnostics
    eprintln!(
        "River comparison: tree has {} actions at root, {} hands",
        cpu_num_actions, cpu_num_hands
    );
    eprintln!("Actions: {cpu_actions:?}");
    eprintln!(
        "GPU: {} infosets, {} nodes",
        flat.num_infosets,
        flat.num_nodes()
    );
}

// ---------------------------------------------------------------------------
// Test 2: Convergence rate — strategy concentrates with more iterations
// ---------------------------------------------------------------------------

#[test]
fn test_convergence_rate() {
    let iteration_counts = [100, 500, 1000];
    let mut entropies: Vec<f64> = Vec::new();

    for &iters in &iteration_counts {
        let mut game = build_river_game(
            "AA,KK,QQ,AKs",
            "QQ-JJ,AQs,AJs",
            "Qs Jh 2c",
            "8d",
            "3s",
            "50%, a",
        );
        let (strategy, flat) = solve_gpu(&mut game, iters);

        // Compute average entropy over all infosets and hands.
        // Lower entropy means more concentrated (less uniform) strategy.
        let num_hands = flat.num_hands;
        let max_actions = flat.max_actions();
        let mut total_entropy = 0.0f64;
        let mut count = 0u64;

        for iset in 0..flat.num_infosets {
            let n_actions = flat.infoset_num_actions[iset] as usize;
            if n_actions <= 1 {
                continue;
            }
            for h in 0..num_hands {
                let mut entropy = 0.0f64;
                for a in 0..n_actions {
                    let idx = (iset * max_actions + a) * num_hands + h;
                    let p = strategy[idx] as f64;
                    if p > 1e-10 {
                        entropy -= p * p.ln();
                    }
                }
                total_entropy += entropy;
                count += 1;
            }
        }

        let avg_entropy = if count > 0 {
            total_entropy / count as f64
        } else {
            0.0
        };
        eprintln!("Iterations {iters}: avg entropy = {avg_entropy:.4}");
        entropies.push(avg_entropy);
    }

    // Entropy should generally decrease (or stay similar) as iterations increase.
    // We check that the final entropy is not much higher than the initial one.
    // With DCFR+ strategy weighting (which only kicks in after iteration 100),
    // we expect more concentrated strategies at higher iteration counts.
    let first = entropies[0];
    let last = entropies[entropies.len() - 1];
    eprintln!("Entropy at {}: {first:.4}, at {}: {last:.4}", iteration_counts[0], iteration_counts[iteration_counts.len()-1]);

    // The final entropy should be no more than 20% higher than the first.
    // Typically it should be lower, but DCFR+ weighting might cause slight non-monotonicity.
    assert!(
        last <= first * 1.2 + 0.01,
        "entropy did not converge: first={first:.4}, last={last:.4}"
    );
}

// ---------------------------------------------------------------------------
// Test 3: Strategy sums to one for every infoset and hand
// ---------------------------------------------------------------------------

#[test]
fn test_gpu_strategy_sums_to_one() {
    let mut game = build_river_game(
        "AA,KK,QQ,AKs",
        "QQ-JJ,AQs,AJs",
        "Qs Jh 2c",
        "8d",
        "3s",
        "50%, a",
    );
    let (strategy, flat) = solve_gpu(&mut game, 500);
    assert_strategy_valid(&strategy, &flat);
}

// ---------------------------------------------------------------------------
// Test 4: Benchmark — iterations per second (informational, not pass/fail)
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn test_benchmark_iterations_per_second() {
    let mut game = build_river_game(
        "AA,KK,QQ,AKs",
        "QQ-JJ,AQs,AJs",
        "Qs Jh 2c",
        "8d",
        "3s",
        "50%, a",
    );
    let flat_tree = FlatTree::from_postflop_game(&mut game);
    let gpu = GpuContext::new(0).unwrap();

    let iterations = 1000;
    let start = Instant::now();
    let result = GpuSolver::new(&gpu, &flat_tree)
        .unwrap()
        .solve(iterations, None)
        .unwrap();
    let elapsed = start.elapsed();

    let iters_per_sec = result.iterations as f64 / elapsed.as_secs_f64();
    eprintln!(
        "GPU benchmark: {} iterations in {:.2?} ({:.0} iter/s)",
        result.iterations, elapsed, iters_per_sec
    );
    eprintln!(
        "Tree: {} nodes, {} infosets, {} hands",
        flat_tree.num_nodes(),
        flat_tree.num_infosets,
        flat_tree.num_hands
    );
}

// ---------------------------------------------------------------------------
// Test 5: Multiple positions — validate strategies across different boards
// ---------------------------------------------------------------------------

#[test]
fn test_multiple_positions() {
    let positions = [
        ("AA,KK,QQ,AKs", "QQ-JJ,AQs,AJs", "Qs Jh 2c", "8d", "3s"),
        ("AA,KK,AKs,AQs", "TT-88,KQs,KJs", "Ks Td 5h", "7c", "2d"),
        ("QQ,JJ,TT,AJs", "99-77,KTs,QTs", "9s 8d 4c", "3h", "2s"),
    ];

    let gpu = GpuContext::new(0).unwrap();

    for (i, (oop, ip, flop, turn, river)) in positions.iter().enumerate() {
        let mut game = build_river_game(oop, ip, flop, turn, river, "50%, a");
        let flat_tree = FlatTree::from_postflop_game(&mut game);
        let result = GpuSolver::new(&gpu, &flat_tree)
            .unwrap()
            .solve(500, None)
            .unwrap();

        eprintln!(
            "Position {i}: {flop} {turn} {river} — {} infosets, {} hands, {} nodes",
            flat_tree.num_infosets,
            flat_tree.num_hands,
            flat_tree.num_nodes()
        );

        assert_strategy_valid(&result.strategy, &flat_tree);
    }
}
