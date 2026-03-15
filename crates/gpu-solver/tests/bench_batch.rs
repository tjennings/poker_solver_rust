//! Batch GPU solver benchmark: 300 random river spots, GPU vs CPU.
//!
//! Generates random river spots with random boards and ranges (but uniform
//! pot/stack to ensure identical tree topology), solves them both in a single
//! batched GPU call and sequentially on CPU, then compares strategies for
//! correctness and prints performance results.
//!
//! Run:
//!   cargo test -p poker-solver-gpu --features cuda --release --test bench_batch test_batch_correctness -- --nocapture
//!   cargo test -p poker-solver-gpu --features cuda --release --test bench_batch bench_batch_300 -- --ignored --nocapture

#![cfg(feature = "cuda")]

use poker_solver_gpu::batch::{build_game_from_spot, BatchConfig, BatchGpuSolver, RiverSpot};
use poker_solver_gpu::gpu::GpuContext;
use poker_solver_gpu::tree::FlatTree;
use range_solver::interface::Game;
use range_solver::solve;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// ---------------------------------------------------------------------------
// Combo/card helpers
// ---------------------------------------------------------------------------

/// Map combo index (0..1326) to the two card indices (c1, c2) where c1 < c2.
/// Cards are 0..51 in the standard rank-major order.
fn combo_cards(index: usize) -> (u8, u8) {
    // Precompute once via the triangular formula.
    // combo(c1, c2) for c1 < c2: index = c1*(103-c1)/2 + c2 - c1 - 1
    // But the simple nested-loop mapping is clearest and only runs at init.
    let mut idx = 0usize;
    for c1 in 0..52u8 {
        for c2 in (c1 + 1)..52u8 {
            if idx == index {
                return (c1, c2);
            }
            idx += 1;
        }
    }
    unreachable!("combo index out of range")
}

/// Sample `n` unique cards from 0..51 using the given RNG.
fn sample_unique_cards(rng: &mut StdRng, n: usize) -> Vec<u8> {
    let mut cards = Vec::with_capacity(n);
    while cards.len() < n {
        let c: u8 = rng.gen_range(0..52);
        if !cards.contains(&c) {
            cards.push(c);
        }
    }
    cards
}

// ---------------------------------------------------------------------------
// Random spot generation
// ---------------------------------------------------------------------------

/// Generate `n` random river spots with a seeded RNG for reproducibility.
///
/// All spots use the same pot and stack (100/100) to ensure identical tree
/// topology. The BatchGpuSolver requires all spots to share the same action
/// tree structure, and varying pot/stack ratios can cause the tree builder's
/// allin thresholds (add_allin_threshold, force_allin_threshold) to merge or
/// eliminate bet sizes near all-in, producing trees with different node counts.
///
/// Only boards and ranges vary across spots — these affect per-hand data but
/// not tree shape.
fn generate_random_spots(n: usize, seed: u64) -> Vec<RiverSpot> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut spots = Vec::with_capacity(n);

    // Fixed pot/stack for all spots to guarantee identical tree topology.
    let pot = 100;
    let effective_stack = 100;

    for _ in 0..n {
        // Sample 5 unique board cards
        let board = sample_unique_cards(&mut rng, 5);
        let flop = [board[0], board[1], board[2]];
        let turn = board[3];
        let river = board[4];

        // Build a set of board cards for conflict checking
        let board_set: u64 = board.iter().fold(0u64, |mask, &c| mask | (1u64 << c));

        // Generate random range weights for OOP and IP
        let mut oop_range = vec![0.0f32; 1326];
        let mut ip_range = vec![0.0f32; 1326];

        for i in 0..1326 {
            let (c1, c2) = combo_cards(i);
            let combo_mask = (1u64 << c1) | (1u64 << c2);
            if combo_mask & board_set != 0 {
                // Combo conflicts with board cards — zero weight
                continue;
            }
            oop_range[i] = rng.gen_range(0.0f32..1.0f32);
            ip_range[i] = rng.gen_range(0.0f32..1.0f32);
        }

        spots.push(RiverSpot {
            flop,
            turn,
            river,
            oop_range,
            ip_range,
            pot,
            effective_stack,
        });
    }

    spots
}

// ---------------------------------------------------------------------------
// Strategy comparison helpers
// ---------------------------------------------------------------------------

/// Find the dominant (highest probability) action for a given hand at infoset 0
/// in a GPU per-spot strategy with layout `(iset * max_actions + action) * num_hands + hand`.
/// Returns (action_index, probability).
fn gpu_dominant_action(
    strategy: &[f32],
    num_actions: usize,
    max_actions: usize,
    num_hands: usize,
    hand: usize,
) -> (usize, f32) {
    let mut best_a = 0;
    let mut best_p = -1.0f32;
    for a in 0..num_actions {
        // Infoset 0: idx = (0 * max_actions + a) * num_hands + hand
        let idx = a * num_hands + hand;
        if idx < strategy.len() {
            let p = strategy[idx];
            if p > best_p {
                best_p = p;
                best_a = a;
            }
        }
    }
    (best_a, best_p)
}

/// Validate that a strategy vector is well-formed: sums to ~1.0 per hand per
/// infoset, no NaN/Inf, no negative values.
fn validate_strategy(strategy: &[f32], flat: &FlatTree) -> (bool, f32) {
    let num_hands = flat.num_hands;
    let max_actions = flat.max_actions();
    let mut max_deviation = 0.0f32;
    let mut valid = true;

    for iset in 0..flat.num_infosets {
        let n_actions = flat.infoset_num_actions[iset] as usize;
        for h in 0..num_hands {
            let mut total = 0.0f32;
            for a in 0..n_actions {
                let idx = (iset * max_actions + a) * num_hands + h;
                if idx >= strategy.len() {
                    continue;
                }
                let prob = strategy[idx];
                if !prob.is_finite() || prob < -1e-5 {
                    valid = false;
                }
                total += prob;
            }
            let dev = (total - 1.0).abs();
            if dev > max_deviation {
                max_deviation = dev;
            }
            if dev > 0.01 {
                valid = false;
            }
        }
    }

    (valid, max_deviation)
}

// ---------------------------------------------------------------------------
// Benchmark: 300 spots
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_batch_300_spots() {
    let num_spots = 300;
    let iterations = 500;
    let seed = 42u64;

    println!("\nGenerating {num_spots} random river spots...");
    let gen_start = Instant::now();
    let spots = generate_random_spots(num_spots, seed);
    let gen_time = gen_start.elapsed();
    println!(
        "Generation time: {:.1}ms",
        gen_time.as_secs_f64() * 1000.0
    );

    let config = BatchConfig::default();
    let gpu = GpuContext::new(0).unwrap();

    // --- Batch GPU solve ---
    println!("Building batch solver...");
    let build_start = Instant::now();
    let mut batch_solver = BatchGpuSolver::new(&gpu, &spots, &config).unwrap();
    let build_time = build_start.elapsed();
    println!(
        "Batch build time: {:.1}ms",
        build_time.as_secs_f64() * 1000.0
    );

    println!("Solving {num_spots} spots on GPU ({iterations} iterations)...");
    let gpu_start = Instant::now();
    let gpu_result = batch_solver.solve(iterations, None).unwrap();
    let gpu_time = gpu_start.elapsed();
    println!(
        "GPU solve time: {:.1}ms",
        gpu_time.as_secs_f64() * 1000.0
    );

    // --- Sequential CPU solve ---
    println!("Solving {num_spots} spots on CPU ({iterations} iterations)...");
    let cpu_start = Instant::now();
    let mut cpu_strategies: Vec<Vec<f32>> = Vec::with_capacity(num_spots);
    let mut cpu_num_hands: Vec<usize> = Vec::with_capacity(num_spots);
    let mut cpu_num_actions: Vec<usize> = Vec::with_capacity(num_spots);

    for spot in &spots {
        let mut game = build_game_from_spot(spot, &config).unwrap();
        solve(&mut game, iterations, 0.0, false);
        game.back_to_root();
        let player = game.current_player();
        let n_hands = game.num_private_hands(player);
        let actions = game.available_actions();
        let strat = game.strategy();
        cpu_strategies.push(strat);
        cpu_num_hands.push(n_hands);
        cpu_num_actions.push(actions.len());
    }
    let cpu_time = cpu_start.elapsed();
    println!("CPU solve time: {:.1}ms", cpu_time.as_secs_f64() * 1000.0);

    // --- Compute hand-per-spot statistics ---
    let hands_per_spot: Vec<usize> = cpu_num_hands.clone();
    let min_hands = hands_per_spot.iter().copied().min().unwrap_or(0);
    let max_hands = hands_per_spot.iter().copied().max().unwrap_or(0);
    let avg_hands = if hands_per_spot.is_empty() {
        0.0
    } else {
        hands_per_spot.iter().sum::<usize>() as f64 / hands_per_spot.len() as f64
    };

    // --- Compare strategies ---
    let mut max_diff_overall = 0.0f32;
    let mut dominant_agree = 0usize;
    let mut dominant_total = 0usize;
    let mut gpu_valid_count = 0usize;
    let mut gpu_invalid_count = 0usize;

    for (spot_idx, spot) in spots.iter().enumerate() {
        // Build a FlatTree to get GPU layout metadata
        let mut game = build_game_from_spot(spot, &config).unwrap();
        let flat = FlatTree::from_postflop_game(&mut game);
        let gpu_strat = &gpu_result.strategies[spot_idx];
        let cpu_strat = &cpu_strategies[spot_idx];

        // Validate GPU strategy
        let (valid, _max_dev) = validate_strategy(gpu_strat, &flat);
        if valid {
            gpu_valid_count += 1;
        } else {
            gpu_invalid_count += 1;
        }

        // Compare at root node (infoset 0) for the acting player
        let gpu_num_hands = flat.num_hands;
        let gpu_max_actions = flat.max_actions();
        let n_actions_root = flat.infoset_num_actions[0] as usize;
        let cpu_n_hands = cpu_num_hands[spot_idx];
        let cpu_n_actions = cpu_num_actions[spot_idx];

        // The number of actions should match
        if n_actions_root != cpu_n_actions {
            eprintln!(
                "Spot {spot_idx}: action count mismatch (GPU={n_actions_root}, CPU={cpu_n_actions})"
            );
            continue;
        }

        // Compare strategies hand-by-hand at the root
        let comparable_hands = gpu_num_hands.min(cpu_n_hands);
        for h in 0..comparable_hands {
            for a in 0..n_actions_root {
                let gpu_idx = (0 * gpu_max_actions + a) * gpu_num_hands + h;
                let cpu_idx = a * cpu_n_hands + h;
                if gpu_idx < gpu_strat.len() && cpu_idx < cpu_strat.len() {
                    let diff = (gpu_strat[gpu_idx] - cpu_strat[cpu_idx]).abs();
                    if diff > max_diff_overall {
                        max_diff_overall = diff;
                    }
                }
            }

            // Compare dominant actions
            let (gpu_dom, _) = gpu_dominant_action(gpu_strat, n_actions_root, gpu_max_actions, gpu_num_hands, h);
            // For CPU strategy: layout is [action * num_hands + hand]
            let mut cpu_best_a = 0;
            let mut cpu_best_p = -1.0f32;
            for a in 0..cpu_n_actions {
                let idx = a * cpu_n_hands + h;
                if idx < cpu_strat.len() && cpu_strat[idx] > cpu_best_p {
                    cpu_best_p = cpu_strat[idx];
                    cpu_best_a = a;
                }
            }

            dominant_total += 1;
            if gpu_dom == cpu_best_a {
                dominant_agree += 1;
            }
        }
    }

    // --- Print results ---
    println!("\n{}", "=".repeat(70));
    println!(
        "BATCH GPU BENCHMARK: {num_spots} spots x {iterations} iterations"
    );
    println!("{}", "=".repeat(70));
    println!(
        "GPU batch solve:    {:>10.1}ms ({:.0} spots/sec)",
        gpu_time.as_secs_f64() * 1000.0,
        num_spots as f64 / gpu_time.as_secs_f64()
    );
    println!(
        "CPU sequential:     {:>10.1}ms ({:.0} spots/sec)",
        cpu_time.as_secs_f64() * 1000.0,
        num_spots as f64 / cpu_time.as_secs_f64()
    );
    println!(
        "Speedup:            {:>10.2}x",
        cpu_time.as_secs_f64() / gpu_time.as_secs_f64()
    );
    println!(
        "Build overhead:     {:>10.1}ms",
        build_time.as_secs_f64() * 1000.0
    );
    println!(
        "GPU total (build+solve): {:>5.1}ms",
        (build_time + gpu_time).as_secs_f64() * 1000.0
    );
    println!(
        "Hands per spot:     min={min_hands}, max={max_hands}, avg={avg_hands:.1}"
    );
    println!("{}", "-".repeat(70));
    println!("Max strategy diff:           {max_diff_overall:.6}");
    println!("Dominant action agreement:   {dominant_agree}/{dominant_total}");
    println!(
        "Agreement rate:              {:.1}%",
        if dominant_total > 0 {
            dominant_agree as f64 / dominant_total as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "GPU strategies valid:        {gpu_valid_count}/{num_spots}"
    );
    if gpu_invalid_count > 0 {
        println!(
            "GPU strategies INVALID:      {gpu_invalid_count}/{num_spots}"
        );
    }
    println!("{}", "=".repeat(70));
}

// ---------------------------------------------------------------------------
// Smaller correctness test (not ignored — runs in CI)
// ---------------------------------------------------------------------------

#[test]
fn test_batch_correctness_10_spots() {
    let num_spots = 10;
    let iterations = 500;
    let seed = 123u64;

    let spots = generate_random_spots(num_spots, seed);
    let config = BatchConfig::default();
    let gpu = GpuContext::new(0).unwrap();

    // --- Batch GPU solve ---
    let mut batch_solver = BatchGpuSolver::new(&gpu, &spots, &config).unwrap();
    let gpu_result = batch_solver.solve(iterations, None).unwrap();
    assert_eq!(gpu_result.strategies.len(), num_spots);

    // --- Sequential CPU solve + compare ---
    for (spot_idx, spot) in spots.iter().enumerate() {
        let mut game = build_game_from_spot(spot, &config).unwrap();
        solve(&mut game, iterations, 0.0, false);
        game.back_to_root();

        let player = game.current_player();
        let cpu_n_hands = game.num_private_hands(player);
        let cpu_actions = game.available_actions();
        let cpu_n_actions = cpu_actions.len();
        let cpu_strat = game.strategy();

        // Build FlatTree for GPU layout metadata
        let mut game2 = build_game_from_spot(spot, &config).unwrap();
        let flat = FlatTree::from_postflop_game(&mut game2);
        let gpu_strat = &gpu_result.strategies[spot_idx];

        // Validate GPU strategy
        let (valid, max_dev) = validate_strategy(gpu_strat, &flat);
        assert!(
            valid,
            "Spot {spot_idx}: GPU strategy invalid (max deviation from 1.0: {max_dev:.6})"
        );

        // Compare at root
        let gpu_num_hands = flat.num_hands;
        let gpu_max_actions = flat.max_actions();
        let n_actions_root = flat.infoset_num_actions[0] as usize;

        assert_eq!(
            n_actions_root, cpu_n_actions,
            "Spot {spot_idx}: action count mismatch"
        );

        let comparable_hands = gpu_num_hands.min(cpu_n_hands);
        let mut max_diff = 0.0f32;

        for h in 0..comparable_hands {
            for a in 0..n_actions_root {
                let gpu_idx = (0 * gpu_max_actions + a) * gpu_num_hands + h;
                let cpu_idx = a * cpu_n_hands + h;
                if gpu_idx < gpu_strat.len() && cpu_idx < cpu_strat.len() {
                    let diff = (gpu_strat[gpu_idx] - cpu_strat[cpu_idx]).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
            }
        }

        eprintln!(
            "Spot {spot_idx}: board=[{},{},{},{},{}] pot={} stack={} hands={} actions={} max_diff={:.6}",
            spot.flop[0], spot.flop[1], spot.flop[2], spot.turn, spot.river,
            spot.pot, spot.effective_stack, gpu_num_hands, n_actions_root, max_diff
        );

        assert!(
            max_diff < 0.01,
            "Spot {spot_idx}: GPU vs CPU root strategy diff too large: {max_diff:.6}"
        );
    }
}
