//! Evaluation command for the bucketed GPU solver.
//!
//! Samples random river situations, solves them in bucket space on the GPU,
//! and reports CFV statistics. Used to verify the bucketed solver produces
//! sensible results before integrating neural network training.

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use crate::bucketed::equity::BucketedBoardCache;
#[cfg(feature = "cuda")]
use crate::bucketed::solver::BucketedGpuSolver;
#[cfg(feature = "cuda")]
use crate::bucketed::tree::BucketedTree;
#[cfg(feature = "cuda")]
use crate::gpu::GpuContext;
#[cfg(feature = "cuda")]
use poker_solver_core::blueprint_v2::bucket_file::BucketFile;
#[cfg(feature = "cuda")]
use range_solver::bet_size::BetSizeOptions;

/// Configuration for bucketed solver evaluation.
#[cfg(feature = "cuda")]
pub struct BucketedEvalConfig {
    pub bucket_path: std::path::PathBuf,
    pub num_spots: usize,
    pub solve_iterations: u32,
    pub delay: u32,
    pub seed: u64,
    pub bet_sizes: BetSizeOptions,
    pub initial_stack: i32,
}

/// Result for a single evaluated spot.
#[cfg(feature = "cuda")]
struct SpotResult {
    solve_time_ms: u64,
    strategy_valid: bool,
}

/// Run the bucketed solver evaluation.
///
/// For each of `config.num_spots` random river situations:
/// 1. Sample a random river board from the bucket file
/// 2. Build a bucketed tree
/// 3. Solve with BucketedGpuSolver
/// 4. Verify strategy and extract CFV statistics
#[cfg(feature = "cuda")]
pub fn eval_bucketed_solver(config: &BucketedEvalConfig) -> Result<(), String> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let bf = BucketFile::load(&config.bucket_path)
        .map_err(|e| format!("Failed to load bucket file: {e}"))?;
    let num_buckets = bf.header.bucket_count as usize;
    let cache = BucketedBoardCache::new(&bf);

    eprintln!("Evaluating bucketed solver ({num_buckets} buckets)");
    eprintln!("  Buckets: {}", config.bucket_path.display());
    eprintln!(
        "  Solving {} spots at {} iterations (delay={})...",
        config.num_spots, config.solve_iterations, config.delay
    );
    eprintln!();

    let gpu = GpuContext::new(0).map_err(|e| format!("Failed to initialize GPU: {e}"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    let mut results: Vec<SpotResult> = Vec::new();

    for spot_idx in 0..config.num_spots {
        // Pick a random board from the bucket file
        let board_idx = rand::Rng::gen_range(&mut rng, 0..bf.header.board_count);
        let packed = bf.boards[board_idx as usize];
        let board_cards_rs = packed.to_cards(5);
        let board_u8: Vec<u8> = board_cards_rs
            .iter()
            .map(|c| {
                poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id(*c)
            })
            .collect();

        let flop = [board_u8[0], board_u8[1], board_u8[2]];
        let turn = board_u8[3];
        let river = board_u8[4];

        // Random pot (stratified)
        let pot = rand::Rng::gen_range(&mut rng, 20..200);
        let stack = config.initial_stack;

        // Build the game and tree
        let card_config = range_solver::card::CardConfig {
            range: [
                range_solver::range::Range::ones(),
                range_solver::range::Range::ones(),
            ],
            flop,
            turn,
            river,
        };

        let tree_config = range_solver::action_tree::TreeConfig {
            initial_state: range_solver::action_tree::BoardState::River,
            starting_pot: pot,
            effective_stack: stack,
            river_bet_sizes: [config.bet_sizes.clone(), config.bet_sizes.clone()],
            ..Default::default()
        };

        let action_tree = range_solver::action_tree::ActionTree::new(tree_config)
            .map_err(|e| format!("Failed to build action tree: {e}"))?;
        let mut game = range_solver::PostFlopGame::with_config(card_config, action_tree)
            .map_err(|e| format!("Failed to build game: {e}"))?;
        game.allocate_memory(false);

        let tree = BucketedTree::from_postflop_game(&mut game, &bf, &cache, num_buckets);

        // Uniform initial reach
        let initial_reach = vec![1.0f32; num_buckets];

        let mut solver = BucketedGpuSolver::new(&gpu, &tree, &initial_reach, &initial_reach)
            .map_err(|e| format!("Failed to create solver: {e}"))?;

        let start = Instant::now();
        let solve_result = solver
            .solve(config.solve_iterations, config.delay)
            .map_err(|e| format!("Solver failed: {e}"))?;
        let solve_time_ms = start.elapsed().as_millis() as u64;

        // Verify strategy
        let strategy = &solve_result.strategy;
        let nb = num_buckets;
        let ma = tree.max_actions();
        let ni = tree.num_infosets;
        let mut strategy_valid = true;

        for iset in 0..ni {
            let n_actions = tree.infoset_num_actions[iset] as usize;
            for bucket in 0..nb {
                let mut sum = 0.0f32;
                for a in 0..n_actions {
                    let idx = (iset * ma + a) * nb + bucket;
                    sum += strategy[idx];
                }
                if (sum - 1.0).abs() > 0.05 {
                    strategy_valid = false;
                }
            }
        }

        let board_str = format_board(&board_u8);
        let num_actions = tree.max_actions();

        eprintln!(
            "  Spot {:3}/{}: board={}, pot={}, {} actions, strategy {}, solve={}ms",
            spot_idx + 1,
            config.num_spots,
            board_str,
            pot,
            num_actions,
            if strategy_valid { "valid" } else { "INVALID" },
            solve_time_ms,
        );

        results.push(SpotResult {
            solve_time_ms,
            strategy_valid,
        });
    }

    // Summary
    eprintln!();
    let valid_count = results.iter().filter(|r| r.strategy_valid).count();
    let avg_time: f64 = results.iter().map(|r| r.solve_time_ms as f64).sum::<f64>()
        / results.len().max(1) as f64;

    eprintln!("Summary:");
    eprintln!("  All {} spots processed", results.len());
    eprintln!("  Avg solve time: {:.0}ms", avg_time);
    eprintln!("  Strategy valid: {}/{}", valid_count, results.len());

    if valid_count < results.len() {
        eprintln!("  WARNING: {} spots had invalid strategies", results.len() - valid_count);
    }

    Ok(())
}

/// Evaluate the bucketed solver using the batch solver for CFV extraction.
///
/// This version uses `BatchBucketedSolver` to solve spots and extract
/// root CFVs for both players, enabling anti-symmetry verification.
#[cfg(feature = "training")]
pub fn eval_bucketed_solver_with_cfvs(config: &BucketedEvalConfig) -> Result<(), String> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::batch::BatchBucketedSolver;
    use super::sampler::BucketedSituation;

    let bf = BucketFile::load(&config.bucket_path)
        .map_err(|e| format!("Failed to load bucket file: {e}"))?;
    let num_buckets = bf.header.bucket_count as usize;
    let cache = BucketedBoardCache::new(&bf);

    eprintln!("Evaluating bucketed solver with CFV extraction ({num_buckets} buckets)");
    eprintln!("  Buckets: {}", config.bucket_path.display());
    eprintln!(
        "  Solving {} spots at {} iterations (delay={})...",
        config.num_spots, config.solve_iterations, config.delay
    );
    eprintln!();

    let gpu = GpuContext::new(0).map_err(|e| format!("Failed to initialize GPU: {e}"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    // Build a reference tree from the first board (topology only)
    let first_packed = bf.boards[0];
    let first_board_rs = first_packed.to_cards(5);
    let first_board_u8: Vec<u8> = first_board_rs
        .iter()
        .map(|c| {
            poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id(*c)
        })
        .collect();

    let ref_flop = [first_board_u8[0], first_board_u8[1], first_board_u8[2]];
    let ref_turn = first_board_u8[3];
    let ref_river = first_board_u8[4];
    let ref_pot = 100;

    let ref_card_config = range_solver::card::CardConfig {
        range: [
            range_solver::range::Range::ones(),
            range_solver::range::Range::ones(),
        ],
        flop: ref_flop,
        turn: ref_turn,
        river: ref_river,
    };

    let ref_tree_config = range_solver::action_tree::TreeConfig {
        initial_state: range_solver::action_tree::BoardState::River,
        starting_pot: ref_pot,
        effective_stack: config.initial_stack,
        river_bet_sizes: [config.bet_sizes.clone(), config.bet_sizes.clone()],
        ..Default::default()
    };

    let ref_action_tree = range_solver::action_tree::ActionTree::new(ref_tree_config)
        .map_err(|e| format!("Failed to build reference action tree: {e}"))?;
    let mut ref_game = range_solver::PostFlopGame::with_config(ref_card_config, ref_action_tree)
        .map_err(|e| format!("Failed to build reference game: {e}"))?;
    ref_game.allocate_memory(false);

    let ref_tree = BucketedTree::from_postflop_game(&mut ref_game, &bf, &cache, num_buckets);

    // Solve spots one at a time using batch solver (batch_size=1)
    // This gives us CFV extraction.
    let mut total_solve_ms: u64 = 0;
    let mut all_valid = true;

    for spot_idx in 0..config.num_spots {
        let board_idx = rand::Rng::gen_range(&mut rng, 0..bf.header.board_count);
        let packed = bf.boards[board_idx as usize];
        let board_cards_rs = packed.to_cards(5);
        let board_u8: Vec<u8> = board_cards_rs
            .iter()
            .map(|c| {
                poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id(*c)
            })
            .collect();

        let board_arr: [u8; 5] = [
            board_u8[0], board_u8[1], board_u8[2], board_u8[3], board_u8[4],
        ];

        let pot = rand::Rng::gen_range(&mut rng, 20..200);
        let initial_reach = vec![1.0f32; num_buckets];

        let situation = BucketedSituation {
            board: board_arr,
            board_idx,
            pot,
            effective_stack: config.initial_stack,
            oop_reach: initial_reach.clone(),
            ip_reach: initial_reach.clone(),
        };

        let start = Instant::now();
        let mut batch_solver = BatchBucketedSolver::new(
            &gpu,
            &ref_tree,
            &[situation],
            &bf,
            &cache,
        )
        .map_err(|e| format!("Failed to create batch solver: {e}"))?;

        let result = batch_solver
            .solve_with_cfvs(config.solve_iterations, config.delay)
            .map_err(|e| format!("Batch solve failed: {e}"))?;
        let solve_ms = start.elapsed().as_millis() as u64;
        total_solve_ms += solve_ms;

        let cfvs_oop = &result.cfvs_oop;
        let cfvs_ip = &result.cfvs_ip;

        let oop_min = cfvs_oop.iter().cloned().fold(f32::INFINITY, f32::min);
        let oop_max = cfvs_oop.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let oop_mean: f32 = cfvs_oop.iter().sum::<f32>() / num_buckets as f32;

        let ip_min = cfvs_ip.iter().cloned().fold(f32::INFINITY, f32::min);
        let ip_max = cfvs_ip.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ip_mean: f32 = cfvs_ip.iter().sum::<f32>() / num_buckets as f32;

        // Check approximate anti-symmetry: weighted sum of (oop + ip) ~ 0
        let sum_check: f32 = cfvs_oop
            .iter()
            .zip(cfvs_ip.iter())
            .map(|(a, b)| (a + b).abs())
            .sum::<f32>()
            / num_buckets as f32;

        let board_str = format_board(&board_u8);
        let valid = sum_check < 50.0; // generous tolerance for bucket approximation
        if !valid {
            all_valid = false;
        }

        eprintln!(
            "  Spot {:3}/{}: board={}, pot={}, solve={}ms",
            spot_idx + 1,
            config.num_spots,
            board_str,
            pot,
            solve_ms,
        );
        eprintln!(
            "    OOP cfv: [{:.2}, {:.2}] mean={:.2}",
            oop_min, oop_max, oop_mean,
        );
        eprintln!(
            "    IP  cfv: [{:.2}, {:.2}] mean={:.2}",
            ip_min, ip_max, ip_mean,
        );
        eprintln!(
            "    |OOP+IP| avg: {:.4}{}",
            sum_check,
            if valid { "" } else { " [SUSPICIOUS]" },
        );
    }

    let avg_ms = total_solve_ms as f64 / config.num_spots.max(1) as f64;
    eprintln!();
    eprintln!("Summary:");
    eprintln!("  All {} spots solved", config.num_spots);
    eprintln!("  Avg solve time: {:.0}ms", avg_ms);
    eprintln!(
        "  CFVs consistent: {}",
        if all_valid { "yes" } else { "NO — some spots look off" }
    );

    Ok(())
}

/// Evaluate the bucketed solver vs the concrete CPU range-solver.
///
/// For each random river spot, solves the same position with both:
/// 1. CPU range-solver (concrete, exact — all valid combos)
/// 2. Bucketed GPU solver (num_buckets buckets)
///
/// Then compares aggregate action frequencies at the root:
/// - CPU: for each action, average strategy weight across all hands
/// - Bucketed: for each action, average strategy weight across all buckets
/// - Report per-action frequencies and the max absolute difference
#[cfg(feature = "cuda")]
pub fn eval_bucketed_vs_concrete(config: &BucketedEvalConfig) -> Result<(), String> {
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let bf = BucketFile::load(&config.bucket_path)
        .map_err(|e| format!("Failed to load bucket file: {e}"))?;
    let num_buckets = bf.header.bucket_count as usize;
    let cache = BucketedBoardCache::new(&bf);

    eprintln!(
        "Evaluating bucketed solver ({num_buckets} buckets) vs CPU range-solver"
    );
    eprintln!("  Buckets: {}", config.bucket_path.display());
    eprintln!(
        "  Solving {} spots at {} iterations...",
        config.num_spots, config.solve_iterations
    );
    eprintln!();

    let gpu = GpuContext::new(0).map_err(|e| format!("Failed to initialize GPU: {e}"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    let mut max_diffs: Vec<f64> = Vec::new();

    for spot_idx in 0..config.num_spots {
        // Pick a random board from the bucket file
        let board_idx = rand::Rng::gen_range(&mut rng, 0..bf.header.board_count);
        let packed = bf.boards[board_idx as usize];
        let board_cards_rs = packed.to_cards(5);
        let board_u8: Vec<u8> = board_cards_rs
            .iter()
            .map(|c| {
                poker_solver_core::blueprint_v2::full_depth_solver::rs_poker_card_to_id(*c)
            })
            .collect();

        let flop = [board_u8[0], board_u8[1], board_u8[2]];
        let turn = board_u8[3];
        let river = board_u8[4];

        // Random pot (stratified)
        let pot = rand::Rng::gen_range(&mut rng, 20..200);
        let stack = config.initial_stack;

        // ---- Build PostFlopGame (shared by both CPU solve and bucketed tree) ----
        let card_config = range_solver::card::CardConfig {
            range: [
                range_solver::range::Range::ones(),
                range_solver::range::Range::ones(),
            ],
            flop,
            turn,
            river,
        };

        let tree_config = range_solver::action_tree::TreeConfig {
            initial_state: range_solver::action_tree::BoardState::River,
            starting_pot: pot,
            effective_stack: stack,
            river_bet_sizes: [config.bet_sizes.clone(), config.bet_sizes.clone()],
            ..Default::default()
        };

        let action_tree = range_solver::action_tree::ActionTree::new(tree_config)
            .map_err(|e| format!("Failed to build action tree: {e}"))?;
        let mut game = range_solver::PostFlopGame::with_config(card_config, action_tree)
            .map_err(|e| format!("Failed to build game: {e}"))?;
        game.allocate_memory(false);

        let start = Instant::now();

        // ---- CPU range-solver ----
        range_solver::solve(&mut game, config.solve_iterations, 0.0, false);
        game.back_to_root();
        let cpu_strategy = game.strategy();
        let cpu_actions = game.available_actions();
        let cpu_player = game.current_player();
        let cpu_num_hands = game.private_cards(cpu_player).len();
        let cpu_num_actions = cpu_actions.len();

        // Compute CPU aggregate action frequencies (uniform weight per hand)
        let mut cpu_action_freq = vec![0.0f64; cpu_num_actions];
        for a in 0..cpu_num_actions {
            let mut sum = 0.0f64;
            for h in 0..cpu_num_hands {
                sum += cpu_strategy[a * cpu_num_hands + h] as f64;
            }
            cpu_action_freq[a] = sum / cpu_num_hands as f64;
        }

        // ---- Bucketed GPU solver ----
        let tree = BucketedTree::from_postflop_game(&mut game, &bf, &cache, num_buckets);
        let initial_reach = vec![1.0f32; num_buckets];

        let mut solver = BucketedGpuSolver::new(&gpu, &tree, &initial_reach, &initial_reach)
            .map_err(|e| format!("Failed to create solver: {e}"))?;

        let solve_result = solver
            .solve(config.solve_iterations, config.delay)
            .map_err(|e| format!("Solver failed: {e}"))?;

        let solve_time_ms = start.elapsed().as_millis() as u64;

        // Extract bucketed strategy at root (infoset 0)
        let bucketed_strategy = &solve_result.strategy;
        let root_num_actions = tree.infoset_num_actions[0] as usize;

        // Compute bucketed aggregate action frequencies (uniform weight per bucket)
        let mut bucketed_action_freq = vec![0.0f64; root_num_actions];
        for a in 0..root_num_actions {
            let mut sum = 0.0f64;
            for b in 0..num_buckets {
                // Root is infoset 0: strategy[(0 * ma + a) * nb + b]
                let idx = a * num_buckets + b;
                sum += bucketed_strategy[idx] as f64;
            }
            bucketed_action_freq[a] = sum / num_buckets as f64;
        }

        // Compare
        let num_actions_compare = cpu_num_actions.min(root_num_actions);
        let mut max_diff = 0.0f64;
        for a in 0..num_actions_compare {
            let diff = (cpu_action_freq[a] - bucketed_action_freq[a]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        let board_str = format_board(&board_u8);
        eprintln!(
            "  Spot {:3}/{}: board={}, pot={}, solve={}ms",
            spot_idx + 1,
            config.num_spots,
            board_str,
            pot,
            solve_time_ms,
        );

        // Print per-action frequencies
        let mut cpu_parts = Vec::new();
        let mut bkt_parts = Vec::new();
        for a in 0..num_actions_compare {
            let action_name = format!("{}", cpu_actions[a]);
            cpu_parts.push(format!("{} {:.1}%", action_name, cpu_action_freq[a] * 100.0));
            bkt_parts.push(format!(
                "{} {:.1}%",
                action_name,
                bucketed_action_freq[a] * 100.0
            ));
        }
        eprintln!("    CPU strategy:      {}", cpu_parts.join("  "));
        eprintln!("    Bucketed strategy: {}", bkt_parts.join("  "));
        eprintln!("    Max action freq diff: {:.1}%", max_diff * 100.0);

        max_diffs.push(max_diff);
    }

    // Summary
    eprintln!();
    let avg_max_diff: f64 =
        max_diffs.iter().sum::<f64>() / max_diffs.len().max(1) as f64;
    let worst_diff = max_diffs
        .iter()
        .cloned()
        .fold(0.0f64, f64::max);

    eprintln!("Summary:");
    eprintln!("  All {} spots solved successfully", max_diffs.len());
    eprintln!("  Avg max action freq diff: {:.1}%", avg_max_diff * 100.0);
    eprintln!("  Worst max action freq diff: {:.1}%", worst_diff * 100.0);

    Ok(())
}

/// Format board cards as a human-readable string (e.g. "Qs Jh 2c 8d 3s").
#[cfg(feature = "cuda")]
fn format_board(board_u8: &[u8]) -> String {
    let ranks = [
        '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A',
    ];
    let suits = ['c', 'd', 'h', 's'];
    board_u8
        .iter()
        .map(|&c| {
            let r = (c / 4) as usize;
            let s = (c % 4) as usize;
            format!("{}{}", ranks.get(r).unwrap_or(&'?'), suits.get(s).unwrap_or(&'?'))
        })
        .collect::<Vec<_>>()
        .join(" ")
}
