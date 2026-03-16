//! Bucketed datagen pipeline: sample → solve → extract CFVs.
//!
//! This module wraps the batch solver and sampler into a single entry point
//! that can be called from the trainer CLI without version conflicts.

#[cfg(feature = "training")]
use range_solver::bet_size::BetSizeOptions;

/// Configuration for the bucketed datagen pipeline.
#[cfg(feature = "training")]
pub struct BucketedDatagenConfig {
    pub bucket_path: std::path::PathBuf,
    pub num_samples: usize,
    pub solve_iterations: u32,
    pub delay: u32,
    pub batch_size: usize,
    pub seed: u64,
    pub bet_sizes: BetSizeOptions,
    pub initial_stack: i32,
}

/// Run the bucketed datagen pipeline.
///
/// Samples batches of river situations, solves them on the GPU using
/// `BatchBucketedSolver`, and reports timing and CFV statistics.
///
/// This function manages the rand 0.8 RNG internally, avoiding
/// version conflicts with callers that use rand 0.9.
#[cfg(feature = "training")]
pub fn run_bucketed_datagen(config: &BucketedDatagenConfig) -> Result<(), String> {
    use std::time::Instant;

    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use crate::bucketed::batch::BatchBucketedSolver;
    use crate::bucketed::equity::BucketedBoardCache;
    use crate::bucketed::sampler::{sample_bucketed_situations, BucketedSamplingConfig};
    use crate::bucketed::tree::BucketedTree;
    use crate::gpu::GpuContext;
    use poker_solver_core::blueprint_v2::bucket_file::BucketFile;

    let bf = BucketFile::load(&config.bucket_path)
        .map_err(|e| format!("Failed to load bucket file: {e}"))?;
    let num_buckets = bf.header.bucket_count as usize;
    let cache = BucketedBoardCache::new(&bf);

    eprintln!("GPU Bucketed Datagen");
    eprintln!(
        "  Buckets: {} ({} buckets)",
        config.bucket_path.display(),
        num_buckets
    );
    eprintln!(
        "  Samples: {}, batch_size: {}",
        config.num_samples, config.batch_size
    );
    eprintln!(
        "  Solve iters: {}, delay: {}",
        config.solve_iterations, config.delay
    );
    eprintln!(
        "  Stack: {}, seed: {}",
        config.initial_stack, config.seed
    );
    eprintln!();

    let gpu = GpuContext::new(0).map_err(|e| format!("Failed to initialize GPU: {e}"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);

    // Build reference tree from first board (topology only)
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
        starting_pot: 100,
        effective_stack: config.initial_stack,
        river_bet_sizes: [config.bet_sizes.clone(), config.bet_sizes.clone()],
        ..Default::default()
    };

    let ref_action_tree = range_solver::action_tree::ActionTree::new(ref_tree_config)
        .map_err(|e| format!("Failed to build action tree: {e}"))?;
    let mut ref_game =
        range_solver::PostFlopGame::with_config(ref_card_config, ref_action_tree)
            .map_err(|e| format!("Failed to build game: {e}"))?;
    ref_game.allocate_memory(false);

    let ref_tree = BucketedTree::from_postflop_game(&mut ref_game, &bf, &cache, num_buckets);

    eprintln!(
        "Reference tree: {} nodes, {} infosets, {} levels",
        ref_tree.num_nodes(),
        ref_tree.num_infosets,
        ref_tree.num_levels(),
    );

    let sampling_config = BucketedSamplingConfig {
        pot_intervals: vec![[4, 50], [50, 100], [100, 150], [150, 200]],
        spr_intervals: Some(vec![
            [0.0, 0.5],
            [0.5, 1.5],
            [1.5, 4.0],
            [4.0, 8.0],
            [8.0, 50.0],
        ]),
        initial_stack: config.initial_stack,
    };

    let mut total_solved = 0usize;
    let mut total_solve_ms: u64 = 0;
    let mut total_setup_ms: u64 = 0;
    let mut batch_count = 0usize;

    // Reservoir statistics
    let input_size = 2 * num_buckets + 1;
    let output_size = 2 * num_buckets;
    let mut reservoir_count = 0usize;

    let overall_start = Instant::now();

    while total_solved < config.num_samples {
        let this_batch = config.batch_size.min(config.num_samples - total_solved);

        // Sample situations
        let situations = sample_bucketed_situations(
            &sampling_config,
            &bf,
            &cache,
            num_buckets,
            5,
            this_batch,
            &mut rng,
        );

        if situations.is_empty() {
            eprintln!("  Warning: no valid situations sampled, retrying...");
            continue;
        }

        let setup_start = Instant::now();
        let mut batch_solver =
            BatchBucketedSolver::new(&gpu, &ref_tree, &situations, &bf, &cache)
                .map_err(|e| format!("Failed to create batch solver: {e}"))?;
        let setup_ms = setup_start.elapsed().as_millis() as u64;
        total_setup_ms += setup_ms;

        let solve_start = Instant::now();
        let result = batch_solver
            .solve_with_cfvs(config.solve_iterations, config.delay)
            .map_err(|e| format!("Batch solve failed: {e}"))?;
        let solve_ms = solve_start.elapsed().as_millis() as u64;
        total_solve_ms += solve_ms;

        reservoir_count += situations.len();
        total_solved += situations.len();
        batch_count += 1;

        // Report progress
        let oop_mean: f32 =
            result.cfvs_oop.iter().sum::<f32>() / result.cfvs_oop.len().max(1) as f32;
        let ip_mean: f32 =
            result.cfvs_ip.iter().sum::<f32>() / result.cfvs_ip.len().max(1) as f32;

        eprintln!(
            "  Batch {:3}: {} spots, setup={}ms, solve={}ms, OOP_mean={:.2}, IP_mean={:.2} [{}/{}]",
            batch_count,
            situations.len(),
            setup_ms,
            solve_ms,
            oop_mean,
            ip_mean,
            total_solved,
            config.num_samples,
        );
    }

    let total_elapsed = overall_start.elapsed();
    let samples_per_sec = total_solved as f64 / total_elapsed.as_secs_f64();
    let reservoir_mb =
        reservoir_count as f64 * (input_size + output_size) as f64 * 4.0 / 1_048_576.0;

    eprintln!();
    eprintln!("Datagen complete:");
    eprintln!("  Total samples: {total_solved}");
    eprintln!("  Total batches: {batch_count}");
    eprintln!("  Total time: {:.1}s", total_elapsed.as_secs_f64());
    eprintln!("  Throughput: {:.1} samples/sec", samples_per_sec);
    eprintln!(
        "  Avg setup: {:.0}ms",
        total_setup_ms as f64 / batch_count.max(1) as f64
    );
    eprintln!(
        "  Avg solve: {:.0}ms",
        total_solve_ms as f64 / batch_count.max(1) as f64
    );
    eprintln!(
        "  Reservoir: {} records, {:.1} MB (if stored)",
        reservoir_count, reservoir_mb
    );

    Ok(())
}
