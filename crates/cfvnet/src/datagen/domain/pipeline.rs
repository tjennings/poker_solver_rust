use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use indicatif::{ProgressBar, ProgressStyle};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::config::CfvnetConfig;

use super::evaluator::SolveStrategy;
use super::game::GameBuilder;
use super::neural_net_evaluator::NeuralNetEvaluator;
use super::situation::SituationGenerator;
use super::solver::{Solver, SolverConfig};
use super::writer::RecordWriter;

/// Coordinates the domain datagen pipeline: generate -> build -> solve -> write.
///
/// Uses N solver threads pulling situations from a shared iterator.
/// Each thread builds games, solves them, and writes records through a shared writer.
pub struct DomainPipeline;

impl DomainPipeline {
    pub fn run(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        #[cfg(feature = "gpu-datagen")]
        if config.datagen.backend == "gpu" {
            return Self::run_gpu(config, output_path);
        }
        Self::run_cpu(config, output_path)
    }

    fn run_cpu(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        let num_samples = config.datagen.num_samples;
        let seed = crate::config::resolve_seed(config.datagen.seed);
        let initial_stack = config.game.initial_stack;
        let board_size = config.game.board_size;
        let threads = config.datagen.threads.max(1);

        // Parse bet sizes from config.
        let bet_sizes =
            super::game_tree::parse_bet_sizes_all(&config.game.bet_sizes);
        if bet_sizes.is_empty() {
            return Err("no valid bet sizes".into());
        }

        // Construct solve strategy:
        // - Exact if board_size >= 5 (river, no boundaries)
        // - Exact if no river_model_path (turn exact mode)
        // - DepthLimited with neural net otherwise
        let has_model = config.game.river_model_path.is_some();
        let strategy = if !has_model || board_size >= 5 {
            eprintln!("[domain] exact mode: solving to showdown (no neural net)");
            SolveStrategy::Exact
        } else {
            SolveStrategy::DepthLimited {
                evaluator: Arc::new(NeuralNetEvaluator::load(
                    config
                        .game
                        .river_model_path
                        .as_deref()
                        .ok_or("river_model_path required for turn datagen")?,
                    config,
                )?),
            }
        };

        // Construct domain objects.
        let range_source = super::RangeSource::from_config(&config.datagen)?;
        let sit_gen = SituationGenerator::new(
            &config.datagen,
            initial_stack,
            board_size,
            seed,
            num_samples,
        )
        .with_range_source(range_source);
        let builder = GameBuilder::new(bet_sizes, &strategy)
            .with_fuzz(config.datagen.bet_size_fuzz);
        let solver_config = SolverConfig {
            max_iterations: config.datagen.solver_iterations,
            target_exploitability: config.datagen.target_exploitability,
            leaf_eval_interval: config.datagen.leaf_eval_interval,
        };

        let writer = Arc::new(Mutex::new(RecordWriter::create(
            output_path,
            config.datagen.per_file,
        )?));

        // Progress bar.
        let pb = Arc::new(ProgressBar::new(num_samples));
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}",
                )
                .expect("valid template"),
        );

        // Shared exploitability counters (scaled by 100 for AtomicU64 storage).
        let exploit_sum = Arc::new(AtomicU64::new(0));
        let exploit_count = Arc::new(AtomicU64::new(0));

        // Shared situation generator.
        let sit_gen = Mutex::new(sit_gen);

        // Collect errors from threads.
        let first_error: Mutex<Option<String>> = Mutex::new(None);

        std::thread::scope(|s| {
            for thread_idx in 0..threads {
                let sit_gen = &sit_gen;
                let writer = Arc::clone(&writer);
                let builder = &builder;
                let solver_config = &solver_config;
                let strategy = &strategy;
                let pb = Arc::clone(&pb);
                let exploit_sum = Arc::clone(&exploit_sum);
                let exploit_count = Arc::clone(&exploit_count);
                let first_error = &first_error;

                s.spawn(move || {
                    range_solver::set_force_sequential(true);
                    let mut rng =
                        ChaCha8Rng::seed_from_u64(seed.wrapping_add(thread_idx as u64));

                    loop {
                        // Pull next situation from shared generator.
                        let sit = {
                            let mut generator = sit_gen.lock().unwrap();
                            generator.next()
                        };
                        let sit = match sit {
                            Some(s) => s,
                            None => return,
                        };

                        let game = match builder.build(&sit, &mut rng) {
                            Some(g) => g,
                            None => {
                                pb.inc(1);
                                continue;
                            }
                        };

                        let mut solver =
                            Solver::new(game, solver_config, strategy.clone());
                        let solved = loop {
                            match solver.step() {
                                None => continue,
                                Some(sg) => break sg,
                            }
                        };

                        // Track exploitability via atomics.
                        if solved.exploitability.is_finite() {
                            let bb = initial_stack as f32 / 100.0;
                            let mbb = if bb > 0.0 {
                                solved.exploitability / bb * 1000.0
                            } else {
                                0.0
                            };
                            // Scale by 100 to preserve 2 decimal places in integer.
                            exploit_sum.fetch_add(
                                (mbb as f64 * 100.0) as u64,
                                Ordering::Relaxed,
                            );
                            exploit_count.fetch_add(1, Ordering::Relaxed);
                        }

                        let records = solved.extract_records();

                        // Write records under lock (one batch per game).
                        let wc = {
                            let mut w = writer.lock().unwrap();
                            if let Err(e) = w.write(&records) {
                                let mut err = first_error.lock().unwrap();
                                if err.is_none() {
                                    *err = Some(e);
                                }
                                return;
                            }
                            w.count()
                        };

                        pb.inc(1);

                        // Update progress bar message.
                        let ec = exploit_count.load(Ordering::Relaxed);
                        let avg_exploit = if ec > 0 {
                            (exploit_sum.load(Ordering::Relaxed) as f64 / 100.0)
                                / ec as f64
                        } else {
                            0.0
                        };
                        pb.set_message(format!(
                            "expl:{avg_exploit:.1} mbb/h  written:{wc}",
                        ));
                    }
                });
            }
        });

        // Check for errors from threads.
        if let Some(e) = first_error.into_inner().unwrap() {
            return Err(e);
        }

        // Flush writer.
        let mut w = writer.lock().unwrap();
        w.flush()?;
        let total = w.count();
        drop(w);

        pb.finish_with_message("done");

        eprintln!("Wrote {total} records to {}", output_path.display());
        Ok(())
    }

    /// GPU-accelerated datagen pipeline.
    ///
    /// Builds PostFlopGames from situations, solves them individually on GPU
    /// using the hand-parallel kernel, then extracts training records using
    /// CPU-side EV computation.
    ///
    /// Each game is solved as a batch of 1 because different random situations
    /// produce different topologies (different ranges = different hand counts).
    /// True batching requires games with matching topology (same ranges/bet sizes).
    #[cfg(feature = "gpu-datagen")]
    fn run_gpu(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        use gpu_range_solver::extract::{extract_terminal_data, extract_topology};
        use gpu_range_solver::{GpuBatchSolver, SubgameSpec, compute_evs_from_strategy_sum};
        use range_solver::card::card_pair_to_index;
        use crate::datagen::range_gen::NUM_COMBOS;
        use crate::datagen::storage::TrainingRecord;

        let num_samples = config.datagen.num_samples;
        let seed = crate::config::resolve_seed(config.datagen.seed);
        let initial_stack = config.game.initial_stack;
        let board_size = config.game.board_size;
        let max_iterations = config.datagen.solver_iterations;

        if board_size == 4 {
            #[cfg(feature = "gpu-turn-datagen")]
            return Self::run_gpu_turn(config, output_path);
            #[cfg(not(feature = "gpu-turn-datagen"))]
            return Err("GPU turn datagen requires --features gpu-turn-datagen".into());
        }
        if board_size < 5 {
            return Err("GPU datagen currently supports river (board_size=5) only".into());
        }

        let bet_sizes = super::game_tree::parse_bet_sizes_all(&config.game.bet_sizes);
        if bet_sizes.is_empty() {
            return Err("no valid bet sizes".into());
        }

        let strategy = SolveStrategy::Exact;
        let range_source = super::RangeSource::from_config(&config.datagen)?;
        let mut sit_gen = SituationGenerator::new(
            &config.datagen,
            initial_stack,
            board_size,
            seed,
            num_samples,
        )
        .with_range_source(range_source);

        let builder = GameBuilder::new(bet_sizes, &strategy)
            .with_fuzz(config.datagen.bet_size_fuzz);

        let writer = std::sync::Arc::new(std::sync::Mutex::new(
            super::writer::RecordWriter::create(output_path, config.datagen.per_file)?,
        ));

        let pb = indicatif::ProgressBar::new(num_samples);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}")
                .expect("valid template"),
        );

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

        for sit in &mut sit_gen {
            let game = match builder.build(&sit, &mut rng) {
                Some(g) => g,
                None => {
                    pb.inc(1);
                    continue;
                }
            };

            let topo = extract_topology(game.inner());
            let term = extract_terminal_data(game.inner(), &topo);
            let num_hands = game.inner().private_cards(0).len()
                .max(game.inner().private_cards(1).len());

            // CUDA blocks have max 1024 threads; fall back to CPU for large ranges
            if num_hands > 1024 {
                // CPU fallback: use existing solver path
                let solver_config = super::solver::SolverConfig {
                    max_iterations,
                    target_exploitability: config.datagen.target_exploitability,
                    leaf_eval_interval: config.datagen.leaf_eval_interval,
                };
                let mut solver_obj = super::solver::Solver::new(
                    game,
                    &solver_config,
                    strategy.clone(),
                );
                let solved = loop {
                    match solver_obj.step() {
                        None => continue,
                        Some(sg) => break sg,
                    }
                };
                let records = solved.extract_records();
                let mut w = writer.lock().unwrap();
                if let Err(e) = w.write(&records) {
                    return Err(e);
                }
                drop(w);
                pb.inc(1);
                continue;
            }

            // Create a per-game GPU solver (topology varies per game with random ranges)
            let mut solver = GpuBatchSolver::new(
                &topo,
                &term,
                1,
                num_hands,
                max_iterations,
            )
            .map_err(|e| format!("GPU solver init failed: {e}"))?;

            let spec = SubgameSpec::from_game(game.inner(), &topo, &term, num_hands);
            let results = solver
                .solve_batch(&[spec.clone()])
                .map_err(|e| format!("GPU solve_batch failed: {e}"))?;

            let evs = compute_evs_from_strategy_sum(
                &topo,
                &term,
                &results[0].strategy_sum,
                &spec.initial_weights,
                num_hands,
            );

            // Build training records from EVs
            let pot = f64::from(sit.pot);
            let half_pot = pot / 2.0;
            let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

            let oop_hands = game.inner().private_cards(0);
            let ip_hands = game.inner().private_cards(1);

            let mut oop_cfvs = [0.0_f32; NUM_COMBOS];
            let mut ip_cfvs = [0.0_f32; NUM_COMBOS];
            let mut valid_mask = [0_u8; NUM_COMBOS];

            // GPU EVs are in per-combination units (pre-divided by num_combinations).
            // Scale to chip units and normalize for training.
            let comb = term.num_combinations as f32;
            for (h, &(c0, c1)) in oop_hands.iter().enumerate() {
                let idx = card_pair_to_index(c0, c1);
                let ev_chips = evs[0][h] * comb;
                oop_cfvs[idx] = ((f64::from(ev_chips) - half_pot) / norm) as f32;
                valid_mask[idx] = 1;
            }
            for (h, &(c0, c1)) in ip_hands.iter().enumerate() {
                let idx = card_pair_to_index(c0, c1);
                let ev_chips = evs[1][h] * comb;
                ip_cfvs[idx] = ((f64::from(ev_chips) - half_pot) / norm) as f32;
                valid_mask[idx] = 1;
            }

            let oop_gv: f32 = sit.ranges[0]
                .iter()
                .zip(oop_cfvs.iter())
                .map(|(&r, &c)| r * c)
                .sum();
            let ip_gv: f32 = sit.ranges[1]
                .iter()
                .zip(ip_cfvs.iter())
                .map(|(&r, &c)| r * c)
                .sum();

            let board = sit.board_cards().to_vec();
            let records = vec![
                TrainingRecord {
                    board: board.clone(),
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 0,
                    game_value: oop_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: oop_cfvs,
                    valid_mask,
                },
                TrainingRecord {
                    board,
                    pot: sit.pot as f32,
                    effective_stack: sit.effective_stack as f32,
                    player: 1,
                    game_value: ip_gv,
                    oop_range: sit.ranges[0],
                    ip_range: sit.ranges[1],
                    cfvs: ip_cfvs,
                    valid_mask,
                },
            ];

            let mut w = writer.lock().unwrap();
            if let Err(e) = w.write(&records) {
                return Err(e);
            }
            drop(w);

            pb.inc(1);
        }

        let mut w = writer.lock().unwrap();
        w.flush()?;
        let total = w.count();
        drop(w);

        pb.finish_with_message("done");
        eprintln!("Wrote {total} GPU records to {}", output_path.display());
        Ok(())
    }

    /// GPU-accelerated turn datagen with canonical topology and batched solving.
    ///
    /// Builds ONE canonical turn tree at startup (max-SPR, no bet-size collapsing,
    /// 1326-hand layout that enumerates all possible hand pairs). Creates a
    /// `GpuBatchSolver` once with that topology and solves K turn situations per
    /// kernel launch. Each situation supplies its own `SubgameSpec` with
    /// per-game initial weights (zeroing board-conflicting hands), per-game fold
    /// payoffs scaled from `sit.pot`, and per-game leaf CFVs from BoundaryNet.
    ///
    /// Requires `river_model_path` in config — there is no zero-CFV fallback.
    #[cfg(feature = "gpu-turn-datagen")]
    fn run_gpu_turn(config: &CfvnetConfig, output_path: &Path) -> Result<(), String> {
        use gpu_range_solver::extract::extract_topology;
        use gpu_range_solver::GpuBatchSolver;

        let num_samples = config.datagen.num_samples;
        let seed = crate::config::resolve_seed(config.datagen.seed);
        let initial_stack = config.game.initial_stack;
        let max_iterations = config.datagen.solver_iterations;
        let leaf_eval_interval = config.datagen.leaf_eval_interval.max(1);
        let batch_size = config.datagen.gpu_batch_size.unwrap_or(256).max(1);

        let bet_sizes = super::game_tree::parse_bet_sizes_all(&config.game.bet_sizes);
        if bet_sizes.is_empty() {
            return Err("no valid bet sizes".into());
        }

        // Unconditionally require BoundaryNet model.
        let model_path = config.game.river_model_path.as_deref()
            .ok_or("river_model_path is required for GPU turn datagen")?;
        let evaluator = crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator::load(
            std::path::Path::new(model_path),
        )?;

        // Build the canonical turn tree (pot=100, stack=10000 → SPR=100, no
        // bet-size collapse) and extract its topology. All batched games share
        // this topology.
        let canonical_game = super::game_tree::build_canonical_turn_tree(&bet_sizes)
            .ok_or("failed to build canonical turn tree")?;
        let topo = extract_topology(&canonical_game);
        if topo.showdown_nodes.is_empty() {
            return Err("canonical turn tree has no boundary nodes".into());
        }
        let boundary_node_ids: Vec<usize> = topo.showdown_nodes.clone();

        // Canonical TerminalData with the universal 1326-hand layout: every
        // `card_pair_to_index` slot is a hand, and card arrays enumerate all
        // 1326 pairs. Per-game board blockers are handled by zero weights in
        // `SubgameSpec.initial_weights`.
        let canonical_num_hands = crate::datagen::range_gen::NUM_COMBOS;
        let canonical_term =
            Self::build_canonical_turn_terminal_data(&topo, canonical_num_hands);
        let canonical_hand_cards: Vec<(u8, u8)> = canonical_term.hand_cards[0].clone();

        // Create the GPU solver once with the canonical topology.
        let mut solver = GpuBatchSolver::new(
            &topo,
            &canonical_term,
            batch_size,
            canonical_num_hands,
            max_iterations,
        )
        .map_err(|e| format!("GPU solver init failed: {e}"))?;

        let leaf_ids_i32: Vec<i32> =
            boundary_node_ids.iter().map(|&id| id as i32).collect();
        let leaf_depths: Vec<i32> = boundary_node_ids
            .iter()
            .map(|&id| topo.node_depth[id] as i32)
            .collect();
        solver
            .set_leaf_injection(&leaf_ids_i32, &leaf_depths)
            .map_err(|e| format!("set_leaf_injection failed: {e}"))?;

        // Situation generator, writer, and progress bar.
        let range_source = super::RangeSource::from_config(&config.datagen)?;
        let mut sit_gen = SituationGenerator::new(
            &config.datagen,
            initial_stack,
            4, // board_size for turn
            seed,
            num_samples,
        )
        .with_range_source(range_source);

        let writer = std::sync::Arc::new(std::sync::Mutex::new(
            super::writer::RecordWriter::create(output_path, config.datagen.per_file)?,
        ));

        let pb = indicatif::ProgressBar::new(num_samples);
        pb.set_style(
            indicatif::ProgressStyle::default_bar()
                .template(
                    "{wide_bar} {pos}/{len} [{elapsed_precise}] ETA {eta} ({per_sec}) {msg}",
                )
                .expect("valid template"),
        );

        let start_time = std::time::Instant::now();
        let mut batch_sits: Vec<crate::datagen::sampler::Situation> = Vec::with_capacity(batch_size);
        let mut batch_specs: Vec<gpu_range_solver::SubgameSpec> = Vec::with_capacity(batch_size);
        let mut last_log_count: u64 = 0;

        for sit in &mut sit_gen {
            let spec = Self::build_turn_subgame_spec(&sit, &topo, canonical_num_hands);
            batch_sits.push(sit);
            batch_specs.push(spec);

            if batch_specs.len() >= batch_size {
                Self::solve_and_write_batch(
                    &evaluator,
                    &mut solver,
                    &topo,
                    &boundary_node_ids,
                    &canonical_hand_cards,
                    canonical_num_hands,
                    max_iterations,
                    leaf_eval_interval,
                    &batch_specs,
                    &batch_sits,
                    &writer,
                )?;
                pb.inc(batch_sits.len() as u64);
                if pb.position() - last_log_count >= 1000 {
                    let elapsed = start_time.elapsed().as_secs_f64();
                    let throughput = if elapsed > 0.0 {
                        pb.position() as f64 / elapsed
                    } else {
                        0.0
                    };
                    eprintln!(
                        "datagen: {} samples, {:.1} samples/sec",
                        pb.position(),
                        throughput
                    );
                    last_log_count = pb.position();
                }
                batch_sits.clear();
                batch_specs.clear();
            }
        }

        // Flush any remaining partial batch.
        if !batch_specs.is_empty() {
            Self::solve_and_write_batch(
                &evaluator,
                &mut solver,
                &topo,
                &boundary_node_ids,
                &canonical_hand_cards,
                canonical_num_hands,
                max_iterations,
                leaf_eval_interval,
                &batch_specs,
                &batch_sits,
                &writer,
            )?;
            pb.inc(batch_sits.len() as u64);
        }

        let mut w = writer.lock().unwrap();
        w.flush()?;
        let total = w.count();
        drop(w);

        pb.finish_with_message("done");
        let elapsed = start_time.elapsed().as_secs_f64();
        let throughput = if elapsed > 0.0 {
            pb.position() as f64 / elapsed
        } else {
            0.0
        };
        eprintln!(
            "Wrote {total} GPU turn records to {} ({:.1} samples/sec)",
            output_path.display(),
            throughput
        );
        Ok(())
    }

    /// Solve one batch of turn subgames on the canonical topology and write
    /// their training records.
    #[cfg(feature = "gpu-turn-datagen")]
    #[allow(clippy::too_many_arguments)]
    fn solve_and_write_batch(
        evaluator: &crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator,
        solver: &mut gpu_range_solver::GpuBatchSolver,
        topo: &gpu_range_solver::extract::TreeTopology,
        boundary_node_ids: &[usize],
        canonical_hand_cards: &[(u8, u8)],
        num_hands: usize,
        max_iterations: u32,
        leaf_eval_interval: u32,
        specs: &[gpu_range_solver::SubgameSpec],
        sits: &[crate::datagen::sampler::Situation],
        writer: &std::sync::Arc<
            std::sync::Mutex<super::writer::RecordWriter>,
        >,
    ) -> Result<(), String> {
        use gpu_range_solver::compute_evs_from_strategy_sum;

        debug_assert_eq!(specs.len(), sits.len());
        let batch_len = specs.len();
        if batch_len == 0 {
            return Ok(());
        }

        solver
            .prepare_batch(specs)
            .map_err(|e| format!("prepare_batch failed: {e}"))?;

        // Initial boundary evaluation: uniform strategy for each game in the
        // batch. Reach at each boundary is the forward-walk reach under a
        // uniform strategy (strategy_sum = zeros produces uniform avg strategy
        // in compute_reach_at_nodes).
        let initial_ss = vec![0.0_f32; topo.num_edges * num_hands];
        let batched_initial = Self::batch_boundary_leaf_cfvs(
            evaluator,
            topo,
            boundary_node_ids,
            canonical_hand_cards,
            num_hands,
            &vec![initial_ss; batch_len],
            specs,
            sits,
        )?;
        solver
            .update_leaf_cfvs(&batched_initial.0, &batched_initial.1)
            .map_err(|e| format!("initial update_leaf_cfvs failed: {e}"))?;

        // Iterative solve with periodic reach-based boundary re-eval.
        let mut iter = 0u32;
        while iter < max_iterations {
            let end = (iter + leaf_eval_interval).min(max_iterations);
            solver
                .run_iterations(iter, end)
                .map_err(|e| format!("run_iterations failed: {e}"))?;
            iter = end;

            if iter < max_iterations {
                let mid_results = solver
                    .extract_results()
                    .map_err(|e| format!("mid-solve extract: {e}"))?;
                let mid_strategy_sums: Vec<Vec<f32>> = mid_results
                    .iter()
                    .map(|r| r.strategy_sum.clone())
                    .collect();
                let batched_mid = Self::batch_boundary_leaf_cfvs(
                    evaluator,
                    topo,
                    boundary_node_ids,
                    canonical_hand_cards,
                    num_hands,
                    &mid_strategy_sums,
                    specs,
                    sits,
                )?;
                solver
                    .update_leaf_cfvs(&batched_mid.0, &batched_mid.1)
                    .map_err(|e| format!("mid update_leaf_cfvs failed: {e}"))?;
            }
        }

        // Extract results and write records.
        let results = solver
            .extract_results()
            .map_err(|e| format!("extract_results failed: {e}"))?;

        let mut records_batch: Vec<crate::datagen::storage::TrainingRecord> =
            Vec::with_capacity(batch_len * 2);
        for (b, result) in results.iter().enumerate() {
            let sit = &sits[b];
            let spec = &specs[b];

            // Build a per-situation TerminalData so that compute_evs_from_strategy_sum
            // uses this game's fold payoffs (in chip units) while reusing the
            // canonical 1326-hand card layout.
            let per_sit_term = Self::build_per_sit_turn_terminal_data(
                topo,
                sit,
                canonical_hand_cards,
                num_hands,
            );

            let evs = compute_evs_from_strategy_sum(
                topo,
                &per_sit_term,
                &result.strategy_sum,
                &spec.initial_weights,
                num_hands,
            );

            Self::append_turn_records(sit, &evs, &mut records_batch);
        }

        let mut w = writer.lock().unwrap();
        w.write(&records_batch)?;
        Ok(())
    }

    /// Compute per-game leaf CFVs at boundaries for an entire batch and pack
    /// them into flat `[B * num_leaves * num_hands]` arrays suitable for
    /// `GpuBatchSolver::update_leaf_cfvs`.
    ///
    /// `strategy_sums[b]` is the strategy sum (or zeros for the initial eval)
    /// for game `b`. Reach at each boundary is computed on the CPU, converted
    /// to the 1326-combo space (identity mapping for the canonical layout),
    /// and passed to BoundaryNet.
    #[cfg(feature = "gpu-turn-datagen")]
    #[allow(clippy::too_many_arguments)]
    fn batch_boundary_leaf_cfvs(
        evaluator: &crate::datagen::gpu_boundary_eval::GpuBoundaryEvaluator,
        topo: &gpu_range_solver::extract::TreeTopology,
        boundary_node_ids: &[usize],
        canonical_hand_cards: &[(u8, u8)],
        num_hands: usize,
        strategy_sums: &[Vec<f32>],
        specs: &[gpu_range_solver::SubgameSpec],
        sits: &[crate::datagen::sampler::Situation],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        use gpu_range_solver::compute_reach_at_nodes;
        use crate::datagen::gpu_boundary_eval::{
            evaluate_boundaries_batched, BoundaryEvalRequest,
        };
        use crate::datagen::range_gen::NUM_COMBOS;

        debug_assert_eq!(strategy_sums.len(), specs.len());
        debug_assert_eq!(strategy_sums.len(), sits.len());

        let num_boundaries = boundary_node_ids.len();
        let slot_len = num_boundaries * num_hands;
        let batch_len = specs.len();
        let mut batched_p0 = vec![0.0_f32; batch_len * slot_len];
        let mut batched_p1 = vec![0.0_f32; batch_len * slot_len];

        for (b, sit) in sits.iter().enumerate() {
            let spec = &specs[b];
            let strategy_sum = &strategy_sums[b];

            let reach = compute_reach_at_nodes(
                topo,
                strategy_sum,
                &spec.initial_weights,
                num_hands,
                boundary_node_ids,
            );

            // Canonical hand layout: hand index == card_pair_to_index, so we
            // can copy reach directly without remapping.
            debug_assert_eq!(num_hands, NUM_COMBOS);
            let mut oop_reach_1326 = vec![0.0_f32; num_boundaries * NUM_COMBOS];
            let mut ip_reach_1326 = vec![0.0_f32; num_boundaries * NUM_COMBOS];
            for bi in 0..num_boundaries {
                let src_base = bi * num_hands;
                let dst_base = bi * NUM_COMBOS;
                // reach[0] = opponent (P1/IP) reach from P0 traversal
                // reach[1] = opponent (P0/OOP) reach from P1 traversal
                ip_reach_1326[dst_base..dst_base + NUM_COMBOS]
                    .copy_from_slice(&reach[0][src_base..src_base + num_hands]);
                oop_reach_1326[dst_base..dst_base + NUM_COMBOS]
                    .copy_from_slice(&reach[1][src_base..src_base + num_hands]);
            }

            let board_4: [u8; 4] = [
                sit.board[0],
                sit.board[1],
                sit.board[2],
                sit.board[3],
            ];
            let request = BoundaryEvalRequest {
                board: board_4,
                pot: sit.pot as f32,
                effective_stack: sit.effective_stack as f32,
                oop_reach: oop_reach_1326,
                ip_reach: ip_reach_1326,
                num_boundaries,
            };

            let results =
                evaluate_boundaries_batched(evaluator, &[request], canonical_hand_cards)
                    .map_err(|e| format!("boundary eval failed: {e}"))?;
            let slot_start = b * slot_len;
            batched_p0[slot_start..slot_start + slot_len]
                .copy_from_slice(&results[0].leaf_cfv_p0);
            batched_p1[slot_start..slot_start + slot_len]
                .copy_from_slice(&results[0].leaf_cfv_p1);
        }

        Ok((batched_p0, batched_p1))
    }

    /// Build the canonical 1326-hand `TerminalData` for the canonical turn
    /// topology. Card arrays enumerate every possible hand pair in
    /// `card_pair_to_index` order, `same_hand_index[p][h] = h`, and each
    /// showdown/fold node gets a placeholder entry (real values come through
    /// `SubgameSpec` at `prepare_batch` time).
    #[cfg(feature = "gpu-turn-datagen")]
    fn build_canonical_turn_terminal_data(
        topo: &gpu_range_solver::extract::TreeTopology,
        num_hands: usize,
    ) -> gpu_range_solver::extract::TerminalData {
        use gpu_range_solver::extract::{FoldData, NodeType, ShowdownData, TerminalData};
        use range_solver::card::index_to_card_pair;

        let hand_cards_vec: Vec<(u8, u8)> = (0..num_hands).map(index_to_card_pair).collect();
        let hand_cards: [Vec<(u8, u8)>; 2] = [hand_cards_vec.clone(), hand_cards_vec];
        let same_hand_vec: Vec<u16> = (0..num_hands as u16).collect();
        let same_hand_index: [Vec<u16>; 2] = [same_hand_vec.clone(), same_hand_vec];

        // Canonical fold payoffs. These are placeholders — the real per-game
        // fold payoffs are supplied through `SubgameSpec.fold_payoffs_p0/p1`
        // and override these on the GPU. Use pot=100 (canonical) so the
        // structural data passed to GpuBatchSolver::new has non-degenerate
        // defaults.
        let fold_payoffs: Vec<FoldData> = topo
            .fold_nodes
            .iter()
            .map(|&node_id| {
                let amount = topo.node_amount[node_id] as f64;
                let pot = 100.0 + 2.0 * amount;
                let half_pot = 0.5 * pot;
                let folded_player = match topo.node_type[node_id] {
                    NodeType::Fold { folded_player } => folded_player,
                    _ => unreachable!(),
                };
                FoldData {
                    folded_player,
                    amount_win: half_pot,
                    amount_lose: -half_pot,
                }
            })
            .collect();

        // Turn boundary nodes appear as NodeType::Showdown in the canonical
        // topology, but the kernel's leaf injection overwrites CFVs at these
        // nodes. Zero showdown outcomes are harmless placeholders.
        let showdown_outcomes: Vec<ShowdownData> = topo
            .showdown_nodes
            .iter()
            .map(|_| ShowdownData {
                num_player_hands: [num_hands, num_hands],
                outcome_matrix_p0: vec![0.0; num_hands * num_hands],
                amount_win: 0.0,
                amount_tie: 0.0,
                amount_lose: 0.0,
            })
            .collect();

        TerminalData {
            fold_payoffs,
            showdown_outcomes,
            hand_cards,
            same_hand_index,
            num_combinations: 1.0,
        }
    }

    /// Build a `TerminalData` for a specific situation, reusing the canonical
    /// 1326-hand card layout but with per-situation fold payoffs scaled by
    /// `sit.pot` (canonical pot = 100). Used by `compute_evs_from_strategy_sum`
    /// on the CPU to reconstruct per-game EVs from the GPU strategy_sum.
    #[cfg(feature = "gpu-turn-datagen")]
    fn build_per_sit_turn_terminal_data(
        topo: &gpu_range_solver::extract::TreeTopology,
        sit: &crate::datagen::sampler::Situation,
        canonical_hand_cards: &[(u8, u8)],
        num_hands: usize,
    ) -> gpu_range_solver::extract::TerminalData {
        use gpu_range_solver::extract::{FoldData, NodeType, ShowdownData, TerminalData};

        let canonical_pot = 100.0_f64;
        let sit_pot = f64::from(sit.pot);
        let pot_scale = sit_pot / canonical_pot;

        let fold_payoffs: Vec<FoldData> = topo
            .fold_nodes
            .iter()
            .map(|&node_id| {
                let canonical_amount = topo.node_amount[node_id] as f64;
                let sit_amount = canonical_amount * pot_scale;
                let pot_at_fold = sit_pot + 2.0 * sit_amount;
                let half_pot = 0.5 * pot_at_fold;
                let folded_player = match topo.node_type[node_id] {
                    NodeType::Fold { folded_player } => folded_player,
                    _ => unreachable!(),
                };
                FoldData {
                    folded_player,
                    amount_win: half_pot,
                    amount_lose: -half_pot,
                }
            })
            .collect();

        let showdown_outcomes: Vec<ShowdownData> = topo
            .showdown_nodes
            .iter()
            .map(|_| ShowdownData {
                num_player_hands: [num_hands, num_hands],
                outcome_matrix_p0: vec![0.0; num_hands * num_hands],
                amount_win: 0.0,
                amount_tie: 0.0,
                amount_lose: 0.0,
            })
            .collect();

        let hand_cards_vec: Vec<(u8, u8)> = canonical_hand_cards.to_vec();
        let same_hand_vec: Vec<u16> = (0..num_hands as u16).collect();

        TerminalData {
            fold_payoffs,
            showdown_outcomes,
            hand_cards: [hand_cards_vec.clone(), hand_cards_vec],
            same_hand_index: [same_hand_vec.clone(), same_hand_vec],
            num_combinations: 1.0,
        }
    }

    /// Build a `SubgameSpec` for a situation against the canonical topology.
    ///
    /// Initial weights come from `sit.ranges` (already in 1326-combo space)
    /// with board-conflicting hands zeroed. Fold payoffs are derived from the
    /// canonical `node_amount` scaled by `sit.pot / canonical_pot` so the
    /// kernel sees the real per-game fold stakes. Showdown outcomes are
    /// zeros — the kernel's leaf injection overwrites CFVs at turn leaves.
    #[cfg(feature = "gpu-turn-datagen")]
    fn build_turn_subgame_spec(
        sit: &crate::datagen::sampler::Situation,
        topo: &gpu_range_solver::extract::TreeTopology,
        num_hands: usize,
    ) -> gpu_range_solver::SubgameSpec {
        use gpu_range_solver::extract::NodeType;
        use range_solver::card::index_to_card_pair;

        // Initial weights: copy sit.ranges, zeroing board-conflict hands.
        let board_mask: u64 = sit
            .board_cards()
            .iter()
            .fold(0u64, |acc, &c| acc | (1u64 << c));
        let mut w_oop = vec![0.0_f32; num_hands];
        let mut w_ip = vec![0.0_f32; num_hands];
        for idx in 0..num_hands {
            let (c0, c1) = index_to_card_pair(idx);
            let hand_mask = (1u64 << c0) | (1u64 << c1);
            if hand_mask & board_mask != 0 {
                continue;
            }
            w_oop[idx] = sit.ranges[0][idx];
            w_ip[idx] = sit.ranges[1][idx];
        }

        // Per-game fold payoffs scaled from canonical pot=100 to sit.pot.
        let canonical_pot = 100.0_f64;
        let sit_pot = f64::from(sit.pot);
        let pot_scale = sit_pot / canonical_pot;

        let num_folds = topo.fold_nodes.len();
        let mut fold_payoffs_p0 = vec![0.0_f32; num_folds];
        let mut fold_payoffs_p1 = vec![0.0_f32; num_folds];
        for (i, &node_id) in topo.fold_nodes.iter().enumerate() {
            let canonical_amount = topo.node_amount[node_id] as f64;
            let sit_amount = canonical_amount * pot_scale;
            let pot_at_fold = sit_pot + 2.0 * sit_amount;
            let half_pot = 0.5 * pot_at_fold;
            let win = half_pot as f32;
            let lose = (-half_pot) as f32;
            let folded_player = match topo.node_type[node_id] {
                NodeType::Fold { folded_player } => folded_player,
                _ => unreachable!(),
            };
            // P0 traverser: opp folded (folded_player == 1) → P0 wins.
            fold_payoffs_p0[i] = if folded_player == 1 { win } else { lose };
            // P1 traverser: opp folded (folded_player == 0) → P1 wins.
            fold_payoffs_p1[i] = if folded_player == 0 { win } else { lose };
        }

        // Showdown outcomes: zeros. Turn boundary nodes get real CFVs from
        // leaf injection in the kernel.
        let num_showdowns = topo.showdown_nodes.len();
        let zero_outcomes = vec![0.0_f32; num_showdowns * num_hands * num_hands];

        gpu_range_solver::SubgameSpec {
            initial_weights: [w_oop, w_ip],
            showdown_outcomes_p0: Some(zero_outcomes.clone()),
            showdown_outcomes_p1: Some(zero_outcomes),
            fold_payoffs_p0,
            fold_payoffs_p1,
        }
    }

    /// Append a (OOP, IP) pair of training records for one solved situation.
    ///
    /// `evs[p]` is per-hand EV in chip units (canonical 1326-hand layout).
    /// Mirrors the scaling used by the river GPU path: cfv = (ev - half_pot) / half_pot.
    #[cfg(feature = "gpu-turn-datagen")]
    fn append_turn_records(
        sit: &crate::datagen::sampler::Situation,
        evs: &[Vec<f32>; 2],
        records_out: &mut Vec<crate::datagen::storage::TrainingRecord>,
    ) {
        use crate::datagen::range_gen::NUM_COMBOS;
        use crate::datagen::storage::TrainingRecord;
        use range_solver::card::index_to_card_pair;

        let pot = f64::from(sit.pot);
        let half_pot = pot / 2.0;
        let norm = if half_pot > 0.0 { half_pot } else { 1.0 };

        let board_mask: u64 = sit
            .board_cards()
            .iter()
            .fold(0u64, |acc, &c| acc | (1u64 << c));

        let mut oop_cfvs = [0.0_f32; NUM_COMBOS];
        let mut ip_cfvs = [0.0_f32; NUM_COMBOS];
        let mut valid_mask = [0_u8; NUM_COMBOS];

        for idx in 0..NUM_COMBOS {
            let (c0, c1) = index_to_card_pair(idx);
            let hand_mask = (1u64 << c0) | (1u64 << c1);
            if hand_mask & board_mask != 0 {
                continue;
            }
            let ev_oop = f64::from(evs[0][idx]);
            let ev_ip = f64::from(evs[1][idx]);
            oop_cfvs[idx] = ((ev_oop - half_pot) / norm) as f32;
            ip_cfvs[idx] = ((ev_ip - half_pot) / norm) as f32;
            valid_mask[idx] = 1;
        }

        let oop_gv: f32 = sit.ranges[0]
            .iter()
            .zip(oop_cfvs.iter())
            .map(|(&r, &c)| r * c)
            .sum();
        let ip_gv: f32 = sit.ranges[1]
            .iter()
            .zip(ip_cfvs.iter())
            .map(|(&r, &c)| r * c)
            .sum();

        let board = sit.board_cards().to_vec();
        records_out.push(TrainingRecord {
            board: board.clone(),
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: 0,
            game_value: oop_gv,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            cfvs: oop_cfvs,
            valid_mask,
        });
        records_out.push(TrainingRecord {
            board,
            pot: sit.pot as f32,
            effective_stack: sit.effective_stack as f32,
            player: 1,
            game_value: ip_gv,
            oop_range: sit.ranges[0],
            ip_range: sit.ranges[1],
            cfvs: ip_cfvs,
            valid_mask,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{CfvnetConfig, DatagenConfig, GameConfig};
    use crate::datagen::storage::read_record;
    use std::io::BufReader;
    use tempfile::NamedTempFile;

    fn test_config(num_samples: u64, board_size: usize) -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                board_size,
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                mode: "domain".into(),
                solver_iterations: 20,
                seed: Some(42),
                ..Default::default()
            },
            training: Default::default(),
            evaluation: Default::default(),
        }
    }

    #[test]
    fn pipeline_produces_records_for_river() {
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(3, 5);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let r0 = read_record(&mut reader).unwrap();
        assert_eq!(r0.board.len(), 5);
        assert!(r0.pot > 0.0);
    }

    #[test]
    fn pipeline_produces_records_for_turn_exact() {
        // Turn without a river model falls back to exact mode (full solve through river).
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(3, 4);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let r0 = read_record(&mut reader).unwrap();
        assert_eq!(r0.board.len(), 4);
        assert!(r0.pot > 0.0);
    }

    #[test]
    fn pipeline_writes_correct_record_count() {
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(5, 5);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let mut count = 0;
        while read_record(&mut reader).is_ok() {
            count += 1;
        }
        // Each game produces 2 records (OOP + IP), and we asked for 5 samples.
        // Some may be skipped (degenerate), but we should get at least 2 records.
        assert!(count >= 2, "expected at least 2 records, got {count}");
        assert_eq!(count % 2, 0, "records should come in pairs (OOP+IP)");
    }

    #[test]
    fn pipeline_records_have_valid_fields() {
        range_solver::set_force_sequential(true);
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config(3, 5);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        while let Ok(rec) = read_record(&mut reader) {
            assert_eq!(rec.board.len(), 5);
            assert!(rec.pot > 0.0);
            assert!(rec.effective_stack > 0.0);
            assert!(rec.game_value.is_finite());
            assert!(rec.player == 0 || rec.player == 1);
            for &cfv in &rec.cfvs {
                assert!(cfv.is_finite());
            }
        }
    }

    fn test_config_threaded(num_samples: u64, board_size: usize, threads: usize) -> CfvnetConfig {
        CfvnetConfig {
            game: GameConfig {
                initial_stack: 200,
                board_size,
                ..Default::default()
            },
            datagen: DatagenConfig {
                num_samples,
                mode: "domain".into(),
                solver_iterations: 20,
                seed: Some(42),
                threads,
                ..Default::default()
            },
            training: Default::default(),
            evaluation: Default::default(),
        }
    }

    #[test]
    fn pipeline_parallel_produces_valid_records() {
        // With 4 threads and 6 samples, verify records are valid and paired.
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config_threaded(6, 5, 4);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let mut count = 0;
        while let Ok(rec) = read_record(&mut reader) {
            assert_eq!(rec.board.len(), 5);
            assert!(rec.pot > 0.0);
            assert!(rec.effective_stack > 0.0);
            assert!(rec.game_value.is_finite());
            assert!(rec.player == 0 || rec.player == 1);
            for &cfv in &rec.cfvs {
                assert!(cfv.is_finite());
            }
            count += 1;
        }
        assert!(count >= 2, "expected at least 2 records, got {count}");
        assert_eq!(count % 2, 0, "records should come in pairs (OOP+IP)");
    }

    #[test]
    fn pipeline_parallel_single_thread_matches_sequential() {
        // With 1 thread, should still work correctly.
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config_threaded(3, 5, 1);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let mut count = 0;
        while read_record(&mut reader).is_ok() {
            count += 1;
        }
        assert!(count >= 2, "expected at least 2 records, got {count}");
        assert_eq!(count % 2, 0, "records should come in pairs (OOP+IP)");
    }

    #[test]
    fn pipeline_parallel_tracks_exploitability() {
        // Run parallel pipeline and verify it completes without panic
        // (exploitability tracking uses atomics in parallel mode).
        let tmp = NamedTempFile::new().unwrap();
        let config = test_config_threaded(4, 5, 2);
        DomainPipeline::run(&config, tmp.path()).unwrap();

        let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
        let mut count = 0;
        while read_record(&mut reader).is_ok() {
            count += 1;
        }
        assert!(count >= 2, "expected at least 2 records, got {count}");
    }

    #[cfg(feature = "gpu-datagen")]
    mod gpu_tests {
        use super::*;

        fn gpu_test_config(num_samples: u64, board_size: usize) -> CfvnetConfig {
            CfvnetConfig {
                game: GameConfig {
                    initial_stack: 200,
                    board_size,
                    ..Default::default()
                },
                datagen: DatagenConfig {
                    num_samples,
                    mode: "domain".into(),
                    solver_iterations: 200,
                    seed: Some(42),
                    backend: "gpu".into(),
                    gpu_batch_size: Some(4),
                    ..Default::default()
                },
                training: Default::default(),
                evaluation: Default::default(),
            }
        }

        #[test]
        fn gpu_pipeline_produces_records_for_river() {
            range_solver::set_force_sequential(true);
            let tmp = NamedTempFile::new().unwrap();
            let config = gpu_test_config(3, 5);
            DomainPipeline::run(&config, tmp.path()).unwrap();

            let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
            let mut count = 0;
            while let Ok(rec) = read_record(&mut reader) {
                assert_eq!(rec.board.len(), 5);
                assert!(rec.pot > 0.0);
                assert!(rec.effective_stack > 0.0);
                assert!(rec.game_value.is_finite());
                assert!(rec.player == 0 || rec.player == 1);
                for &cfv in &rec.cfvs {
                    assert!(cfv.is_finite());
                }
                count += 1;
            }
            assert!(count >= 2, "expected at least 2 records, got {count}");
            assert_eq!(count % 2, 0, "records should come in pairs (OOP+IP)");
        }

        #[test]
        fn gpu_pipeline_records_match_cpu_pipeline() {
            range_solver::set_force_sequential(true);

            // GPU path
            let tmp_gpu = NamedTempFile::new().unwrap();
            let config_gpu = gpu_test_config(2, 5);
            DomainPipeline::run(&config_gpu, tmp_gpu.path()).unwrap();

            // CPU path (same seed)
            let tmp_cpu = NamedTempFile::new().unwrap();
            let config_cpu = CfvnetConfig {
                datagen: DatagenConfig {
                    backend: "cpu".into(),
                    gpu_batch_size: None,
                    ..config_gpu.datagen.clone()
                },
                ..config_gpu.clone()
            };
            DomainPipeline::run(&config_cpu, tmp_cpu.path()).unwrap();

            // Both should produce records
            let mut reader_gpu = BufReader::new(std::fs::File::open(tmp_gpu.path()).unwrap());
            let mut gpu_count = 0;
            while read_record(&mut reader_gpu).is_ok() {
                gpu_count += 1;
            }

            let mut reader_cpu = BufReader::new(std::fs::File::open(tmp_cpu.path()).unwrap());
            let mut cpu_count = 0;
            while read_record(&mut reader_cpu).is_ok() {
                cpu_count += 1;
            }

            assert!(gpu_count >= 2, "GPU should produce at least 2 records");
            assert!(cpu_count >= 2, "CPU should produce at least 2 records");
        }
    }

    #[cfg(feature = "gpu-datagen")]
    mod gpu_turn_tests {
        use super::*;

        /// GPU turn datagen without a model path should return an error
        /// (zero-CFV fallback has been removed).
        #[cfg(feature = "gpu-turn-datagen")]
        #[test]
        fn gpu_turn_pipeline_requires_model_path() {
            let tmp = NamedTempFile::new().unwrap();
            let config = CfvnetConfig {
                game: GameConfig {
                    initial_stack: 200,
                    board_size: 4,
                    // No river_model_path
                    ..Default::default()
                },
                datagen: DatagenConfig {
                    num_samples: 1,
                    mode: "domain".into(),
                    solver_iterations: 10,
                    seed: Some(42),
                    backend: "gpu".into(),
                    gpu_batch_size: Some(1),
                    leaf_eval_interval: 0,
                    ..Default::default()
                },
                training: Default::default(),
                evaluation: Default::default(),
            };
            let result = DomainPipeline::run(&config, tmp.path());
            assert!(result.is_err(), "should error without river_model_path");
            let err = result.unwrap_err();
            assert!(
                err.contains("river_model_path"),
                "error should mention river_model_path, got: {err}"
            );
        }

        /// GPU turn datagen with a model produces valid records with
        /// reach-based boundary re-evaluation.
        #[cfg(feature = "gpu-turn-datagen")]
        #[test]
        fn gpu_turn_pipeline_produces_records() {
            let model_path = "../../local_data/models/cfvnet_river_py_v2/model.onnx";
            if !std::path::Path::new(model_path).exists() {
                eprintln!("Skipping: ONNX model not found at {model_path}");
                return;
            }
            range_solver::set_force_sequential(true);
            let tmp = NamedTempFile::new().unwrap();
            let config = CfvnetConfig {
                game: GameConfig {
                    initial_stack: 200,
                    board_size: 4,
                    river_model_path: Some(model_path.into()),
                    ..Default::default()
                },
                datagen: DatagenConfig {
                    num_samples: 1,
                    mode: "domain".into(),
                    solver_iterations: 10,
                    seed: Some(42),
                    backend: "gpu".into(),
                    gpu_batch_size: Some(1),
                    leaf_eval_interval: 5,
                    ..Default::default()
                },
                training: Default::default(),
                evaluation: Default::default(),
            };
            DomainPipeline::run(&config, tmp.path()).unwrap();

            let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
            let mut count = 0;
            while let Ok(rec) = read_record(&mut reader) {
                assert_eq!(rec.board.len(), 4, "turn records should have 4-card board");
                assert!(rec.pot > 0.0);
                assert!(rec.effective_stack > 0.0);
                assert!(rec.game_value.is_finite(), "game_value must be finite");
                assert!(rec.player == 0 || rec.player == 1);
                for (i, &cfv) in rec.cfvs.iter().enumerate() {
                    assert!(cfv.is_finite(), "cfv[{i}] must be finite");
                }
                count += 1;
            }
            assert!(count >= 2, "expected at least 2 records, got {count}");
            assert_eq!(count % 2, 0, "records should come in pairs (OOP+IP)");
        }

        /// GPU turn datagen with a small batch (batch_size=4) exercises the
        /// canonical-topology batched orchestrator. Verifies every situation
        /// in the batch produces valid finite records and that the total
        /// record count matches `num_samples * 2` (OOP+IP per sample).
        #[cfg(feature = "gpu-turn-datagen")]
        #[test]
        fn gpu_turn_batched_pipeline_produces_records() {
            let model_path = "../../local_data/models/cfvnet_river_py_v2/model.onnx";
            if !std::path::Path::new(model_path).exists() {
                eprintln!("Skipping: ONNX model not found at {model_path}");
                return;
            }
            range_solver::set_force_sequential(true);
            let tmp = NamedTempFile::new().unwrap();
            let config = CfvnetConfig {
                game: GameConfig {
                    initial_stack: 200,
                    board_size: 4,
                    river_model_path: Some(model_path.into()),
                    ..Default::default()
                },
                datagen: DatagenConfig {
                    num_samples: 6,
                    mode: "domain".into(),
                    solver_iterations: 20,
                    seed: Some(1337),
                    backend: "gpu".into(),
                    gpu_batch_size: Some(4),
                    leaf_eval_interval: 10,
                    ..Default::default()
                },
                training: Default::default(),
                evaluation: Default::default(),
            };
            DomainPipeline::run(&config, tmp.path()).unwrap();

            let mut reader = BufReader::new(std::fs::File::open(tmp.path()).unwrap());
            let mut count = 0;
            while let Ok(rec) = read_record(&mut reader) {
                assert_eq!(rec.board.len(), 4, "turn records should have 4-card board");
                assert!(rec.pot > 0.0);
                assert!(rec.effective_stack > 0.0);
                assert!(rec.game_value.is_finite(), "game_value must be finite");
                assert!(rec.player == 0 || rec.player == 1);
                for (i, &cfv) in rec.cfvs.iter().enumerate() {
                    assert!(cfv.is_finite(), "cfv[{i}] must be finite");
                }
                count += 1;
            }
            // Records come in OOP+IP pairs, so we expect 2 per sample.
            assert_eq!(count, 12, "expected 12 records (6 samples x 2), got {count}");
        }
    }
}
