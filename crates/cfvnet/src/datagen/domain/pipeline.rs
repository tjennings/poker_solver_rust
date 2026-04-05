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
}
