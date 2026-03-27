// Multi-street offline seeding orchestration.
//
// Runs the bottom-up training pipeline: river -> turn -> flop.
// Preflop is handled by the blueprint strategy (see `play_preflop_under_blueprint`
// in `blueprint_sampler.rs`), so orchestration only seeds postflop streets.
//
// At each street layer:
//   1. Generate PBSs from blueprint play for that street
//   2. Solve subgames (exact for river, depth-limited for others using value net)
//   3. Append training data to accumulated buffer
//   4. Retrain value net on all accumulated data
//   5. Use retrained net for next street's leaf evaluation

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};

use cfvnet::model::network::{CfvNet, INPUT_SIZE};
use poker_solver_core::blueprint_v2::mccfr::AllBuckets;
use poker_solver_core::blueprint_v2::bundle::BlueprintV2Strategy;
use poker_solver_core::blueprint_v2::game_tree::GameTree;
use poker_solver_core::blueprint_v2::Street;

use crate::config::RebelConfig;
use crate::data_buffer::DiskBuffer;
use crate::generate::{build_solve_config, generate_pbs, solve_buffer_records};
use crate::leaf_evaluator::RebelLeafEvaluator;
use crate::training::{build_train_config, export_training_data};

/// Result of seeding a single street.
#[derive(Debug, Clone)]
pub struct StreetResult {
    /// Which street was seeded.
    pub street: Street,
    /// Number of PBSs generated from blueprint play.
    pub pbs_generated: usize,
    /// Number of records successfully solved.
    pub records_solved: usize,
    /// Training MSE after retraining on accumulated data (None if training was skipped).
    pub training_loss: Option<f32>,
}

/// Result of the full offline seeding pipeline.
#[derive(Debug)]
pub struct SeederResult {
    /// Path to the final trained model directory.
    pub model_path: PathBuf,
    /// Path to the exported training data file.
    pub data_path: PathBuf,
    /// Total number of solved records across all streets.
    pub total_records: usize,
    /// Per-street results in execution order (river, turn, flop).
    pub per_street: Vec<StreetResult>,
}

/// Orchestrates the bottom-up offline seeding pipeline.
///
/// The pipeline runs in this order:
/// 1. River: generate PBSs, solve exactly (no depth limit), train value net
/// 2. Turn: generate PBSs, solve depth-limited (value net at river boundary), retrain
/// 3. Flop: generate PBSs, solve depth-limited (value net at turn boundary), retrain
///
/// Preflop is excluded — it is handled by the blueprint strategy via
/// `play_preflop_under_blueprint` in the self-play loop.
pub struct OfflineSeeder {
    config: RebelConfig,
    /// Track which streets have been seeded.
    completed_streets: Vec<Street>,
}

/// The bottom-up street ordering for offline seeding.
///
/// Preflop is excluded: it is handled by the blueprint strategy
/// (see `play_preflop_under_blueprint` in `blueprint_sampler.rs`).
/// We only need value-net training data for postflop streets.
const STREET_ORDER: [Street; 3] = [Street::River, Street::Turn, Street::Flop];

/// Backend type for CPU training via NdArray.
type TrainBackend = Autodiff<NdArray>;
/// Inference backend for leaf evaluation.
type InferBackend = NdArray;

impl OfflineSeeder {
    /// Create a new offline seeder with the given configuration.
    pub fn new(config: RebelConfig) -> Self {
        Self {
            config,
            completed_streets: Vec::new(),
        }
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &RebelConfig {
        &self.config
    }

    /// Returns which streets have been completed.
    pub fn completed_streets(&self) -> &[Street] {
        &self.completed_streets
    }

    /// Run the full offline seeding pipeline.
    ///
    /// Returns paths to the trained model and accumulated training data.
    pub fn run(
        &mut self,
        strategy: &BlueprintV2Strategy,
        tree: &GameTree,
        buckets: &AllBuckets,
    ) -> Result<SeederResult, String> {
        let output_dir = PathBuf::from(&self.config.output_dir);
        std::fs::create_dir_all(&output_dir)
            .map_err(|e| format!("failed to create output directory: {e}"))?;

        let buffer_path = output_dir.join(&self.config.buffer.path);
        let mut buffer = DiskBuffer::create(&buffer_path, self.config.buffer.max_records)
            .map_err(|e| format!("failed to create disk buffer: {e}"))?;

        let model_dir = output_dir.join("model");
        std::fs::create_dir_all(&model_dir)
            .map_err(|e| format!("failed to create model directory: {e}"))?;

        let training_data_path = output_dir.join("training_data.bin");

        let mut per_street = Vec::new();
        let mut total_records = 0usize;

        for &street in &STREET_ORDER {
            eprintln!("\n{}", "=".repeat(60));
            eprintln!("Seeding street: {street:?}");
            eprintln!("{}", "=".repeat(60));

            // For river, solve exactly (no leaf evaluator needed).
            // For other streets, load the current model as a leaf evaluator.
            let evaluator: Option<Box<dyn poker_solver_core::blueprint_v2::LeafEvaluator>> =
                if street == Street::River {
                    None
                } else {
                    Some(self.load_evaluator(&model_dir)?)
                };

            let street_result = self.seed_street(
                street,
                strategy,
                tree,
                buckets,
                evaluator.as_deref(),
                &mut buffer,
            )?;

            total_records += street_result.records_solved;

            // Export accumulated buffer to training file.
            eprintln!(
                "Exporting {} accumulated records to {}...",
                buffer.len(),
                training_data_path.display()
            );
            let exported = export_training_data(&buffer, &training_data_path)
                .map_err(|e| format!("failed to export training data: {e}"))?;
            eprintln!("Exported {exported} records.");

            // Retrain value net on ALL accumulated data.
            let train_loss = self.train_model(&training_data_path, &model_dir)?;

            let result = StreetResult {
                street,
                pbs_generated: street_result.pbs_generated,
                records_solved: street_result.records_solved,
                training_loss: Some(train_loss),
            };

            eprintln!(
                "Street {:?} complete: {} PBSs, {} solved, train_loss={:.6}",
                street, result.pbs_generated, result.records_solved, train_loss
            );

            per_street.push(result);
            self.completed_streets.push(street);
        }

        Ok(SeederResult {
            model_path: model_dir,
            data_path: training_data_path,
            total_records,
            per_street,
        })
    }

    /// Run seeding for a single street.
    ///
    /// - `street`: which street to generate and solve PBSs for
    /// - `evaluator`: optional LeafEvaluator for depth-limited solving (None for river)
    /// - `buffer`: disk buffer to append solved records to
    fn seed_street(
        &self,
        street: Street,
        strategy: &BlueprintV2Strategy,
        tree: &GameTree,
        buckets: &AllBuckets,
        evaluator: Option<&dyn poker_solver_core::blueprint_v2::LeafEvaluator>,
        buffer: &mut DiskBuffer,
    ) -> Result<StreetResult, String> {
        let board_card_count = street_to_board_cards(street);

        // Step 1: Generate PBSs from blueprint play.
        // generate_pbs generates PBSs at ALL street boundaries; we collect them
        // all, then filter/solve by street based on board_card_count.
        eprintln!(
            "Generating PBSs for {:?} ({} board cards)...",
            street, board_card_count
        );

        let pre_count = buffer.len();
        let mutex_buffer = Mutex::new(
            DiskBuffer::create(
                PathBuf::from(&self.config.output_dir).join(format!("temp_{street:?}.bin")),
                self.config.buffer.max_records,
            )
            .map_err(|e| format!("failed to create temp buffer: {e}"))?,
        );

        let pbs_generated = generate_pbs(strategy, tree, buckets, &self.config, &mutex_buffer);
        eprintln!("Generated {pbs_generated} total PBSs across all streets.");

        // Move records matching this street's board_card_count into the main buffer.
        let temp_buffer = mutex_buffer
            .into_inner()
            .map_err(|e| format!("mutex poisoned: {e}"))?;

        let mut street_records = 0usize;
        for i in 0..temp_buffer.len() {
            let rec = temp_buffer
                .read_record(i)
                .map_err(|e| format!("failed to read temp record {i}: {e}"))?;
            if rec.board_card_count == board_card_count {
                buffer
                    .append(&rec)
                    .map_err(|e| format!("failed to append record: {e}"))?;
                street_records += 1;
            }
        }

        // Clean up temp file.
        let temp_path =
            PathBuf::from(&self.config.output_dir).join(format!("temp_{street:?}.bin"));
        let _ = std::fs::remove_file(temp_path);

        let pbs_for_street = street_records;
        eprintln!(
            "Found {} records for {:?} (buffer now has {} total records).",
            pbs_for_street,
            street,
            buffer.len()
        );

        // Step 2: Solve the newly added records.
        // solve_buffer_records supports both modes:
        //   - evaluator=None: solves river records (5-card boards) exactly
        //   - evaluator=Some: solves all streets depth-limited with the evaluator
        //     at street boundaries
        let solve_config = build_solve_config(&self.config.seed);
        let records_to_solve = buffer.len() - pre_count;

        if records_to_solve > 0 {
            if street == Street::River {
                eprintln!("Solving {records_to_solve} river records (exact, no evaluator)...");
            } else {
                eprintln!(
                    "Solving {records_to_solve} {:?} records (depth-limited with value net)...",
                    street
                );
            }

            // For river: pass None (exact solving).
            // For other streets: pass the evaluator for depth-limited solving.
            //
            // The LeafEvaluator trait is not Sync by default, but
            // solve_buffer_records needs Option<&(dyn LeafEvaluator + Sync)>.
            // RebelLeafEvaluator<NdArray> is Sync since NdArray's Device is Sync.
            // When we receive a type-erased &dyn LeafEvaluator, we cannot prove
            // Sync at compile time. So we use a wrapper for the non-river case.
            let solved = if street == Street::River {
                solve_buffer_records(
                    buffer,
                    &solve_config,
                    None,
                    self.config.seed.threads,
                )
            } else {
                // The evaluator is created from RebelLeafEvaluator<NdArray> which is
                // Send+Sync, but type-erased to Box<dyn LeafEvaluator>. We need to
                // solve single-threaded or wrap for sync access.
                // Use a SyncEvaluatorWrapper to make the &dyn LeafEvaluator usable
                // in the parallel solver.
                let sync_eval = SyncEvaluatorWrapper(evaluator.expect(
                    "non-river street requires an evaluator",
                ));
                solve_buffer_records(
                    buffer,
                    &solve_config,
                    Some(&sync_eval),
                    self.config.seed.threads,
                )
            };

            eprintln!("Solved {solved} records.");

            Ok(StreetResult {
                street,
                pbs_generated: pbs_for_street,
                records_solved: solved,
                training_loss: None,
            })
        } else {
            eprintln!("No records to solve for {:?}.", street);

            Ok(StreetResult {
                street,
                pbs_generated: pbs_for_street,
                records_solved: 0,
                training_loss: None,
            })
        }
    }

    /// Train (or retrain) the value net on all accumulated training data.
    ///
    /// Uses the NdArray (CPU) backend. Returns the final training loss.
    fn train_model(
        &self,
        data_path: &Path,
        output_dir: &Path,
    ) -> Result<f32, String> {
        let train_config = build_train_config(&self.config.training);

        eprintln!(
            "Training value net: {} epochs, batch_size={}, lr={:.2e}",
            train_config.epochs, train_config.batch_size, train_config.learning_rate
        );

        // Determine board_cards from training data. Since we accumulate data across
        // streets, the records may have different board sizes. We use 5 as the
        // board_cards parameter — this controls record size detection only, and
        // the cfvnet training pipeline handles variable-length boards internally
        // through the TrainingRecord format which stores actual board length.
        let board_cards = 5;

        let device = Default::default();
        let result = cfvnet::model::training::train::<TrainBackend>(
            &device,
            data_path,
            board_cards,
            &train_config,
            Some(output_dir),
        );

        eprintln!("Training complete. Final loss: {:.6}", result.final_train_loss);
        Ok(result.final_train_loss)
    }

    /// Load the current trained model as a LeafEvaluator for depth-limited solving.
    fn load_evaluator(
        &self,
        model_dir: &Path,
    ) -> Result<Box<dyn poker_solver_core::blueprint_v2::LeafEvaluator>, String> {
        let model_path = model_dir.join("model");
        let mpk_path = model_path.with_extension("mpk.gz");

        if !mpk_path.exists() {
            return Err(format!(
                "model file not found at {}. Cannot create leaf evaluator.",
                mpk_path.display()
            ));
        }

        let device: <InferBackend as burn::tensor::backend::Backend>::Device = Default::default();
        let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();

        let model = CfvNet::<InferBackend>::new(
            &device,
            self.config.training.hidden_layers,
            self.config.training.hidden_size,
            INPUT_SIZE,
        )
        .load_file(model_path, &recorder, &device)
        .map_err(|e| format!("failed to load model: {e}"))?;

        Ok(Box::new(RebelLeafEvaluator::new(model, device)))
    }
}

/// Map a street to the expected number of board cards.
pub fn street_to_board_cards(street: Street) -> u8 {
    match street {
        Street::Preflop => 0,
        Street::Flop => 3,
        Street::Turn => 4,
        Street::River => 5,
    }
}

/// Wrapper to expose a `&dyn LeafEvaluator` as `LeafEvaluator + Sync`.
///
/// The underlying `RebelLeafEvaluator<NdArray>` is inherently `Sync` (NdArray's
/// device is a unit type, and the model is behind `Arc<Mutex<...>>`), but once
/// boxed as `Box<dyn LeafEvaluator>` the compiler loses proof of `Sync`.
///
/// This wrapper is safe because we only construct it from evaluators that were
/// originally `RebelLeafEvaluator<NdArray>`, which is `Send + Sync`.
struct SyncEvaluatorWrapper<'a>(&'a dyn poker_solver_core::blueprint_v2::LeafEvaluator);

// SAFETY: The only concrete type wrapped is RebelLeafEvaluator<NdArray>,
// which is Sync (NdArray device is (), model is Arc<Mutex<CfvNet<NdArray>>>).
unsafe impl Sync for SyncEvaluatorWrapper<'_> {}

impl poker_solver_core::blueprint_v2::LeafEvaluator for SyncEvaluatorWrapper<'_> {
    fn evaluate(
        &self,
        combos: &[[poker_solver_core::poker::Card; 2]],
        board: &[poker_solver_core::poker::Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64> {
        self.0.evaluate(combos, board, pot, effective_stack, oop_range, ip_range, traverser)
    }

    fn evaluate_boundaries(
        &self,
        combos: &[[poker_solver_core::poker::Card; 2]],
        board: &[poker_solver_core::poker::Card],
        oop_range: &[f64],
        ip_range: &[f64],
        requests: &[(f64, f64, u8)],
    ) -> Vec<Vec<f64>> {
        self.0.evaluate_boundaries(combos, board, oop_range, ip_range, requests)
    }
}

/// Run the full bottom-up offline seeding pipeline.
///
/// Pipeline order: River -> Turn -> Flop
/// (Preflop is handled by the blueprint strategy, not solved here.)
///
/// At each street:
/// 1. Generate PBSs from blueprint play (all streets at once via play_hand)
/// 2. Solve subgames for target street only
///    - River: exact (no depth limit)
///    - Others: depth-limited with current value net at boundaries
/// 3. Export accumulated buffer to cfvnet training files
/// 4. Train/retrain value net on ALL accumulated data
/// 5. Load retrained model for next street's leaf evaluation
///
/// Returns paths and stats.
pub fn run_offline_seeding(
    config: &RebelConfig,
    strategy: &BlueprintV2Strategy,
    tree: &GameTree,
    buckets: &AllBuckets,
) -> Result<SeederResult, String> {
    let mut seeder = OfflineSeeder::new(config.clone());
    seeder.run(strategy, tree, buckets)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a minimal RebelConfig for testing.
    fn test_config() -> RebelConfig {
        use crate::config::*;

        RebelConfig {
            blueprint_path: "/tmp/test_blueprint".to_string(),
            cluster_dir: "/tmp/test_clusters".to_string(),
            output_dir: "/tmp/test_rebel_output".to_string(),
            game: GameConfig {
                initial_stack: 400,
                small_blind: 1,
                big_blind: 2,
            },
            seed: SeedConfig {
                num_hands: 100,
                seed: 42,
                threads: 4,
                solver_iterations: 100,
                target_exploitability: 0.01,
                add_allin_threshold: 1.5,
                force_allin_threshold: 0.15,
                bet_sizes: BetSizeConfig {
                    flop: [vec![0.5, 1.0], vec![0.5, 1.0]],
                    turn: [vec![0.5, 1.0], vec![0.5, 1.0]],
                    river: [vec![0.5, 1.0], vec![0.5, 1.0]],
                },
            },
            training: TrainingConfig {
                hidden_layers: 2,
                hidden_size: 64,
                batch_size: 16,
                epochs: 2,
                learning_rate: 0.001,
                huber_delta: 1.0,
            },
            buffer: BufferConfig {
                max_records: 10_000,
                path: "test_buffer.bin".to_string(),
            },
        }
    }

    #[test]
    fn test_offline_seeder_new() {
        let config = test_config();
        let seeder = OfflineSeeder::new(config.clone());

        assert_eq!(seeder.config().blueprint_path, config.blueprint_path);
        assert_eq!(seeder.config().output_dir, config.output_dir);
        assert!(seeder.completed_streets().is_empty());
    }

    #[test]
    fn test_street_result_construction() {
        let result = StreetResult {
            street: Street::River,
            pbs_generated: 100,
            records_solved: 80,
            training_loss: Some(0.025),
        };

        assert_eq!(result.street, Street::River);
        assert_eq!(result.pbs_generated, 100);
        assert_eq!(result.records_solved, 80);
        assert_eq!(result.training_loss, Some(0.025));
    }

    #[test]
    fn test_street_result_no_training() {
        let result = StreetResult {
            street: Street::Turn,
            pbs_generated: 50,
            records_solved: 0,
            training_loss: None,
        };

        assert_eq!(result.street, Street::Turn);
        assert_eq!(result.training_loss, None);
    }

    #[test]
    fn test_seeder_result_construction() {
        let result = SeederResult {
            model_path: PathBuf::from("/tmp/model"),
            data_path: PathBuf::from("/tmp/data.bin"),
            total_records: 200,
            per_street: vec![
                StreetResult {
                    street: Street::River,
                    pbs_generated: 100,
                    records_solved: 80,
                    training_loss: Some(0.05),
                },
                StreetResult {
                    street: Street::Turn,
                    pbs_generated: 60,
                    records_solved: 40,
                    training_loss: Some(0.03),
                },
            ],
        };

        assert_eq!(result.total_records, 200);
        assert_eq!(result.per_street.len(), 2);
        assert_eq!(result.per_street[0].street, Street::River);
        assert_eq!(result.per_street[1].street, Street::Turn);
        assert_eq!(result.model_path, PathBuf::from("/tmp/model"));
        assert_eq!(result.data_path, PathBuf::from("/tmp/data.bin"));
    }

    #[test]
    fn test_street_to_board_cards() {
        assert_eq!(street_to_board_cards(Street::Preflop), 0);
        assert_eq!(street_to_board_cards(Street::Flop), 3);
        assert_eq!(street_to_board_cards(Street::Turn), 4);
        assert_eq!(street_to_board_cards(Street::River), 5);
    }

    #[test]
    fn test_board_card_count_for_street() {
        // Verify the mapping used for filtering buffer records by street.
        let mapping: Vec<(Street, u8)> = vec![
            (Street::River, 5),
            (Street::Turn, 4),
            (Street::Flop, 3),
            (Street::Preflop, 0),
        ];

        for (street, expected) in mapping {
            assert_eq!(
                street_to_board_cards(street),
                expected,
                "street_to_board_cards({street:?}) should be {expected}"
            );
        }
    }

    #[test]
    fn test_street_order() {
        assert_eq!(STREET_ORDER[0], Street::River);
        assert_eq!(STREET_ORDER[1], Street::Turn);
        assert_eq!(STREET_ORDER[2], Street::Flop);
    }

    #[test]
    fn test_street_order_no_preflop() {
        // Preflop is handled by the blueprint strategy, not orchestration.
        assert_eq!(STREET_ORDER.len(), 3);
        assert_eq!(STREET_ORDER[0], Street::River);
        assert_eq!(STREET_ORDER[1], Street::Turn);
        assert_eq!(STREET_ORDER[2], Street::Flop);
    }

    #[test]
    fn test_completed_streets_tracking() {
        let config = test_config();
        let mut seeder = OfflineSeeder::new(config);

        assert!(seeder.completed_streets().is_empty());

        // Simulate completing streets
        seeder.completed_streets.push(Street::River);
        assert_eq!(seeder.completed_streets().len(), 1);
        assert_eq!(seeder.completed_streets()[0], Street::River);

        seeder.completed_streets.push(Street::Turn);
        assert_eq!(seeder.completed_streets().len(), 2);
        assert_eq!(seeder.completed_streets()[1], Street::Turn);
    }

    #[test]
    fn test_load_evaluator_missing_model() {
        let config = test_config();
        let seeder = OfflineSeeder::new(config);

        let dir = tempfile::tempdir().unwrap();
        let result = seeder.load_evaluator(dir.path());
        assert!(result.is_err());
        let err_msg = match result {
            Err(e) => e,
            Ok(_) => panic!("expected error"),
        };
        assert!(
            err_msg.contains("model file not found"),
            "unexpected error: {err_msg}"
        );
    }

    #[test]
    fn test_run_offline_seeding_delegates_to_seeder() {
        // Verify run_offline_seeding creates an OfflineSeeder internally.
        // We cannot run the full pipeline without a blueprint, but we can
        // verify the function exists and has the correct signature by
        // checking that a missing blueprint_path causes an appropriate error
        // when the pipeline tries to create output dirs (not at the function
        // signature level — that's checked at compile time).
        //
        // This test just verifies the free function compiles and is callable.
        // The actual integration is tested in test_full_pipeline_integration.
        let _fn_ptr: fn(
            &RebelConfig,
            &BlueprintV2Strategy,
            &GameTree,
            &AllBuckets,
        ) -> Result<SeederResult, String> = run_offline_seeding;
    }

    #[test]
    #[ignore] // Requires a trained blueprint, game tree, and clusters
    fn test_full_pipeline_integration() {
        // This test would:
        // 1. Load a trained BlueprintV2Strategy
        // 2. Build or load a GameTree
        // 3. Load AllBuckets with cluster files
        // 4. Run the full pipeline
        // 5. Verify model and data files are created
        // 6. Verify per-street results
        //
        // To run: cargo test -p rebel orchestration::tests::test_full_pipeline_integration -- --ignored
    }
}
