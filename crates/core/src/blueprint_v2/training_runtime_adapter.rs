//! Heads-up blueprint adapter for the shared training runtime.

use std::error::Error;
use std::sync::atomic::Ordering;

use crate::training_runtime::{
    BatchBudget, BatchOutcome, RuntimeControls, RuntimeCounters, RuntimeError, RuntimeLimits,
    RuntimeResult, TrainingBackendKind, TrainingRuntimeBackend, TrainingUnit,
};

use super::trainer::BlueprintTrainer;

/// Shared-runtime adapter for the existing heads-up blueprint trainer.
pub struct HeadsUpBlueprintTrainingRuntimeAdapter {
    trainer: BlueprintTrainer,
    limits: RuntimeLimits,
    controls: RuntimeControls,
    counters: RuntimeCounters,
}

impl HeadsUpBlueprintTrainingRuntimeAdapter {
    /// Create an adapter and bridge runtime controls into the trainer flags.
    #[must_use]
    pub fn new(mut trainer: BlueprintTrainer) -> Self {
        let controls = RuntimeControls::new();
        trainer.paused = controls.pause_flag();
        trainer.quit_requested = controls.quit_flag();
        trainer.snapshot_trigger = controls.snapshot_flag();
        trainer.strategy_refresh_trigger = controls.strategy_refresh_flag();
        trainer.config_reload_trigger = controls.config_reload_flag();

        let limits = limits_from_trainer(&trainer);
        Self {
            trainer,
            limits,
            controls,
            counters: RuntimeCounters::new(),
        }
    }

    /// Return the wrapped trainer.
    #[must_use]
    pub fn into_inner(self) -> BlueprintTrainer {
        self.trainer
    }

    fn sync_limits_from_trainer(&mut self) {
        self.limits = limits_from_trainer(&self.trainer);
    }
}

impl TrainingRuntimeBackend for HeadsUpBlueprintTrainingRuntimeAdapter {
    fn kind(&self) -> TrainingBackendKind {
        TrainingBackendKind::HeadsUpBlueprintV2
    }

    fn unit(&self) -> TrainingUnit {
        TrainingUnit::Iteration
    }

    fn limits(&self) -> &RuntimeLimits {
        &self.limits
    }

    fn controls(&self) -> &RuntimeControls {
        &self.controls
    }

    fn counters(&self) -> &RuntimeCounters {
        &self.counters
    }

    fn prepare(&mut self) -> RuntimeResult<()> {
        if self.trainer.config.snapshots.resume {
            self.trainer.try_resume().map_err(runtime_error)?;
        }
        self.trainer
            .validate_training_startup()
            .map_err(runtime_error)?;
        self.trainer
            .shared_iterations
            .store(self.trainer.iterations, Ordering::Relaxed);
        self.counters.seed_units(self.trainer.iterations);
        self.sync_limits_from_trainer();
        Ok(())
    }

    fn run_next_batch(&mut self, budget: BatchBudget) -> RuntimeResult<BatchOutcome> {
        if self.trainer.trainer_specific_stop_reached() {
            return Ok(BatchOutcome::idle());
        }

        let completed = self
            .trainer
            .run_batch_with_budget(budget.remaining_target_units())
            .map_err(runtime_error)?;
        Ok(BatchOutcome::new(completed, completed > 0))
    }

    fn snapshot_now(&mut self) -> RuntimeResult<()> {
        self.trainer.save_snapshot().map_err(runtime_error)
    }

    fn refresh_telemetry(&mut self) -> RuntimeResult<()> {
        self.trainer.refresh_strategy_telemetry_now();
        Ok(())
    }

    fn reload_config(&mut self) -> RuntimeResult<()> {
        self.trainer.reload_config_now();
        self.sync_limits_from_trainer();
        Ok(())
    }
}

fn limits_from_trainer(trainer: &BlueprintTrainer) -> RuntimeLimits {
    RuntimeLimits {
        target_units: trainer.config.training.iterations,
        time_limit_minutes: trainer.config.training.time_limit_minutes,
    }
}

fn runtime_error(error: Box<dyn Error>) -> RuntimeError {
    std::io::Error::other(error.to_string()).into()
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::atomic::Ordering;

    use crate::blueprint_v2::config::{
        ActionAbstractionConfig, BlueprintV2Config, ClusteringAlgorithm, ClusteringConfig,
        GameConfig, SnapshotConfig, StreetClusterConfig, TrainingConfig,
    };
    use crate::training_runtime::{
        BatchBudget, RuntimeLimits, RuntimeStopReason, TrainingRuntimeBackend, run_until_stopped,
    };

    use super::*;

    fn toy_config(output_dir: &Path) -> BlueprintV2Config {
        BlueprintV2Config {
            game: GameConfig {
                name: "runtime-adapter-test".to_string(),
                players: 2,
                stack_depth: 10.0,
                small_blind: 1.0,
                big_blind: 2.0,
                rake_rate: 0.0,
                rake_cap: 0.0,
                allow_preflop_limp: true,
            },
            clustering: ClusteringConfig {
                algorithm: ClusteringAlgorithm::PotentialAwareEmd,
                preflop: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                flop: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                turn: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                river: StreetClusterConfig {
                    buckets: 10,
                    delta_bins: None,
                    expected_delta: false,
                    sample_boards: None,
                    metric: Default::default(),
                },
                seed: 42,
                kmeans_iterations: 50,
                cfvnet_river_data: None,
                per_flop: None,
            },
            action_abstraction: ActionAbstractionConfig {
                preflop: vec![vec!["5bb".to_string()]],
                flop: vec![vec![1.0]],
                turn: vec![vec![1.0]],
                river: vec![vec![1.0]],
            },
            training: TrainingConfig {
                cluster_path: None,
                iterations: Some(100),
                time_limit_minutes: None,
                lcfr_warmup_iterations: 999_999,
                lcfr_discount_interval: 1,
                prune_after_iterations: 999_999,
                prune_threshold: 0,
                prune_explore_pct: 0.05,
                print_every_minutes: 999_999,
                batch_size: 200,
                target_strategy_delta: None,
                purify_threshold: 0.0,
                equity_cache_path: None,
                dcfr_alpha: 1.0,
                dcfr_beta: 1.0,
                dcfr_gamma: 1.0,
                dcfr_epoch_cap: None,
                optimizer: "dcfr".to_string(),
                storage_backend: "dense".to_string(),
                sapcfr_eta: 0.5,
                brcfr_eta: 0.6,
                brcfr_warmup_iterations: 0,
                brcfr_interval: 100_000_000,
                use_baselines: false,
                baseline_alpha: 0.01,
                baseline_validation: Default::default(),
                prune_streets: None,
                regret_floor: None,
                exploitability_interval_minutes: 0,
                exploitability_samples: 100_000,
            },
            snapshots: SnapshotConfig {
                warmup_minutes: 999_999,
                snapshot_every_minutes: 999_999,
                output_dir: output_dir.to_string_lossy().to_string(),
                resume: false,
                max_snapshots: None,
            },
        }
    }

    fn toy_trainer(config: BlueprintV2Config) -> BlueprintTrainer {
        let mut trainer = BlueprintTrainer::new(config);
        trainer.skip_bucket_validation = true;
        trainer
    }

    #[test]
    fn adapter_seeds_counters_from_existing_iterations() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = toy_config(dir.path());
        config.training.iterations = Some(3);
        config.training.batch_size = 10;

        let mut trainer = toy_trainer(config);
        trainer.iterations = 2;
        trainer.shared_iterations.store(2, Ordering::Relaxed);

        let mut adapter = HeadsUpBlueprintTrainingRuntimeAdapter::new(trainer);
        let mut telemetry = Vec::new();
        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 3,
                target_units: 3,
            }
        );
        assert_eq!(adapter.counters.load_units(), 3);
        assert_eq!(adapter.trainer.iterations, 3);
    }

    #[test]
    fn adapter_batch_budget_prevents_overshoot() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = toy_config(dir.path());
        config.training.iterations = Some(100);
        config.training.batch_size = 50;

        let mut adapter = HeadsUpBlueprintTrainingRuntimeAdapter::new(toy_trainer(config));
        adapter.prepare().expect("prepare");

        let budget = BatchBudget::from_limits(
            &RuntimeLimits {
                target_units: Some(7),
                time_limit_minutes: None,
            },
            adapter.counters(),
        );
        let outcome = adapter.run_next_batch(budget).expect("batch runs");

        assert_eq!(outcome.units_completed, 7);
        assert!(outcome.did_work);
        assert_eq!(adapter.trainer.iterations, 7);
    }

    #[test]
    fn adapter_prepare_runs_startup_bucket_validation() {
        let dir = tempfile::tempdir().expect("tempdir");
        let clusters = tempfile::tempdir().expect("clusters");
        let mut config = toy_config(dir.path());
        config.training.cluster_path = Some(clusters.path().to_string_lossy().to_string());

        let mut adapter =
            HeadsUpBlueprintTrainingRuntimeAdapter::new(BlueprintTrainer::new(config));
        let err = adapter.prepare().expect_err("missing bucket files fail");

        assert!(
            err.to_string().contains("No bucket files found for:"),
            "unexpected error: {err}",
        );
    }

    #[test]
    fn adapter_returns_idle_when_target_strategy_delta_already_reached() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut config = toy_config(dir.path());
        config.training.iterations = None;
        config.training.target_strategy_delta = Some(1.0);
        config.training.batch_size = 10;

        let mut trainer = toy_trainer(config);
        trainer.last_strategy_delta = 0.5;

        let mut adapter = HeadsUpBlueprintTrainingRuntimeAdapter::new(trainer);
        adapter.prepare().expect("prepare");

        let budget = BatchBudget::from_limits(&RuntimeLimits::default(), adapter.counters());
        let outcome = adapter.run_next_batch(budget).expect("batch");

        assert_eq!(outcome, BatchOutcome::idle());
        assert_eq!(adapter.trainer.iterations, 0);
        assert_eq!(adapter.counters.load_units(), 0);
    }
}
