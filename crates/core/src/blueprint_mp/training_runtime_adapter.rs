//! Multiplayer lazy sparse adapter for the shared training runtime.

use std::io::{Error as IoError, ErrorKind};
use std::sync::atomic::Ordering;

use crate::training_runtime::{
    BatchBudget, BatchOutcome, RuntimeControls, RuntimeCounters, RuntimeError, RuntimeLimits,
    RuntimeResult, TrainingBackendKind, TrainingRuntimeBackend, TrainingUnit,
};

use super::config::{BlueprintMpConfig, MpTrainingBackend};
use super::trainer::{LazyMpTrainingStepper, LazyTrainContext, setup_lazy_training};

type LazyRuntimeHook = Box<dyn FnMut(&LazyTrainContext) -> RuntimeResult<()> + Send + 'static>;

/// Shared-runtime adapter for lazy sparse multiplayer blueprint training.
pub struct LazySparseMpTrainingRuntimeAdapter {
    config: BlueprintMpConfig,
    ctx: LazyTrainContext,
    limits: RuntimeLimits,
    controls: RuntimeControls,
    counters: RuntimeCounters,
    stepper: Option<LazyMpTrainingStepper>,
    snapshot_hook: Option<LazyRuntimeHook>,
    telemetry_refresh_hook: Option<LazyRuntimeHook>,
    reload_hook: Option<LazyRuntimeHook>,
}

impl LazySparseMpTrainingRuntimeAdapter {
    /// Build a lazy sparse adapter with a fresh training context.
    #[must_use]
    pub fn new(config: BlueprintMpConfig) -> Self {
        let ctx = setup_lazy_training(&config);
        Self::from_context(config, ctx)
    }

    /// Build a lazy sparse adapter around an existing training context.
    #[must_use]
    pub fn from_context(config: BlueprintMpConfig, mut ctx: LazyTrainContext) -> Self {
        let controls = RuntimeControls::new();
        ctx.quit = controls.quit_flag();
        let limits = limits_from_config(&config);
        Self {
            config,
            ctx,
            limits,
            controls,
            counters: RuntimeCounters::new(),
            stepper: None,
            snapshot_hook: None,
            telemetry_refresh_hook: None,
            reload_hook: None,
        }
    }

    /// Return the wrapped lazy training context.
    #[must_use]
    pub const fn context(&self) -> &LazyTrainContext {
        &self.ctx
    }

    /// Return the wrapped config.
    #[must_use]
    pub const fn config(&self) -> &BlueprintMpConfig {
        &self.config
    }

    /// Consume the adapter and return its lazy training context.
    #[must_use]
    pub fn into_context(self) -> LazyTrainContext {
        self.ctx
    }

    /// Install a trainer-owned snapshot hook.
    #[must_use]
    pub fn with_snapshot_hook<F>(mut self, hook: F) -> Self
    where
        F: FnMut(&LazyTrainContext) -> RuntimeResult<()> + Send + 'static,
    {
        self.snapshot_hook = Some(Box::new(hook));
        self
    }

    /// Install a trainer-owned telemetry refresh hook.
    #[must_use]
    pub fn with_telemetry_refresh_hook<F>(mut self, hook: F) -> Self
    where
        F: FnMut(&LazyTrainContext) -> RuntimeResult<()> + Send + 'static,
    {
        self.telemetry_refresh_hook = Some(Box::new(hook));
        self
    }

    /// Install a trainer-owned config reload hook.
    #[must_use]
    pub fn with_reload_hook<F>(mut self, hook: F) -> Self
    where
        F: FnMut(&LazyTrainContext) -> RuntimeResult<()> + Send + 'static,
    {
        self.reload_hook = Some(Box::new(hook));
        self
    }
}

impl TrainingRuntimeBackend for LazySparseMpTrainingRuntimeAdapter {
    fn kind(&self) -> TrainingBackendKind {
        TrainingBackendKind::MultiplayerLazySparse
    }

    fn unit(&self) -> TrainingUnit {
        TrainingUnit::MetaIteration
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
        if self.config.training.backend != MpTrainingBackend::LazySparse {
            return Err(runtime_error(format!(
                "lazy sparse MP runtime adapter requires training.backend=lazy_sparse, got {:?}",
                self.config.training.backend
            )));
        }
        if self.config.snapshots.resume {
            return Err(unsupported_error(
                "lazy sparse MP runtime resume is not implemented in core adapter",
            ));
        }

        let start_meta_iter = self.ctx.iterations.load(Ordering::Relaxed);
        self.counters.seed_units(start_meta_iter);
        self.limits = limits_from_config(&self.config);
        self.stepper = Some(LazyMpTrainingStepper::from_context(
            &self.ctx,
            &self.config.training,
            &self.config.game,
            start_meta_iter,
        ));
        Ok(())
    }

    fn run_next_batch(&mut self, budget: BatchBudget) -> RuntimeResult<BatchOutcome> {
        if self.controls.quit_requested() {
            return Ok(BatchOutcome::idle());
        }
        if matches!(budget.remaining_target_units(), Some(0)) {
            return Ok(BatchOutcome::idle());
        }

        let Some(stepper) = &mut self.stepper else {
            return Err(runtime_error(
                "lazy sparse MP adapter must be prepared before running",
            ));
        };
        let completed = stepper.run_next_batch(budget.remaining_target_units());
        Ok(BatchOutcome::new(completed, completed > 0))
    }

    fn snapshot_now(&mut self) -> RuntimeResult<()> {
        let Some(hook) = &mut self.snapshot_hook else {
            return Err(unsupported_error(
                "lazy sparse MP runtime snapshots require trainer-side snapshot hooks",
            ));
        };
        hook(&self.ctx)
    }

    fn refresh_telemetry(&mut self) -> RuntimeResult<()> {
        if let Some(hook) = &mut self.telemetry_refresh_hook {
            hook(&self.ctx)?;
        }
        Ok(())
    }

    fn reload_config(&mut self) -> RuntimeResult<()> {
        let Some(hook) = &mut self.reload_hook else {
            return Err(unsupported_error(
                "lazy sparse MP runtime config reload requires trainer-side reload hooks",
            ));
        };
        hook(&self.ctx)
    }
}

fn limits_from_config(config: &BlueprintMpConfig) -> RuntimeLimits {
    RuntimeLimits {
        target_units: config.training.iterations,
        time_limit_minutes: config.training.time_limit_minutes,
    }
}

fn runtime_error(message: impl Into<String>) -> RuntimeError {
    IoError::other(message.into()).into()
}

fn unsupported_error(message: &'static str) -> RuntimeError {
    IoError::new(ErrorKind::Unsupported, message).into()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::blueprint_mp::config::{
        BlueprintMpConfig, ForcedBet, ForcedBetKind, MpActionAbstractionConfig,
        MpChanceContinuationMode, MpClusteringConfig, MpGameConfig, MpNegativeActionPurgeMode,
        MpSnapshotConfig, MpSnapshotFormat, MpStreetCluster, MpStreetSizes, MpTrainingBackend,
        MpTrainingConfig,
    };
    use crate::training_runtime::{
        BatchBudget, RuntimeLimits, RuntimeStopReason, TrainingRuntimeBackend, TrainingUnit,
        run_until_stopped,
    };

    use super::*;

    fn toy_config(iterations: u64) -> BlueprintMpConfig {
        let blinds = vec![
            ForcedBet {
                seat: 0,
                kind: ForcedBetKind::SmallBlind,
                amount: 1.0,
            },
            ForcedBet {
                seat: 1,
                kind: ForcedBetKind::BigBlind,
                amount: 2.0,
            },
        ];
        let game = MpGameConfig {
            name: "lazy sparse runtime adapter test".to_string(),
            num_players: 2,
            stack_depth: 20.0,
            allow_preflop_limp: true,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action_abstraction = MpActionAbstractionConfig {
            max_flop_players: None,
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        let clustering = MpClusteringConfig {
            preflop: MpStreetCluster { buckets: 10 },
            flop: MpStreetCluster { buckets: 10 },
            turn: MpStreetCluster { buckets: 10 },
            river: MpStreetCluster { buckets: 10 },
        };
        let training = MpTrainingConfig {
            backend: MpTrainingBackend::LazySparse,
            chance_continuation_mode: MpChanceContinuationMode::SampledFullDeal,
            cluster_path: None,
            iterations: Some(iterations),
            time_limit_minutes: None,
            lcfr_warmup_iterations: 100,
            lcfr_discount_interval: 50,
            dcfr_discount_interval_seconds: None,
            prune_after_iterations: 1_000_000,
            traversal_pruning_enabled: false,
            prune_threshold: -250,
            prune_explore_pct: 0.05,
            negative_action_subtree_purge_enabled: false,
            negative_action_prune_below: -1,
            negative_action_reactivate_at: 0,
            negative_action_purge_mode: MpNegativeActionPurgeMode::ScanHistoryPrefix,
            batch_size: 10,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.0,
            dcfr_gamma: 2.0,
            print_every_minutes: 999,
            purify_threshold: 0.0,
            exploitability_interval_minutes: 0,
            exploitability_samples: 0,
        };
        let snapshots = MpSnapshotConfig {
            warmup_minutes: 999,
            snapshot_every_minutes: 999,
            output_dir: "/tmp/mp_runtime_adapter_test".to_string(),
            resume: false,
            max_snapshots: None,
            format: MpSnapshotFormat::Legacy,
        };
        BlueprintMpConfig {
            game,
            action_abstraction,
            clustering,
            training,
            snapshots,
        }
    }

    fn prepared_adapter(iterations: u64) -> LazySparseMpTrainingRuntimeAdapter {
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(toy_config(iterations));
        adapter.prepare().expect("prepare");
        adapter
    }

    #[test]
    fn adapter_reports_lazy_sparse_meta_iteration_identity() {
        let adapter = LazySparseMpTrainingRuntimeAdapter::new(toy_config(1));

        assert_eq!(adapter.kind(), TrainingBackendKind::MultiplayerLazySparse);
        assert_eq!(adapter.unit(), TrainingUnit::MetaIteration);
        assert_eq!(adapter.limits().target_units, Some(1));
    }

    #[test]
    fn prepare_seeds_counters_from_context_iterations() {
        let config = toy_config(5);
        let ctx = setup_lazy_training(&config);
        ctx.iterations.store(2, Ordering::Relaxed);
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::from_context(config, ctx);

        adapter.prepare().expect("prepare");

        assert_eq!(adapter.counters().load_units(), 2);
        assert_eq!(adapter.context().iterations.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn batch_budget_caps_meta_iterations_without_overshoot() {
        let mut adapter = prepared_adapter(100);
        let budget = BatchBudget::from_limits(
            &RuntimeLimits {
                target_units: Some(3),
                time_limit_minutes: None,
            },
            adapter.counters(),
        );

        let outcome = adapter.run_next_batch(budget).expect("batch");

        assert_eq!(outcome.units_completed, 3);
        assert!(outcome.did_work);
        assert_eq!(adapter.counters().load_units(), 0);
        assert_eq!(adapter.context().iterations.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn run_until_stopped_reaches_target_without_double_counting() {
        let mut config = toy_config(3);
        config.training.batch_size = 2;
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(config);
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 3,
                target_units: 3,
            }
        );
        assert_eq!(adapter.counters().load_units(), 3);
        assert_eq!(adapter.context().iterations.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn zero_target_leaves_sparse_storage_empty() {
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(toy_config(0));
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 0,
                target_units: 0,
            }
        );
        assert_eq!(adapter.context().storage.entry_count(), 0);
    }

    #[test]
    fn quit_before_batch_does_not_run() {
        let mut adapter = prepared_adapter(10);
        adapter.controls().request_quit();
        let outcome = adapter
            .run_next_batch(BatchBudget::from_limits(
                adapter.limits(),
                adapter.counters(),
            ))
            .expect("quit batch");

        assert_eq!(outcome, BatchOutcome::idle());
        assert_eq!(adapter.context().storage.entry_count(), 0);
        assert_eq!(adapter.context().iterations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn snapshot_without_hook_is_explicitly_unsupported() {
        let mut adapter = prepared_adapter(1);

        let err = adapter.snapshot_now().expect_err("snapshot unsupported");

        assert!(
            err.to_string().contains("snapshot hooks"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn snapshot_hook_is_invoked_by_runtime_request() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(toy_config(10));
        let controls = adapter.controls().clone();
        let hook_controls = controls.clone();
        let hook_calls = Arc::clone(&calls);
        adapter = adapter.with_snapshot_hook(move |_| {
            hook_calls.fetch_add(1, Ordering::SeqCst);
            hook_controls.request_quit();
            Ok(())
        });
        controls.request_snapshot();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(reason, RuntimeStopReason::QuitRequested);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(adapter.context().iterations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn refresh_hook_is_invoked_by_runtime_request() {
        let calls = Arc::new(AtomicU64::new(0));
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(toy_config(10));
        let controls = adapter.controls().clone();
        let hook_controls = controls.clone();
        let hook_calls = Arc::clone(&calls);
        adapter = adapter.with_telemetry_refresh_hook(move |_| {
            hook_calls.fetch_add(1, Ordering::SeqCst);
            hook_controls.request_quit();
            Ok(())
        });
        controls.request_strategy_refresh();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(reason, RuntimeStopReason::QuitRequested);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(adapter.context().iterations.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn runtime_pause_prevents_progress_until_resumed() {
        let mut config = toy_config(50);
        config.training.batch_size = 1;
        let adapter = LazySparseMpTrainingRuntimeAdapter::new(config);
        let controls = adapter.controls().clone();
        let iterations = Arc::clone(&adapter.context().iterations);
        controls.request_pause();

        let handle = thread::spawn(move || {
            let mut adapter = adapter;
            let mut telemetry = Vec::new();
            run_until_stopped(&mut adapter, &mut telemetry)
        });

        thread::sleep(Duration::from_millis(30));
        assert_eq!(iterations.load(Ordering::Relaxed), 0);
        controls.resume();

        let deadline = Instant::now() + Duration::from_secs(1);
        while iterations.load(Ordering::Relaxed) == 0 && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(5));
        }
        controls.request_quit();
        let reason = handle
            .join()
            .expect("runtime thread panicked")
            .expect("runtime");

        assert!(iterations.load(Ordering::Relaxed) > 0);
        assert!(
            matches!(
                reason,
                RuntimeStopReason::QuitRequested
                    | RuntimeStopReason::TargetUnitsReached {
                        current_units: 50,
                        target_units: 50
                    }
            ),
            "unexpected stop reason: {reason:?}"
        );
    }

    #[test]
    fn runtime_quit_request_stops_before_work() {
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(toy_config(10));
        adapter.controls().request_quit();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(reason, RuntimeStopReason::QuitRequested);
        assert_eq!(adapter.context().iterations.load(Ordering::Relaxed), 0);
        assert_eq!(adapter.context().storage.entry_count(), 0);
    }

    #[test]
    fn prepare_rejects_resume_until_loader_exists() {
        let mut config = toy_config(1);
        config.snapshots.resume = true;
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(config);

        let err = adapter.prepare().expect_err("resume unsupported");

        assert!(
            err.to_string().contains("resume is not implemented"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn prepare_rejects_non_lazy_sparse_backend() {
        let mut config = toy_config(1);
        config.training.backend = MpTrainingBackend::Eager;
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(config);

        let err = adapter.prepare().expect_err("backend mismatch");

        assert!(
            err.to_string().contains("training.backend=lazy_sparse"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn adapter_runs_sampled_turn_exact_river_mode() {
        let mut config = toy_config(1);
        config.training.batch_size = 1;
        config.training.chance_continuation_mode = MpChanceContinuationMode::SampledTurnExactRiver;
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(config);
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 1,
                target_units: 1,
            }
        );
    }

    #[test]
    fn adapter_runs_sampled_flop_exact_turn_river_mode() {
        let mut config = toy_config(1);
        config.training.batch_size = 1;
        config.training.chance_continuation_mode =
            MpChanceContinuationMode::SampledFlopExactTurnRiver;
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(config);
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 1,
                target_units: 1,
            }
        );
    }

    #[test]
    fn run_lazy_training_and_adapter_reach_same_tiny_target() {
        let mut direct_config = toy_config(2);
        direct_config.training.batch_size = 1;
        let direct_ctx = setup_lazy_training(&direct_config);
        let direct_result = super::super::trainer::run_lazy_training(
            &direct_ctx,
            &direct_config.training,
            &direct_config.game,
        );

        let mut adapter_config = toy_config(2);
        adapter_config.training.batch_size = 1;
        let mut adapter = LazySparseMpTrainingRuntimeAdapter::new(adapter_config);
        let mut telemetry = Vec::new();
        let reason = run_until_stopped(&mut adapter, &mut telemetry).expect("runtime completes");

        assert_eq!(direct_result.meta_iterations, 2);
        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 2,
                target_units: 2,
            }
        );
        assert_eq!(
            direct_ctx.iterations.load(Ordering::Relaxed),
            adapter.context().iterations.load(Ordering::Relaxed)
        );
        assert!(!direct_ctx.storage.snapshot_entries().is_empty());
        assert!(!adapter.context().storage.snapshot_entries().is_empty());
    }
}
