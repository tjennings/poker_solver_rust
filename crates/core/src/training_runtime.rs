#![allow(clippy::module_name_repetitions)]

use std::error::Error;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Error type shared by backend-neutral training runtime hooks.
pub type RuntimeError = Box<dyn Error + Send + Sync>;

/// Result type shared by backend-neutral training runtime hooks.
pub type RuntimeResult<T> = Result<T, RuntimeError>;

/// Solver backend family used by a training runtime.
///
/// This is telemetry/display identity only. The shared runtime does not branch on these variants
/// or attach traversal/storage semantics to them.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingBackendKind {
    /// Existing heads-up blueprint v2 trainer.
    HeadsUpBlueprintV2,
    /// Multiplayer trainer that eagerly materializes traversal state.
    MultiplayerEager,
    /// Multiplayer trainer that uses lazy sparse traversal state.
    MultiplayerLazySparse,
}

impl TrainingBackendKind {
    /// Human-readable backend label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::HeadsUpBlueprintV2 => "heads-up blueprint v2",
            Self::MultiplayerEager => "multiplayer eager",
            Self::MultiplayerLazySparse => "multiplayer lazy sparse",
        }
    }
}

/// Unit of progress reported by a training backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TrainingUnit {
    /// One concrete trainer iteration.
    Iteration,
    /// One higher-level iteration that may contain many concrete traversals.
    MetaIteration,
}

impl TrainingUnit {
    /// Singular human-readable label.
    #[must_use]
    pub const fn singular_label(self) -> &'static str {
        match self {
            Self::Iteration => "iteration",
            Self::MetaIteration => "meta-iteration",
        }
    }

    /// Plural human-readable label.
    #[must_use]
    pub const fn plural_label(self) -> &'static str {
        match self {
            Self::Iteration => "iterations",
            Self::MetaIteration => "meta-iterations",
        }
    }

    /// Human-readable label for a concrete count.
    #[must_use]
    pub const fn label_for_count(self, count: u64) -> &'static str {
        if count == 1 {
            self.singular_label()
        } else {
            self.plural_label()
        }
    }
}

/// Backend-neutral stop limits for a training run.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct RuntimeLimits {
    /// Target number of backend progress units to complete.
    pub target_units: Option<u64>,
    /// Wall-clock time limit in minutes.
    pub time_limit_minutes: Option<u64>,
}

impl RuntimeLimits {
    /// Return the remaining target progress units, if a target is configured.
    ///
    /// Backends can use this before selecting a batch size to avoid overshooting a finite target.
    #[must_use]
    pub fn remaining_target_units(&self, counters: &RuntimeCounters) -> Option<u64> {
        self.target_units
            .map(|target_units| target_units.saturating_sub(counters.load_units()))
    }
}

/// Shared runtime controls, intended to be cloned into signal handlers or UI code.
#[derive(Clone, Debug)]
pub struct RuntimeControls {
    pause: Arc<AtomicBool>,
    quit: Arc<AtomicBool>,
    snapshot: Arc<AtomicBool>,
    strategy_refresh: Arc<AtomicBool>,
    config_reload: Arc<AtomicBool>,
}

impl Default for RuntimeControls {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeControls {
    /// Create a fresh set of runtime controls.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pause: Arc::new(AtomicBool::new(false)),
            quit: Arc::new(AtomicBool::new(false)),
            snapshot: Arc::new(AtomicBool::new(false)),
            strategy_refresh: Arc::new(AtomicBool::new(false)),
            config_reload: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Clone the pause flag for integration with signal handlers or UI code.
    ///
    /// Prefer [`Self::request_pause`] and [`Self::resume`] when direct atomic sharing is not
    /// required.
    #[must_use]
    pub fn pause_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.pause)
    }

    /// Clone the quit flag for integration with signal handlers or UI code.
    ///
    /// Prefer [`Self::request_quit`] when direct atomic sharing is not required.
    #[must_use]
    pub fn quit_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.quit)
    }

    /// Clone the snapshot flag for integration with signal handlers or UI code.
    ///
    /// Prefer [`Self::request_snapshot`] when direct atomic sharing is not required.
    #[must_use]
    pub fn snapshot_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.snapshot)
    }

    /// Clone the strategy refresh flag for integration with signal handlers or UI code.
    ///
    /// Prefer [`Self::request_strategy_refresh`] when direct atomic sharing is not required.
    #[must_use]
    pub fn strategy_refresh_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.strategy_refresh)
    }

    /// Clone the config reload flag for integration with signal handlers or UI code.
    ///
    /// Prefer [`Self::request_config_reload`] when direct atomic sharing is not required.
    #[must_use]
    pub fn config_reload_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.config_reload)
    }

    /// Request that the runtime pause before starting more work.
    pub fn request_pause(&self) {
        self.pause.store(true, Ordering::SeqCst);
    }

    /// Clear a pending pause request.
    pub fn resume(&self) {
        self.pause.store(false, Ordering::SeqCst);
    }

    /// Return whether the runtime is currently paused.
    #[must_use]
    pub fn is_paused(&self) -> bool {
        self.pause.load(Ordering::SeqCst)
    }

    /// Request that the runtime stop.
    pub fn request_quit(&self) {
        self.quit.store(true, Ordering::SeqCst);
    }

    /// Return whether a quit has been requested.
    #[must_use]
    pub fn quit_requested(&self) -> bool {
        self.quit.load(Ordering::SeqCst)
    }

    /// Request an immediate snapshot.
    pub fn request_snapshot(&self) {
        self.snapshot.store(true, Ordering::SeqCst);
    }

    /// Consume and return a pending snapshot request.
    #[must_use]
    pub fn take_snapshot_request(&self) -> bool {
        self.snapshot.swap(false, Ordering::SeqCst)
    }

    /// Request a strategy telemetry refresh.
    pub fn request_strategy_refresh(&self) {
        self.strategy_refresh.store(true, Ordering::SeqCst);
    }

    /// Consume and return a pending strategy refresh request.
    #[must_use]
    pub fn take_strategy_refresh_request(&self) -> bool {
        self.strategy_refresh.swap(false, Ordering::SeqCst)
    }

    /// Request a config reload.
    pub fn request_config_reload(&self) {
        self.config_reload.store(true, Ordering::SeqCst);
    }

    /// Consume and return a pending config reload request.
    #[must_use]
    pub fn take_config_reload_request(&self) -> bool {
        self.config_reload.swap(false, Ordering::SeqCst)
    }
}

/// Shared runtime counters.
///
/// During a runtime run, [`run_until_stopped`] is the sole owner of applying completed progress to
/// these counters. Backend hooks report progress through [`BatchOutcome::units_completed`] instead
/// of mutating counters directly.
#[derive(Clone, Debug, Default)]
pub struct RuntimeCounters {
    units: Arc<AtomicU64>,
}

impl RuntimeCounters {
    /// Create counters initialized to zero completed units.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Load the number of completed units.
    #[must_use]
    pub fn load_units(&self) -> u64 {
        self.units.load(Ordering::SeqCst)
    }

    /// Seed the number of completed units before a runtime starts or resumes.
    ///
    /// Adapters may call this to initialize counters from restored snapshot iteration or
    /// meta-iteration totals before [`run_until_stopped`] starts running batches. Backend hooks
    /// must not call this during `run_next_batch`, `after_batch`, `snapshot_now`,
    /// `refresh_telemetry`, or `reload_config`; active-run progress must flow through
    /// [`BatchOutcome::units_completed`] so the runtime can count it exactly once.
    pub fn seed_units(&self, units: u64) {
        self.units.store(units, Ordering::SeqCst);
    }

    /// Add completed units and return the new total.
    fn add_units(&self, units: u64) -> u64 {
        self.units.fetch_add(units, Ordering::SeqCst) + units
    }
}

/// Budget for one backend-neutral training batch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BatchBudget {
    remaining_target_units: Option<u64>,
}

impl BatchBudget {
    /// Create a batch budget from runtime limits and counters.
    #[must_use]
    pub fn from_limits(limits: &RuntimeLimits, counters: &RuntimeCounters) -> Self {
        Self {
            remaining_target_units: limits.remaining_target_units(counters),
        }
    }

    /// Remaining target progress units before this batch starts, if a target is configured.
    ///
    /// Backends should cap finite batches to this value when their batch size is adjustable.
    #[must_use]
    pub const fn remaining_target_units(self) -> Option<u64> {
        self.remaining_target_units
    }
}

/// Result of one backend-neutral training batch.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct BatchOutcome {
    /// Progress units completed by this batch.
    ///
    /// Backends report progress here only. [`run_until_stopped`] is the sole owner of applying
    /// this value to [`RuntimeCounters`].
    pub units_completed: u64,
    /// Whether the backend performed useful work.
    pub did_work: bool,
    /// Optional short batch label.
    pub batch_label: Option<String>,
    /// Optional backend-neutral note for logs or telemetry.
    pub note: Option<String>,
}

impl BatchOutcome {
    /// Create a batch outcome.
    #[must_use]
    pub const fn new(units_completed: u64, did_work: bool) -> Self {
        Self {
            units_completed,
            did_work,
            batch_label: None,
            note: None,
        }
    }

    /// Create an idle batch outcome.
    #[must_use]
    pub const fn idle() -> Self {
        Self::new(0, false)
    }
}

/// Reason a backend-neutral runtime stopped.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeStopReason {
    /// A quit request was observed.
    QuitRequested,
    /// The target number of progress units was reached.
    TargetUnitsReached {
        /// Completed progress units.
        current_units: u64,
        /// Configured target progress units.
        target_units: u64,
    },
    /// The wall-clock time limit was reached.
    TimeLimitReached {
        /// Elapsed whole minutes.
        elapsed_minutes: u64,
        /// Configured whole-minute limit.
        limit_minutes: u64,
    },
    /// The backend reported that no work is currently available.
    NoWorkAvailable,
}

/// Backend-neutral telemetry event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RuntimeTelemetryEvent {
    /// The backend completed preparation.
    Prepared {
        /// Runtime backend kind.
        kind: TrainingBackendKind,
        /// Runtime progress unit.
        unit: TrainingUnit,
    },
    /// Runtime entered a pause loop.
    Paused,
    /// Runtime exited a pause loop.
    Resumed,
    /// A snapshot request was consumed.
    SnapshotRequested,
    /// The backend completed a snapshot.
    SnapshotCompleted,
    /// A strategy refresh request was consumed.
    StrategyRefreshRequested,
    /// The backend refreshed telemetry.
    TelemetryRefreshed,
    /// A config reload request was consumed.
    ConfigReloadRequested,
    /// A batch completed.
    BatchCompleted {
        /// Batch outcome.
        outcome: BatchOutcome,
        /// Total completed progress units after applying the batch.
        total_units: u64,
    },
    /// Runtime stopped.
    Stopped {
        /// Stop reason.
        reason: RuntimeStopReason,
    },
}

/// Sink for backend-neutral runtime telemetry.
pub trait RuntimeTelemetrySink {
    /// Record one telemetry event.
    fn record(&mut self, event: RuntimeTelemetryEvent);
}

impl RuntimeTelemetrySink for () {
    fn record(&mut self, _event: RuntimeTelemetryEvent) {}
}

impl RuntimeTelemetrySink for Vec<RuntimeTelemetryEvent> {
    fn record(&mut self, event: RuntimeTelemetryEvent) {
        self.push(event);
    }
}

/// Backend-neutral training runtime contract.
pub trait TrainingRuntimeBackend {
    /// Runtime backend kind.
    fn kind(&self) -> TrainingBackendKind;

    /// Runtime progress unit.
    fn unit(&self) -> TrainingUnit;

    /// Runtime stop limits.
    fn limits(&self) -> &RuntimeLimits;

    /// Runtime controls.
    fn controls(&self) -> &RuntimeControls;

    /// Runtime counters.
    ///
    /// Backends expose counters so the shared runtime can evaluate stop conditions and apply
    /// [`BatchOutcome::units_completed`]. Adapters may seed counters from restored snapshot totals
    /// before starting or resuming the runtime. Backend implementations must not mutate counters
    /// during `run_next_batch`, `after_batch`, or request hooks; doing so would double-count
    /// progress.
    fn counters(&self) -> &RuntimeCounters;

    /// Prepare the backend before the first batch.
    ///
    /// # Errors
    ///
    /// Returns an error when backend preparation fails.
    fn prepare(&mut self) -> RuntimeResult<()> {
        Ok(())
    }

    /// Run the next unit of backend work.
    ///
    /// # Errors
    ///
    /// Returns an error when backend batch execution fails.
    fn run_next_batch(&mut self, budget: BatchBudget) -> RuntimeResult<BatchOutcome>;

    /// Observe a completed batch.
    ///
    /// # Errors
    ///
    /// Returns an error when backend post-batch handling fails.
    fn after_batch(&mut self, _outcome: &BatchOutcome) -> RuntimeResult<()> {
        Ok(())
    }

    /// Snapshot backend state immediately.
    ///
    /// # Errors
    ///
    /// Returns an error when snapshotting fails.
    fn snapshot_now(&mut self) -> RuntimeResult<()> {
        Ok(())
    }

    /// Refresh backend telemetry.
    ///
    /// # Errors
    ///
    /// Returns an error when telemetry refresh fails.
    fn refresh_telemetry(&mut self) -> RuntimeResult<()> {
        Ok(())
    }

    /// Reload backend configuration after a config reload request is consumed.
    ///
    /// # Errors
    ///
    /// Returns an error when config reload fails.
    fn reload_config(&mut self) -> RuntimeResult<()> {
        Ok(())
    }
}

/// Return the stop reason if the runtime should stop now.
#[must_use]
pub fn should_stop(
    limits: &RuntimeLimits,
    controls: &RuntimeControls,
    counters: &RuntimeCounters,
    started_at: Instant,
) -> Option<RuntimeStopReason> {
    if controls.quit_requested() {
        return Some(RuntimeStopReason::QuitRequested);
    }

    if let Some(target_units) = limits.target_units {
        let current_units = counters.load_units();
        if current_units >= target_units {
            return Some(RuntimeStopReason::TargetUnitsReached {
                current_units,
                target_units,
            });
        }
    }

    if let Some(limit_minutes) = limits.time_limit_minutes {
        let elapsed_minutes = started_at.elapsed().as_secs() / 60;
        if elapsed_minutes >= limit_minutes {
            return Some(RuntimeStopReason::TimeLimitReached {
                elapsed_minutes,
                limit_minutes,
            });
        }
    }

    None
}

/// Wait while the pause flag is set, waking periodically to observe quit requests.
pub fn wait_if_paused<S>(controls: &RuntimeControls, telemetry: &mut S)
where
    S: RuntimeTelemetrySink + ?Sized,
{
    wait_if_paused_with_interval(controls, telemetry, Duration::from_millis(10));
}

/// Wait while paused with an explicit poll interval.
pub fn wait_if_paused_with_interval<S>(
    controls: &RuntimeControls,
    telemetry: &mut S,
    poll_interval: Duration,
) where
    S: RuntimeTelemetrySink + ?Sized,
{
    if !controls.is_paused() {
        return;
    }

    telemetry.record(RuntimeTelemetryEvent::Paused);
    while controls.is_paused() && !controls.quit_requested() {
        thread::sleep(poll_interval);
    }

    if !controls.is_paused() {
        telemetry.record(RuntimeTelemetryEvent::Resumed);
    }
}

/// Run a backend until a generic stop condition is reached.
///
/// # Errors
///
/// Returns an error when any backend hook fails.
pub fn run_until_stopped<B, S>(
    backend: &mut B,
    telemetry: &mut S,
) -> RuntimeResult<RuntimeStopReason>
where
    B: TrainingRuntimeBackend + ?Sized,
    S: RuntimeTelemetrySink + ?Sized,
{
    backend.prepare()?;
    telemetry.record(RuntimeTelemetryEvent::Prepared {
        kind: backend.kind(),
        unit: backend.unit(),
    });

    let started_at = Instant::now();
    let mut pause_recorded = false;
    loop {
        if let Some(reason) = should_stop(
            backend.limits(),
            backend.controls(),
            backend.counters(),
            started_at,
        ) {
            telemetry.record(RuntimeTelemetryEvent::Stopped {
                reason: reason.clone(),
            });
            return Ok(reason);
        }

        if backend.controls().take_snapshot_request() {
            telemetry.record(RuntimeTelemetryEvent::SnapshotRequested);
            backend.snapshot_now()?;
            telemetry.record(RuntimeTelemetryEvent::SnapshotCompleted);
        }

        if backend.controls().take_strategy_refresh_request() {
            telemetry.record(RuntimeTelemetryEvent::StrategyRefreshRequested);
            backend.refresh_telemetry()?;
            telemetry.record(RuntimeTelemetryEvent::TelemetryRefreshed);
        }

        if backend.controls().take_config_reload_request() {
            telemetry.record(RuntimeTelemetryEvent::ConfigReloadRequested);
            backend.reload_config()?;
        }

        if let Some(reason) = should_stop(
            backend.limits(),
            backend.controls(),
            backend.counters(),
            started_at,
        ) {
            telemetry.record(RuntimeTelemetryEvent::Stopped {
                reason: reason.clone(),
            });
            return Ok(reason);
        }

        if backend.controls().is_paused() {
            if !pause_recorded {
                telemetry.record(RuntimeTelemetryEvent::Paused);
                pause_recorded = true;
            }
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        if pause_recorded {
            telemetry.record(RuntimeTelemetryEvent::Resumed);
            pause_recorded = false;
        }

        let budget = BatchBudget::from_limits(backend.limits(), backend.counters());
        let outcome = backend.run_next_batch(budget)?;
        if outcome.units_completed > 0 {
            backend.counters().add_units(outcome.units_completed);
        }
        let total_units = backend.counters().load_units();
        telemetry.record(RuntimeTelemetryEvent::BatchCompleted {
            outcome: outcome.clone(),
            total_units,
        });
        backend.after_batch(&outcome)?;

        if !outcome.did_work {
            let reason = RuntimeStopReason::NoWorkAvailable;
            telemetry.record(RuntimeTelemetryEvent::Stopped {
                reason: reason.clone(),
            });
            return Ok(reason);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    struct FakeBackend {
        limits: RuntimeLimits,
        controls: RuntimeControls,
        counters: RuntimeCounters,
        batches: VecDeque<BatchOutcome>,
        prepare_count: u64,
        batch_count: u64,
        after_batch_count: u64,
        snapshot_count: u64,
        refresh_count: u64,
        reload_count: u64,
        quit_after_snapshot: bool,
        quit_after_refresh: bool,
        quit_after_reload: bool,
        cap_batches_to_budget: bool,
        batch_budgets: Vec<BatchBudget>,
    }

    impl FakeBackend {
        fn new(limits: RuntimeLimits, batches: impl IntoIterator<Item = BatchOutcome>) -> Self {
            Self {
                limits,
                controls: RuntimeControls::new(),
                counters: RuntimeCounters::new(),
                batches: batches.into_iter().collect(),
                prepare_count: 0,
                batch_count: 0,
                after_batch_count: 0,
                snapshot_count: 0,
                refresh_count: 0,
                reload_count: 0,
                quit_after_snapshot: false,
                quit_after_refresh: false,
                quit_after_reload: false,
                cap_batches_to_budget: false,
                batch_budgets: Vec::new(),
            }
        }
    }

    impl TrainingRuntimeBackend for FakeBackend {
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
            self.prepare_count += 1;
            Ok(())
        }

        fn run_next_batch(&mut self, budget: BatchBudget) -> RuntimeResult<BatchOutcome> {
            self.batch_count += 1;
            self.batch_budgets.push(budget);
            let mut outcome = self.batches.pop_front().unwrap_or_else(BatchOutcome::idle);
            if self.cap_batches_to_budget {
                if let Some(remaining_target_units) = budget.remaining_target_units() {
                    outcome.units_completed = outcome.units_completed.min(remaining_target_units);
                    outcome.did_work = outcome.did_work && remaining_target_units > 0;
                }
            }
            Ok(outcome)
        }

        fn after_batch(&mut self, _outcome: &BatchOutcome) -> RuntimeResult<()> {
            self.after_batch_count += 1;
            Ok(())
        }

        fn snapshot_now(&mut self) -> RuntimeResult<()> {
            self.snapshot_count += 1;
            if self.quit_after_snapshot {
                self.controls.request_quit();
            }
            Ok(())
        }

        fn refresh_telemetry(&mut self) -> RuntimeResult<()> {
            self.refresh_count += 1;
            if self.quit_after_refresh {
                self.controls.request_quit();
            }
            Ok(())
        }

        fn reload_config(&mut self) -> RuntimeResult<()> {
            self.reload_count += 1;
            if self.quit_after_reload {
                self.controls.request_quit();
            }
            Ok(())
        }
    }

    #[test]
    fn runtime_controls_request_and_take_triggers() {
        let controls = RuntimeControls::new();
        let pause_flag = controls.pause_flag();
        let quit_flag = controls.quit_flag();
        let snapshot_flag = controls.snapshot_flag();
        let strategy_refresh_flag = controls.strategy_refresh_flag();
        let config_reload_flag = controls.config_reload_flag();

        assert!(!controls.is_paused());
        assert!(!controls.quit_requested());
        assert!(!controls.take_snapshot_request());
        assert!(!controls.take_strategy_refresh_request());
        assert!(!controls.take_config_reload_request());

        controls.request_pause();
        assert!(controls.is_paused());
        assert!(pause_flag.load(Ordering::SeqCst));
        controls.resume();
        assert!(!controls.is_paused());

        controls.request_quit();
        assert!(controls.quit_requested());
        assert!(quit_flag.load(Ordering::SeqCst));

        controls.request_snapshot();
        assert!(snapshot_flag.load(Ordering::SeqCst));
        assert!(controls.take_snapshot_request());
        assert!(!controls.take_snapshot_request());

        controls.request_strategy_refresh();
        assert!(strategy_refresh_flag.load(Ordering::SeqCst));
        assert!(controls.take_strategy_refresh_request());
        assert!(!controls.take_strategy_refresh_request());

        controls.request_config_reload();
        assert!(config_reload_flag.load(Ordering::SeqCst));
        assert!(controls.take_config_reload_request());
        assert!(!controls.take_config_reload_request());
    }

    #[test]
    fn runtime_counters_load_store_and_add_units() {
        let counters = RuntimeCounters::new();
        assert_eq!(counters.load_units(), 0);

        counters.seed_units(7);
        assert_eq!(counters.load_units(), 7);
        assert_eq!(counters.add_units(5), 12);
        assert_eq!(counters.load_units(), 12);
    }

    #[test]
    fn should_stop_when_target_units_are_reached() {
        let limits = RuntimeLimits {
            target_units: Some(2),
            time_limit_minutes: None,
        };
        let controls = RuntimeControls::new();
        let counters = RuntimeCounters::new();
        let started_at = Instant::now();

        assert_eq!(should_stop(&limits, &controls, &counters, started_at), None);
        counters.seed_units(2);
        assert_eq!(
            should_stop(&limits, &controls, &counters, started_at),
            Some(RuntimeStopReason::TargetUnitsReached {
                current_units: 2,
                target_units: 2,
            })
        );
    }

    #[test]
    fn run_until_stopped_stops_at_target_units() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: Some(2),
                time_limit_minutes: None,
            },
            [
                BatchOutcome::new(1, true),
                BatchOutcome::new(1, true),
                BatchOutcome::new(1, true),
            ],
        );
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 2,
                target_units: 2,
            }
        );
        assert_eq!(backend.counters.load_units(), 2);
        assert_eq!(backend.after_batch_count, 2);
        assert_eq!(backend.batches.len(), 1);
    }

    #[test]
    fn run_until_stopped_honors_target_reached_before_first_batch() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: Some(5),
                time_limit_minutes: None,
            },
            [BatchOutcome::new(1, true)],
        );
        backend.counters.seed_units(5);
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 5,
                target_units: 5,
            }
        );
        assert_eq!(backend.prepare_count, 1);
        assert_eq!(backend.batch_count, 0);
        assert_eq!(backend.after_batch_count, 0);
    }

    #[test]
    fn run_until_stopped_uses_seeded_units_for_target_progress() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: Some(6),
                time_limit_minutes: None,
            },
            [
                BatchOutcome::new(1, true),
                BatchOutcome::new(1, true),
                BatchOutcome::new(1, true),
            ],
        );
        backend.counters.seed_units(4);
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 6,
                target_units: 6,
            }
        );
        assert_eq!(backend.counters.load_units(), 6);
        assert_eq!(backend.batch_count, 2);
        assert_eq!(backend.after_batch_count, 2);
        assert_eq!(
            backend.batch_budgets,
            vec![
                BatchBudget {
                    remaining_target_units: Some(2),
                },
                BatchBudget {
                    remaining_target_units: Some(1),
                },
            ]
        );
    }

    #[test]
    fn batch_budget_exposes_remaining_target_units_to_backend() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: Some(5),
                time_limit_minutes: None,
            },
            [BatchOutcome::new(10, true)],
        );
        backend.counters.seed_units(3);
        backend.cap_batches_to_budget = true;
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 5,
                target_units: 5,
            }
        );
        assert_eq!(backend.counters.load_units(), 5);
        assert_eq!(backend.batch_budgets[0].remaining_target_units(), Some(2));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::BatchCompleted {
            outcome: BatchOutcome::new(2, true),
            total_units: 5,
        }));
    }

    #[test]
    fn run_until_stopped_honors_time_limit_before_batch() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: None,
                time_limit_minutes: Some(0),
            },
            [BatchOutcome::new(1, true)],
        );
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TimeLimitReached {
                elapsed_minutes: 0,
                limit_minutes: 0,
            }
        );
        assert_eq!(backend.prepare_count, 1);
        assert_eq!(backend.batch_count, 0);
        assert_eq!(backend.after_batch_count, 0);
    }

    #[test]
    fn run_until_stopped_honors_quit_before_work() {
        let mut backend = FakeBackend::new(
            RuntimeLimits::default(),
            [BatchOutcome::new(1, true), BatchOutcome::idle()],
        );
        backend.controls.request_quit();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(reason, RuntimeStopReason::QuitRequested);
        assert_eq!(backend.prepare_count, 1);
        assert_eq!(backend.counters.load_units(), 0);
        assert_eq!(backend.batch_count, 0);
        assert_eq!(backend.after_batch_count, 0);
    }

    #[test]
    fn run_until_stopped_honors_quit_while_paused_without_batch() {
        let mut backend = FakeBackend::new(RuntimeLimits::default(), [BatchOutcome::new(1, true)]);
        backend.controls.request_pause();
        let controls = backend.controls.clone();
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(5));
            controls.request_quit();
        });
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();
        handle.join().unwrap();

        assert_eq!(reason, RuntimeStopReason::QuitRequested);
        assert_eq!(backend.counters.load_units(), 0);
        assert_eq!(backend.batch_count, 0);
        assert_eq!(backend.after_batch_count, 0);
        assert!(telemetry.contains(&RuntimeTelemetryEvent::Paused));
    }

    #[test]
    fn wait_if_paused_blocks_until_resume() {
        let controls = RuntimeControls::new();
        controls.request_pause();
        let resume_controls = controls.clone();
        let handle = thread::spawn(move || {
            thread::sleep(Duration::from_millis(5));
            resume_controls.resume();
        });
        let mut telemetry = Vec::new();

        wait_if_paused_with_interval(&controls, &mut telemetry, Duration::from_millis(1));
        handle.join().unwrap();

        assert!(!controls.is_paused());
        assert_eq!(
            telemetry,
            vec![
                RuntimeTelemetryEvent::Paused,
                RuntimeTelemetryEvent::Resumed
            ]
        );
    }

    #[test]
    fn snapshot_request_flows_to_backend_snapshot_hook() {
        let mut backend = FakeBackend::new(RuntimeLimits::default(), [BatchOutcome::idle()]);
        backend.controls.request_snapshot();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(reason, RuntimeStopReason::NoWorkAvailable);
        assert_eq!(backend.snapshot_count, 1);
        assert!(telemetry.contains(&RuntimeTelemetryEvent::SnapshotRequested));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::SnapshotCompleted));
    }

    #[test]
    fn paused_runtime_services_snapshot_and_refresh_without_batch() {
        let mut backend = FakeBackend::new(RuntimeLimits::default(), [BatchOutcome::new(1, true)]);
        backend.controls.request_pause();
        backend.controls.request_snapshot();
        backend.controls.request_strategy_refresh();
        backend.quit_after_refresh = true;
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(reason, RuntimeStopReason::QuitRequested);
        assert_eq!(backend.snapshot_count, 1);
        assert_eq!(backend.refresh_count, 1);
        assert_eq!(backend.counters.load_units(), 0);
        assert_eq!(backend.batch_count, 0);
        assert_eq!(backend.after_batch_count, 0);
        assert!(telemetry.contains(&RuntimeTelemetryEvent::SnapshotRequested));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::SnapshotCompleted));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::StrategyRefreshRequested));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::TelemetryRefreshed));
    }

    #[test]
    fn config_reload_request_flows_to_backend_reload_hook_once() {
        let mut backend = FakeBackend::new(RuntimeLimits::default(), [BatchOutcome::idle()]);
        backend.controls.request_config_reload();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(reason, RuntimeStopReason::NoWorkAvailable);
        assert_eq!(backend.reload_count, 1);
        assert!(!backend.controls.take_config_reload_request());
        assert!(telemetry.contains(&RuntimeTelemetryEvent::ConfigReloadRequested));
    }

    #[test]
    fn runtime_applies_batch_outcome_units_exactly_once() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: Some(3),
                time_limit_minutes: None,
            },
            [BatchOutcome::new(3, true)],
        );
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 3,
                target_units: 3,
            }
        );
        assert_eq!(backend.counters.load_units(), 3);
        assert_eq!(backend.batch_count, 1);
        assert_eq!(backend.after_batch_count, 1);
        assert!(telemetry.contains(&RuntimeTelemetryEvent::BatchCompleted {
            outcome: BatchOutcome::new(3, true),
            total_units: 3,
        }));
    }

    #[test]
    fn runtime_events_are_recorded_in_execution_order() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: Some(1),
                time_limit_minutes: None,
            },
            [BatchOutcome::new(1, true)],
        );
        backend.controls.request_snapshot();
        backend.controls.request_strategy_refresh();
        backend.controls.request_config_reload();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 1,
                target_units: 1,
            }
        );
        assert_eq!(
            telemetry,
            vec![
                RuntimeTelemetryEvent::Prepared {
                    kind: TrainingBackendKind::HeadsUpBlueprintV2,
                    unit: TrainingUnit::Iteration,
                },
                RuntimeTelemetryEvent::SnapshotRequested,
                RuntimeTelemetryEvent::SnapshotCompleted,
                RuntimeTelemetryEvent::StrategyRefreshRequested,
                RuntimeTelemetryEvent::TelemetryRefreshed,
                RuntimeTelemetryEvent::ConfigReloadRequested,
                RuntimeTelemetryEvent::BatchCompleted {
                    outcome: BatchOutcome::new(1, true),
                    total_units: 1,
                },
                RuntimeTelemetryEvent::Stopped {
                    reason: RuntimeStopReason::TargetUnitsReached {
                        current_units: 1,
                        target_units: 1,
                    },
                },
            ]
        );
    }

    #[test]
    fn telemetry_sink_receives_runtime_events() {
        let mut backend = FakeBackend::new(
            RuntimeLimits {
                target_units: Some(1),
                time_limit_minutes: None,
            },
            [BatchOutcome::new(1, true)],
        );
        backend.controls.request_strategy_refresh();
        backend.controls.request_config_reload();
        let mut telemetry = Vec::new();

        let reason = run_until_stopped(&mut backend, &mut telemetry).unwrap();

        assert_eq!(
            reason,
            RuntimeStopReason::TargetUnitsReached {
                current_units: 1,
                target_units: 1,
            }
        );
        assert_eq!(backend.refresh_count, 1);
        assert_eq!(backend.reload_count, 1);
        assert!(telemetry.contains(&RuntimeTelemetryEvent::Prepared {
            kind: TrainingBackendKind::HeadsUpBlueprintV2,
            unit: TrainingUnit::Iteration,
        }));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::StrategyRefreshRequested));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::TelemetryRefreshed));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::ConfigReloadRequested));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::BatchCompleted {
            outcome: BatchOutcome::new(1, true),
            total_units: 1,
        }));
        assert!(telemetry.contains(&RuntimeTelemetryEvent::Stopped {
            reason: RuntimeStopReason::TargetUnitsReached {
                current_units: 1,
                target_units: 1,
            },
        }));
    }
}
