use std::path::Path;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use poker_solver_core::blueprint_mp::trainer::{
    MpTrainingEvent, MpTrainingEventSink, format_mp_training_event_compact,
};
use poker_solver_core::blueprint_v2::baseline_validation::BaselineValidationReport;
use poker_solver_core::blueprint_v2::exploitable_spots::ExploitableSpot;

use crate::blueprint_tui::ResolvedScenario;
use crate::blueprint_tui_audit::AuditSnapshot;
use crate::blueprint_tui_audit_widget::AuditPanelState;
use crate::blueprint_tui_widgets::CellStrategy;

/// Maximum number of monitored scenarios.
pub const MAX_SCENARIOS: usize = 16;
const SNAPSHOT_STATUS_DETAIL_CHARS: usize = 48;
const TRAINING_EVENT_TTL: Duration = Duration::from_secs(60);

#[derive(Debug, Clone)]
struct PublishedTrainingStatus {
    text: String,
    published_at: Instant,
    durable: bool,
}

/// One-line status for manual TUI snapshot requests.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotStatus {
    Idle,
    Queued,
    Writing,
    Saved(String),
    Failed(String),
}

/// State for the random-scenario carousel display.
#[derive(Debug, Clone)]
pub struct RandomScenarioState {
    pub name: String,
    pub node_idx: u32,
    pub grid: [[CellStrategy; 13]; 13],
    pub board_display: String,
    pub street_label: String,
}

/// State pushed from the trainer to the TUI after a config reload.
pub struct ReloadedTuiState {
    pub scenarios: Vec<ResolvedScenario>,
    pub audit_panel: Option<AuditPanelState>,
}

/// Shared state bridging the Blueprint V2 training thread and the TUI render
/// thread.
///
/// Hot-path counters use lock-free atomics. Infrequent bulk data (strategy
/// snapshots, grids, delta histories) sit behind `Mutex` locks that the TUI
/// samples at its refresh interval.
pub struct BlueprintTuiMetrics {
    // --- lock-free atomics (hot path) ---
    pub iterations: Arc<AtomicU64>,
    pub target_iterations: Option<u64>,
    pub time_limit_minutes: Option<u64>,
    pub start_time: Instant,

    // --- control flags (shared with training thread) ---
    pub paused: Arc<AtomicBool>,
    pub quit_requested: Arc<AtomicBool>,
    pub snapshot_trigger: Arc<AtomicBool>,
    pub strategy_refresh_trigger: Arc<AtomicBool>,
    pub config_reload_trigger: Arc<AtomicBool>,
    pub snapshot_status: Mutex<SnapshotStatus>,
    latest_training_status: Mutex<Option<PublishedTrainingStatus>>,

    /// Reloaded TUI state pushed by the trainer after a config reload.
    pub reloaded_tui_state: Mutex<Option<ReloadedTuiState>>,

    // --- infrequent bulk data (behind Mutex) ---
    pub strategy_snapshots: Mutex<Vec<Vec<f64>>>,
    pub prev_strategy_snapshots: Mutex<Vec<Vec<f64>>>,
    pub strategy_deltas: Mutex<Vec<f64>>,
    pub random_scenario: Mutex<Option<RandomScenarioState>>,

    // --- strategy grids for TUI scenario refresh ---
    pub strategy_grids: Mutex<Vec<Option<[[CellStrategy; 13]; 13]>>>,

    /// Latest regret audit snapshots, ready for the TUI to consume.
    pub regret_audit_snapshots: Mutex<Option<Vec<AuditSnapshot>>>,
    /// Compact raw-strategy probe lines rendered in the MP TUI metrics panel.
    pub strategy_probe_lines: Mutex<Vec<String>>,

    // --- sparkline history ---
    pub strategy_delta_history: Mutex<Vec<f64>>,
    pub leaf_movement_history: Mutex<Vec<f64>>,
    pub min_regret_history: Mutex<Vec<f64>>,
    pub max_regret_history: Mutex<Vec<f64>>,
    pub avg_pos_regret_history: Mutex<Vec<f64>>,
    pub prune_history: Mutex<Vec<f64>>,
    pub exploitability_history: Mutex<Vec<f64>>,

    // --- exploitability pass progress ---
    pub exploitability_progress: Arc<AtomicU64>,
    pub exploitability_total: Arc<AtomicU64>,
    pub exploitability_start_time: Mutex<Option<Instant>>,

    // --- top exploitable spots from BR pass ---
    pub exploitable_spots: Mutex<Vec<ExploitableSpot>>,
    /// Latest external baseline strategy-frequency validation report.
    pub baseline_validation_report: Mutex<Option<BaselineValidationReport>>,
}

impl BlueprintTuiMetrics {
    pub fn new(target_iterations: Option<u64>, time_limit_minutes: Option<u64>) -> Self {
        let mut snapshots = Vec::with_capacity(MAX_SCENARIOS);
        let mut prev_snapshots = Vec::with_capacity(MAX_SCENARIOS);
        let mut deltas = Vec::with_capacity(MAX_SCENARIOS);
        let mut grids = Vec::with_capacity(MAX_SCENARIOS);
        for _ in 0..MAX_SCENARIOS {
            snapshots.push(Vec::new());
            prev_snapshots.push(Vec::new());
            deltas.push(0.0);
            grids.push(None);
        }

        Self {
            iterations: Arc::new(AtomicU64::new(0)),
            target_iterations,
            time_limit_minutes,
            start_time: Instant::now(),

            paused: Arc::new(AtomicBool::new(false)),
            quit_requested: Arc::new(AtomicBool::new(false)),
            snapshot_trigger: Arc::new(AtomicBool::new(false)),
            strategy_refresh_trigger: Arc::new(AtomicBool::new(false)),
            config_reload_trigger: Arc::new(AtomicBool::new(false)),
            snapshot_status: Mutex::new(SnapshotStatus::Idle),
            latest_training_status: Mutex::new(None),
            reloaded_tui_state: Mutex::new(None),

            strategy_snapshots: Mutex::new(snapshots),
            prev_strategy_snapshots: Mutex::new(prev_snapshots),
            strategy_deltas: Mutex::new(deltas),
            random_scenario: Mutex::new(None),

            strategy_grids: Mutex::new(grids),
            regret_audit_snapshots: Mutex::new(None),
            strategy_probe_lines: Mutex::new(Vec::new()),
            strategy_delta_history: Mutex::new(Vec::new()),
            leaf_movement_history: Mutex::new(Vec::new()),
            min_regret_history: Mutex::new(Vec::new()),
            max_regret_history: Mutex::new(Vec::new()),
            avg_pos_regret_history: Mutex::new(Vec::new()),
            prune_history: Mutex::new(Vec::new()),
            exploitability_history: Mutex::new(Vec::new()),

            exploitability_progress: Arc::new(AtomicU64::new(0)),
            exploitability_total: Arc::new(AtomicU64::new(0)),
            exploitability_start_time: Mutex::new(None),

            exploitable_spots: Mutex::new(Vec::new()),
            baseline_validation_report: Mutex::new(None),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn toggle_pause(&self) {
        self.paused.fetch_xor(true, Ordering::Relaxed);
    }

    pub fn request_snapshot(&self) {
        self.set_snapshot_status(SnapshotStatus::Queued);
        self.snapshot_trigger.store(true, Ordering::Release);
    }

    pub fn mark_snapshot_writing(&self) {
        self.set_snapshot_status(SnapshotStatus::Writing);
    }

    pub fn mark_snapshot_saved(&self, path: &Path) {
        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .map_or_else(|| path.display().to_string(), ToOwned::to_owned);
        self.set_snapshot_status(SnapshotStatus::Saved(truncate_status_detail(&name)));
    }

    pub fn mark_snapshot_failed(&self, error: impl std::fmt::Display) {
        self.set_snapshot_status(SnapshotStatus::Failed(truncate_status_detail(
            &error.to_string(),
        )));
    }

    pub fn snapshot_status_text(&self) -> Option<String> {
        let status = self
            .snapshot_status
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        match &*status {
            SnapshotStatus::Idle => None,
            SnapshotStatus::Queued => Some("snapshot: queued".to_string()),
            SnapshotStatus::Writing => Some("snapshot: writing".to_string()),
            SnapshotStatus::Saved(path) => Some(format!("snapshot: saved {path}")),
            SnapshotStatus::Failed(error) => Some(format!("snapshot: failed {error}")),
        }
    }

    fn set_snapshot_status(&self, status: SnapshotStatus) {
        let mut current = self
            .snapshot_status
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *current = status;
    }

    /// Publish an MP training event at an injected monotonic time.
    ///
    /// This seam keeps replacement and expiry behavior deterministic in tests.
    pub fn publish_training_event_at(&self, event: MpTrainingEvent, now: Instant) {
        let status = PublishedTrainingStatus {
            text: format_mp_training_event_compact(&event),
            published_at: now,
            durable: event.is_durable(),
        };
        let mut latest = self
            .latest_training_status
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        *latest = Some(status);
    }

    /// Latest non-expired MP training status at an injected monotonic time.
    #[must_use]
    pub fn training_status_text_at(&self, now: Instant) -> Option<String> {
        let latest = self
            .latest_training_status
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        latest.as_ref().and_then(|status| {
            (status.durable
                || now.saturating_duration_since(status.published_at) < TRAINING_EVENT_TTL)
                .then(|| status.text.clone())
        })
    }

    /// Latest non-expired MP training status for rendering.
    #[must_use]
    pub fn training_status_text(&self) -> Option<String> {
        self.training_status_text_at(Instant::now())
    }

    pub fn request_strategy_refresh(&self) {
        self.strategy_refresh_trigger.store(true, Ordering::Relaxed);
    }

    pub fn request_config_reload(&self) {
        self.config_reload_trigger.store(true, Ordering::Relaxed);
    }

    /// Consume the snapshot trigger, returning `true` if it was set.
    #[allow(dead_code)]
    pub fn take_snapshot_trigger(&self) -> bool {
        self.snapshot_trigger.swap(false, Ordering::Acquire)
    }

    /// Update the strategy snapshot for a scenario, computing the L1 delta
    /// against the previous snapshot before overwriting.
    #[allow(dead_code)]
    pub fn update_scenario_strategy(&self, scenario_idx: usize, probs: Vec<f64>) {
        if scenario_idx >= MAX_SCENARIOS {
            return;
        }

        // INVARIANT: Both vecs are pre-allocated to MAX_SCENARIOS in `new`.
        let mut snaps = self.strategy_snapshots.lock().unwrap();
        let mut prev = self.prev_strategy_snapshots.lock().unwrap();
        let mut deltas = self.strategy_deltas.lock().unwrap();

        // Compute L1 delta between new probs and current snapshot.
        let old = &snaps[scenario_idx];
        let delta = if old.len() == probs.len() {
            old.iter()
                .zip(probs.iter())
                .map(|(a, b)| (a - b).abs())
                .sum()
        } else {
            0.0
        };
        deltas[scenario_idx] = delta;

        // Rotate current -> previous, install new.
        prev[scenario_idx] = std::mem::take(&mut snaps[scenario_idx]);
        snaps[scenario_idx] = probs;
    }

    /// Store a refreshed strategy grid for a scenario.
    pub fn update_scenario_grid(&self, idx: usize, grid: [[CellStrategy; 13]; 13]) {
        let mut grids = self
            .strategy_grids
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if idx < grids.len() {
            grids[idx] = Some(grid);
        }
    }

    /// Store updated audit snapshots for the TUI to pick up.
    pub fn update_regret_audits(&self, snapshots: Vec<AuditSnapshot>) {
        let mut data = self
            .regret_audit_snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *data = Some(snapshots);
    }

    /// Replace the compact raw-strategy probe lines shown by the MP TUI.
    pub fn update_strategy_probe_lines(&self, lines: Vec<String>) {
        let mut data = self
            .strategy_probe_lines
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *data = lines;
    }

    /// Take the latest audit snapshots (returns None if none pending).
    pub fn take_regret_audits(&self) -> Option<Vec<AuditSnapshot>> {
        let mut data = self
            .regret_audit_snapshots
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        data.take()
    }

    /// Push a strategy delta sample into the sparkline history.
    pub fn push_strategy_delta(&self, delta: f64) {
        let mut hist = self
            .strategy_delta_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        hist.push(delta);
    }

    /// Push a prune fraction sample into the sparkline history.
    pub fn push_prune_fraction(&self, frac: f64) {
        let mut hist = self.prune_history.lock().unwrap_or_else(|e| e.into_inner());
        hist.push(frac);
    }

    /// Push a min-regret sample into the sparkline history.
    pub fn push_min_regret(&self, val: f64) {
        let mut hist = self
            .min_regret_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        hist.push(val);
    }

    /// Push a max-regret sample into the sparkline history.
    pub fn push_max_regret(&self, val: f64) {
        let mut hist = self
            .max_regret_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        hist.push(val);
    }

    /// Push an average positive regret sample into the sparkline history.
    pub fn push_avg_pos_regret(&self, val: f64) {
        let mut hist = self
            .avg_pos_regret_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        hist.push(val);
    }

    /// Push a leaf movement sample into the sparkline history.
    pub fn push_leaf_movement(&self, pct: f64) {
        let mut hist = self
            .leaf_movement_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        hist.push(pct);
    }

    /// Push an exploitability sample into the sparkline history.
    pub fn push_exploitability(&self, val: f64) {
        let mut hist = self
            .exploitability_history
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        hist.push(val);
    }

    /// Push a new random scenario to be picked up by the TUI on its next tick.
    pub fn update_random_scenario(
        &self,
        name: String,
        node_idx: u32,
        grid: [[CellStrategy; 13]; 13],
        board_display: String,
        street_label: String,
    ) {
        let mut rs = self
            .random_scenario
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *rs = Some(RandomScenarioState {
            name,
            node_idx,
            grid,
            board_display,
            street_label,
        });
    }

    /// Seconds elapsed since this metrics instance was created.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Replace the stored exploitable spots with a new set from the latest BR pass.
    pub fn set_exploitable_spots(&self, spots: Vec<ExploitableSpot>) {
        *self
            .exploitable_spots
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = spots;
    }

    /// Replace the latest external baseline validation report.
    pub fn set_baseline_validation_report(&self, report: BaselineValidationReport) {
        *self
            .baseline_validation_report
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(report);
    }

    /// Signal the start of an exploitability/BR pass with the given total sample count.
    pub fn start_exploitability_pass(&self, total: u64) {
        self.exploitability_progress.store(0, Ordering::Relaxed);
        self.exploitability_total.store(total, Ordering::Relaxed);
        let mut start = self
            .exploitability_start_time
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *start = Some(Instant::now());
    }

    /// Increment the exploitability progress counter by one deal.
    pub fn tick_exploitability_progress(&self) {
        self.exploitability_progress.fetch_add(1, Ordering::Relaxed);
    }

    /// Signal the end of an exploitability/BR pass, resetting progress state.
    pub fn finish_exploitability_pass(&self) {
        self.exploitability_total.store(0, Ordering::Relaxed);
        self.exploitability_progress.store(0, Ordering::Relaxed);
        let mut start = self
            .exploitability_start_time
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *start = None;
    }
}

impl MpTrainingEventSink for BlueprintTuiMetrics {
    fn publish(&self, event: MpTrainingEvent) {
        self.publish_training_event_at(event, Instant::now());
    }
}

fn truncate_status_detail(detail: &str) -> String {
    let mut chars = detail.chars();
    let mut truncated: String = chars.by_ref().take(SNAPSHOT_STATUS_DETAIL_CHARS).collect();
    if chars.next().is_some() {
        truncated.push_str("...");
    }
    truncated
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_solver_core::blueprint_mp::trainer::{
        MpDcfrPassEvent, MpDcfrScheduleEvent, MpDcfrScheduleMode,
    };
    use std::num::NonZeroU64;
    use std::sync::atomic::Ordering;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;
    use test_macros::timed_test;

    fn pass_event(pass: u64, max_reached: bool) -> MpTrainingEvent {
        MpTrainingEvent::DcfrPass(MpDcfrPassEvent {
            completed_passes: pass,
            max_passes: NonZeroU64::new(40),
            epoch: pass,
            meta_iteration: pass * 100,
            skipped_slots: 0,
            sweep_duration: Duration::from_millis(25),
            purge_duration: None,
            max_reached,
        })
    }

    #[timed_test(10)]
    fn initial_state() {
        let m = BlueprintTuiMetrics::new(Some(1000), None);
        assert_eq!(m.iterations.load(Ordering::Relaxed), 0);
        assert!(!m.paused.load(Ordering::Relaxed));
        assert!(!m.quit_requested.load(Ordering::Relaxed));
        assert!(!m.take_snapshot_trigger());
        assert_eq!(m.snapshot_status_text(), None);
        assert_eq!(m.target_iterations, Some(1000));
        assert_eq!(m.training_status_text(), None);
    }

    #[timed_test(10)]
    fn ordinary_training_status_expires_after_sixty_seconds() {
        let metrics = BlueprintTuiMetrics::new(None, None);
        let published_at = Instant::now();
        metrics.publish_training_event_at(pass_event(1, false), published_at);

        assert!(
            metrics
                .training_status_text_at(published_at + Duration::from_secs(59))
                .is_some()
        );
        assert_eq!(
            metrics.training_status_text_at(published_at + Duration::from_secs(60)),
            None
        );
    }

    #[timed_test(10)]
    fn latest_training_event_replaces_previous_status() {
        let metrics = BlueprintTuiMetrics::new(None, None);
        let now = Instant::now();
        metrics.publish_training_event_at(
            MpTrainingEvent::DcfrSchedule(MpDcfrScheduleEvent {
                mode: MpDcfrScheduleMode::WallClock {
                    interval_seconds: NonZeroU64::new(600).unwrap(),
                },
                warmup_meta_iterations: 3_000_000,
                max_passes: NonZeroU64::new(40),
            }),
            now,
        );
        metrics.publish_training_event_at(pass_event(2, false), now + Duration::from_secs(1));

        assert_eq!(
            metrics
                .training_status_text_at(now + Duration::from_secs(2))
                .as_deref(),
            Some("DCFR pass 2/40 | epoch 2 | sweep 25.0ms")
        );
    }

    #[timed_test(10)]
    fn max_reached_training_status_is_durable() {
        let metrics = BlueprintTuiMetrics::new(None, None);
        let now = Instant::now();
        metrics.publish_training_event_at(pass_event(40, true), now);

        assert_eq!(
            metrics
                .training_status_text_at(now + Duration::from_secs(86_400))
                .as_deref(),
            Some("DCFR pass 40/40 | epoch 40 | sweep 25.0ms | cap reached; stopped")
        );
    }

    #[timed_test(10)]
    fn strategy_snapshot_lifecycle() {
        let m = BlueprintTuiMetrics::new(None, None);

        // First update -- no previous, delta should be 0.
        m.update_scenario_strategy(0, vec![0.5, 0.3, 0.2]);
        {
            let snaps = m.strategy_snapshots.lock().unwrap();
            assert_eq!(snaps[0], vec![0.5, 0.3, 0.2]);
        }

        // Second update -- delta computed against previous.
        m.update_scenario_strategy(0, vec![0.4, 0.4, 0.2]);
        {
            let snaps = m.strategy_snapshots.lock().unwrap();
            assert_eq!(snaps[0], vec![0.4, 0.4, 0.2]);

            let prev = m.prev_strategy_snapshots.lock().unwrap();
            assert_eq!(prev[0], vec![0.5, 0.3, 0.2]);

            let deltas = m.strategy_deltas.lock().unwrap();
            // |0.5-0.4| + |0.3-0.4| + |0.2-0.2| = 0.1 + 0.1 + 0.0 = 0.2
            assert!((deltas[0] - 0.2).abs() < 1e-9);
        }
    }

    #[timed_test(10)]
    fn pause_resume() {
        let m = BlueprintTuiMetrics::new(None, None);
        assert!(!m.is_paused());

        m.toggle_pause();
        assert!(m.is_paused());

        m.toggle_pause();
        assert!(!m.is_paused());
    }

    #[timed_test(10)]
    fn trigger_flags() {
        let m = BlueprintTuiMetrics::new(None, None);

        // Trigger not set initially.
        assert!(!m.take_snapshot_trigger());

        // Set and consume.
        m.request_snapshot();
        assert!(m.take_snapshot_trigger());
        assert_eq!(
            m.snapshot_status_text(),
            Some("snapshot: queued".to_string())
        );

        // Consumed -- second take returns false.
        assert!(!m.take_snapshot_trigger());
    }

    #[timed_test(10)]
    fn snapshot_status_lifecycle() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.request_snapshot();
        assert_eq!(
            m.snapshot_status_text().as_deref(),
            Some("snapshot: queued")
        );

        m.mark_snapshot_writing();
        assert_eq!(
            m.snapshot_status_text().as_deref(),
            Some("snapshot: writing")
        );

        m.mark_snapshot_saved(Path::new("/tmp/snapshot_0007"));
        assert_eq!(
            m.snapshot_status_text().as_deref(),
            Some("snapshot: saved snapshot_0007")
        );

        m.mark_snapshot_failed("permission denied");
        assert_eq!(
            m.snapshot_status_text().as_deref(),
            Some("snapshot: failed permission denied")
        );
    }

    #[timed_test(10)]
    fn request_snapshot_sets_status_before_publishing_trigger() {
        let m = Arc::new(BlueprintTuiMetrics::new(None, None));
        let status_guard = m.snapshot_status.lock().unwrap();
        let (started_tx, started_rx) = mpsc::channel();
        let worker_metrics = Arc::clone(&m);

        let worker = thread::spawn(move || {
            started_tx.send(()).unwrap();
            worker_metrics.request_snapshot();
        });
        started_rx.recv().unwrap();

        let deadline = Instant::now() + Duration::from_millis(50);
        while Instant::now() < deadline {
            assert!(
                !m.snapshot_trigger.load(Ordering::Acquire),
                "snapshot trigger was published before queued status was stored"
            );
            thread::yield_now();
        }

        drop(status_guard);
        worker.join().unwrap();
        assert!(m.take_snapshot_trigger());
        assert_eq!(
            m.snapshot_status_text().as_deref(),
            Some("snapshot: queued")
        );
    }

    #[timed_test(10)]
    fn strategy_delta_history() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.push_strategy_delta(0.1);
        m.push_strategy_delta(0.05);
        let hist = m.strategy_delta_history.lock().unwrap();
        assert_eq!(hist.len(), 2);
        assert!((hist[0] - 0.1).abs() < 1e-9);
        assert!((hist[1] - 0.05).abs() < 1e-9);
    }

    #[timed_test(10)]
    fn leaf_movement_history() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.push_leaf_movement(0.8);
        m.push_leaf_movement(0.3);
        let hist = m.leaf_movement_history.lock().unwrap();
        assert_eq!(hist.len(), 2);
        assert!((hist[0] - 0.8).abs() < 1e-9);
    }

    #[timed_test(10)]
    fn scenario_grid_update() {
        let m = BlueprintTuiMetrics::new(None, None);
        let grid: [[CellStrategy; 13]; 13] =
            std::array::from_fn(|_| std::array::from_fn(|_| CellStrategy::default()));
        m.update_scenario_grid(0, grid);
        let grids = m.strategy_grids.lock().unwrap();
        assert!(grids[0].is_some());
    }

    #[timed_test(10)]
    fn strategy_probe_lines_update() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.update_strategy_probe_lines(vec!["UTG A5s:P:B97".into()]);
        let lines = m.strategy_probe_lines.lock().unwrap();
        assert_eq!(lines.as_slice(), ["UTG A5s:P:B97"]);
    }

    #[timed_test(10)]
    fn config_reload_trigger_initial_state() {
        let m = BlueprintTuiMetrics::new(None, None);
        // Not set initially.
        assert!(!m.config_reload_trigger.load(Ordering::Relaxed));
    }

    #[timed_test(10)]
    fn config_reload_request_and_consume() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.request_config_reload();
        assert!(m.config_reload_trigger.load(Ordering::Relaxed));
        // Consuming via swap.
        assert!(m.config_reload_trigger.swap(false, Ordering::Relaxed));
        // Second read is false.
        assert!(!m.config_reload_trigger.load(Ordering::Relaxed));
    }

    #[timed_test(10)]
    fn reloaded_tui_state_initially_none() {
        let m = BlueprintTuiMetrics::new(None, None);
        let state = m.reloaded_tui_state.lock().unwrap();
        assert!(state.is_none());
    }

    #[timed_test(10)]
    fn reloaded_tui_state_push_and_take() {
        let m = BlueprintTuiMetrics::new(None, None);
        let state = ReloadedTuiState {
            scenarios: vec![],
            audit_panel: None,
        };
        *m.reloaded_tui_state.lock().unwrap() = Some(state);
        let taken = m.reloaded_tui_state.lock().unwrap().take();
        assert!(taken.is_some());
        // Second take is None.
        assert!(m.reloaded_tui_state.lock().unwrap().is_none());
    }

    #[timed_test(10)]
    fn regret_audit_snapshot_exchange() {
        let m = BlueprintTuiMetrics::new(None, None);
        let snapshot = vec![crate::blueprint_tui_audit::AuditSnapshot {
            regrets: vec![1.0, -2.0, 3.0],
            deltas: vec![0.5, -0.1, 0.2],
            trends: vec![
                crate::blueprint_tui_audit::Trend::Up,
                crate::blueprint_tui_audit::Trend::Down,
                crate::blueprint_tui_audit::Trend::Flat,
            ],
            strategy: vec![0.0, 0.25, 0.75],
            avg_strategy: vec![0.0, 0.20, 0.80],
        }];
        m.update_regret_audits(snapshot.clone());
        let taken = m.take_regret_audits();
        assert!(taken.is_some());
        let taken = taken.unwrap();
        assert_eq!(taken.len(), 1);
        assert_eq!(taken[0].regrets, vec![1.0, -2.0, 3.0]);
        // Second take should return None.
        assert!(m.take_regret_audits().is_none());
    }

    #[timed_test(10)]
    fn push_exploitability_stores_values() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.push_exploitability(42.5);
        m.push_exploitability(38.0);
        let hist = m.exploitability_history.lock().unwrap();
        assert_eq!(hist.len(), 2);
        assert!((hist[0] - 42.5).abs() < 1e-9);
        assert!((hist[1] - 38.0).abs() < 1e-9);
    }

    #[timed_test(10)]
    fn exploitability_progress_initial_state() {
        let m = BlueprintTuiMetrics::new(None, None);
        assert_eq!(m.exploitability_progress.load(Ordering::Relaxed), 0);
        assert_eq!(m.exploitability_total.load(Ordering::Relaxed), 0);
        let start = m.exploitability_start_time.lock().unwrap();
        assert!(start.is_none());
    }

    #[timed_test(10)]
    fn start_exploitability_pass_sets_total_and_resets_progress() {
        let m = BlueprintTuiMetrics::new(None, None);
        // Pre-set progress to simulate leftover from previous run.
        m.exploitability_progress.store(50, Ordering::Relaxed);

        m.start_exploitability_pass(1000);

        assert_eq!(m.exploitability_total.load(Ordering::Relaxed), 1000);
        assert_eq!(m.exploitability_progress.load(Ordering::Relaxed), 0);
        let start = m.exploitability_start_time.lock().unwrap();
        assert!(start.is_some());
    }

    #[timed_test(10)]
    fn tick_exploitability_progress_increments() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.start_exploitability_pass(100);

        m.tick_exploitability_progress();
        assert_eq!(m.exploitability_progress.load(Ordering::Relaxed), 1);

        m.tick_exploitability_progress();
        m.tick_exploitability_progress();
        assert_eq!(m.exploitability_progress.load(Ordering::Relaxed), 3);
    }

    #[timed_test(10)]
    fn finish_exploitability_pass_resets_state() {
        let m = BlueprintTuiMetrics::new(None, None);
        m.start_exploitability_pass(100);
        m.tick_exploitability_progress();

        m.finish_exploitability_pass();

        assert_eq!(m.exploitability_total.load(Ordering::Relaxed), 0);
        assert_eq!(m.exploitability_progress.load(Ordering::Relaxed), 0);
        let start = m.exploitability_start_time.lock().unwrap();
        assert!(start.is_none());
    }

    #[timed_test(10)]
    fn exploitability_progress_full_lifecycle() {
        let m = BlueprintTuiMetrics::new(None, None);

        // Start a pass.
        m.start_exploitability_pass(50);
        assert_eq!(m.exploitability_total.load(Ordering::Relaxed), 50);

        // Tick halfway.
        for _ in 0..25 {
            m.tick_exploitability_progress();
        }
        assert_eq!(m.exploitability_progress.load(Ordering::Relaxed), 25);

        // Finish.
        m.finish_exploitability_pass();
        assert_eq!(m.exploitability_total.load(Ordering::Relaxed), 0);
        assert_eq!(m.exploitability_progress.load(Ordering::Relaxed), 0);
    }

    #[timed_test(10)]
    fn exploitable_spots_initially_empty() {
        let m = BlueprintTuiMetrics::new(None, None);
        let spots = m.exploitable_spots.lock().unwrap();
        assert!(spots.is_empty());
    }

    #[timed_test(10)]
    fn baseline_validation_report_initially_empty_and_settable() {
        let m = BlueprintTuiMetrics::new(None, None);
        assert!(m.baseline_validation_report.lock().unwrap().is_none());

        let mut report = BaselineValidationReport::default();
        report.aggregate.spots_total = 6;
        report.aggregate.spots_scored = 6;
        report.aggregate.mean_total_variation = 0.125;
        m.set_baseline_validation_report(report);

        let stored = m.baseline_validation_report.lock().unwrap();
        let stored = stored.as_ref().expect("report should be stored");
        assert_eq!(stored.aggregate.spots_total, 6);
        assert_eq!(stored.aggregate.spots_scored, 6);
        assert!((stored.aggregate.mean_total_variation - 0.125).abs() < f64::EPSILON);
    }

    #[timed_test(10)]
    fn set_exploitable_spots_stores_spots() {
        use poker_solver_core::blueprint_v2::Street;
        use poker_solver_core::blueprint_v2::exploitable_spots::ExploitableSpot;

        let m = BlueprintTuiMetrics::new(None, None);
        let spots = vec![
            ExploitableSpot {
                spot: "sb:5bb".to_string(),
                street: Street::Preflop,
                player: 0,
                bucket: 3,
                br_action: "fold".to_string(),
                magnitude: 42.5,
            },
            ExploitableSpot {
                spot: "sb:5bb,bb:call".to_string(),
                street: Street::Preflop,
                player: 1,
                bucket: 7,
                br_action: "5bb".to_string(),
                magnitude: 10.0,
            },
        ];
        m.set_exploitable_spots(spots);
        let stored = m.exploitable_spots.lock().unwrap();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].spot, "sb:5bb");
        assert!((stored[0].magnitude - 42.5).abs() < f64::EPSILON);
    }

    #[timed_test(10)]
    fn set_exploitable_spots_replaces_previous() {
        use poker_solver_core::blueprint_v2::Street;
        use poker_solver_core::blueprint_v2::exploitable_spots::ExploitableSpot;

        let m = BlueprintTuiMetrics::new(None, None);
        let spots1 = vec![ExploitableSpot {
            spot: "first".to_string(),
            street: Street::Preflop,
            player: 0,
            bucket: 0,
            br_action: "fold".to_string(),
            magnitude: 1.0,
        }];
        m.set_exploitable_spots(spots1);
        assert_eq!(m.exploitable_spots.lock().unwrap().len(), 1);

        let spots2 = vec![
            ExploitableSpot {
                spot: "second".to_string(),
                street: Street::Flop,
                player: 1,
                bucket: 5,
                br_action: "check".to_string(),
                magnitude: 2.0,
            },
            ExploitableSpot {
                spot: "third".to_string(),
                street: Street::River,
                player: 0,
                bucket: 8,
                br_action: "all-in".to_string(),
                magnitude: 3.0,
            },
        ];
        m.set_exploitable_spots(spots2);
        let stored = m.exploitable_spots.lock().unwrap();
        assert_eq!(stored.len(), 2);
        assert_eq!(stored[0].spot, "second");
        assert_eq!(stored[1].spot, "third");
    }
}
