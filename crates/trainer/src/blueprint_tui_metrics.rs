use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Maximum number of monitored scenarios.
pub const MAX_SCENARIOS: usize = 16;

/// State for the random-scenario carousel display.
#[derive(Debug, Clone)]
pub struct RandomScenarioState {
    pub name: String,
    pub node_idx: u32,
    pub strategy: Vec<f64>,
    pub board_display: Option<String>,
    pub cluster_id: Option<u32>,
    pub action_path: Vec<String>,
    pub started_at: Instant,
}

/// Shared state bridging the Blueprint V2 training thread and the TUI render
/// thread.
///
/// Hot-path counters use lock-free atomics. Infrequent bulk data (strategy
/// snapshots, exploitability history) sit behind `Mutex` locks that the TUI
/// samples at its refresh interval.
#[derive(Debug)]
pub struct BlueprintTuiMetrics {
    // --- lock-free atomics (hot path) ---
    pub iterations: AtomicU64,
    pub target_iterations: Option<u64>,
    pub start_time: Instant,

    // --- control flags ---
    pub paused: AtomicBool,
    pub quit_requested: AtomicBool,
    snapshot_trigger: AtomicBool,
    exploitability_trigger: AtomicBool,

    // --- infrequent bulk data (behind Mutex) ---
    pub strategy_snapshots: Mutex<Vec<Vec<f64>>>,
    pub prev_strategy_snapshots: Mutex<Vec<Vec<f64>>>,
    pub strategy_deltas: Mutex<Vec<f64>>,
    pub exploitability_history: Mutex<VecDeque<(f64, f64)>>,
    pub random_scenario: Mutex<Option<RandomScenarioState>>,
}

impl BlueprintTuiMetrics {
    pub fn new(target_iterations: Option<u64>) -> Self {
        let mut snapshots = Vec::with_capacity(MAX_SCENARIOS);
        let mut prev_snapshots = Vec::with_capacity(MAX_SCENARIOS);
        let mut deltas = Vec::with_capacity(MAX_SCENARIOS);
        for _ in 0..MAX_SCENARIOS {
            snapshots.push(Vec::new());
            prev_snapshots.push(Vec::new());
            deltas.push(0.0);
        }

        Self {
            iterations: AtomicU64::new(0),
            target_iterations,
            start_time: Instant::now(),

            paused: AtomicBool::new(false),
            quit_requested: AtomicBool::new(false),
            snapshot_trigger: AtomicBool::new(false),
            exploitability_trigger: AtomicBool::new(false),

            strategy_snapshots: Mutex::new(snapshots),
            prev_strategy_snapshots: Mutex::new(prev_snapshots),
            strategy_deltas: Mutex::new(deltas),
            exploitability_history: Mutex::new(VecDeque::new()),
            random_scenario: Mutex::new(None),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Relaxed)
    }

    pub fn toggle_pause(&self) {
        self.paused.fetch_xor(true, Ordering::Relaxed);
    }

    pub fn request_snapshot(&self) {
        self.snapshot_trigger.store(true, Ordering::Relaxed);
    }

    /// Consume the snapshot trigger, returning `true` if it was set.
    pub fn take_snapshot_trigger(&self) -> bool {
        self.snapshot_trigger.swap(false, Ordering::Relaxed)
    }

    pub fn request_exploitability(&self) {
        self.exploitability_trigger.store(true, Ordering::Relaxed);
    }

    /// Consume the exploitability trigger, returning `true` if it was set.
    pub fn take_exploitability_trigger(&self) -> bool {
        self.exploitability_trigger.swap(false, Ordering::Relaxed)
    }

    /// Update the strategy snapshot for a scenario, computing the L1 delta
    /// against the previous snapshot before overwriting.
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

        // Rotate current → previous, install new.
        prev[scenario_idx] = std::mem::take(&mut snaps[scenario_idx]);
        snaps[scenario_idx] = probs;
    }

    /// Push an exploitability measurement into the history ring.
    pub fn push_exploitability(&self, mbb: f64) {
        let elapsed = self.elapsed_secs();
        let mut hist = self.exploitability_history.lock().unwrap();
        hist.push_back((elapsed, mbb));
    }

    /// Seconds elapsed since this metrics instance was created.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;
    use test_macros::timed_test;

    #[timed_test(10)]
    fn initial_state() {
        let m = BlueprintTuiMetrics::new(Some(1000));
        assert_eq!(m.iterations.load(Ordering::Relaxed), 0);
        assert!(!m.paused.load(Ordering::Relaxed));
        assert!(!m.quit_requested.load(Ordering::Relaxed));
        assert!(!m.take_snapshot_trigger());
        assert!(!m.take_exploitability_trigger());
        assert_eq!(m.target_iterations, Some(1000));
    }

    #[timed_test(10)]
    fn strategy_snapshot_lifecycle() {
        let m = BlueprintTuiMetrics::new(None);

        // First update — no previous, delta should be 0.
        m.update_scenario_strategy(0, vec![0.5, 0.3, 0.2]);
        {
            let snaps = m.strategy_snapshots.lock().unwrap();
            assert_eq!(snaps[0], vec![0.5, 0.3, 0.2]);
        }

        // Second update — delta computed against previous.
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
    fn exploitability_history_push() {
        let m = BlueprintTuiMetrics::new(None);
        m.push_exploitability(150.0);
        m.push_exploitability(120.0);

        let hist = m.exploitability_history.lock().unwrap();
        assert_eq!(hist.len(), 2);
        assert_eq!(hist[0].1, 150.0);
        assert_eq!(hist[1].1, 120.0);
    }

    #[timed_test(10)]
    fn pause_resume() {
        let m = BlueprintTuiMetrics::new(None);
        assert!(!m.is_paused());

        m.toggle_pause();
        assert!(m.is_paused());

        m.toggle_pause();
        assert!(!m.is_paused());
    }

    #[timed_test(10)]
    fn trigger_flags() {
        let m = BlueprintTuiMetrics::new(None);

        // Trigger not set initially.
        assert!(!m.take_snapshot_trigger());

        // Set and consume.
        m.request_snapshot();
        assert!(m.take_snapshot_trigger());

        // Consumed — second take returns false.
        assert!(!m.take_snapshot_trigger());

        // Same for exploitability.
        m.request_exploitability();
        assert!(m.take_exploitability_trigger());
        assert!(!m.take_exploitability_trigger());
    }
}
