use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use dashmap::DashMap;

/// Per-flop convergence state, written from the on_progress callback.
#[derive(Debug)]
pub struct FlopTuiState {
    pub exploitability_history: Vec<f64>,
    pub iteration: usize,
    pub max_iterations: usize,
    pub pct_actions_pruned: f64,
    pub max_positive_regret: f64,
    pub min_negative_regret: f64,
}

/// Shared metrics between solver threads and the TUI renderer.
///
/// Solver threads increment atomic counters in the hot path.
/// The TUI thread samples them at its refresh interval.
#[derive(Debug)]
pub struct TuiMetrics {
    // SPR-level progress (set by outer loop)
    pub current_spr: AtomicU32,
    pub total_sprs: AtomicU32,
    pub flops_completed: AtomicU32,
    pub total_flops: AtomicU32,

    // Current SPR phase: 0=idle, 1=BuildingTree, 2=SolvingFlops, 3=ComputingValues,
    // 4=ComputingEquityTables (pre-SPR)
    pub spr_phase: AtomicU32,
    /// Milliseconds since Unix epoch when the current phase started.
    pub phase_start_ms: AtomicU64,

    /// Equity table pre-computation progress (pre-SPR phase).
    pub equity_tables_completed: AtomicU32,
    pub equity_tables_total: AtomicU32,

    // Per-flop exploitability (written from on_progress callback)
    pub flop_states: DashMap<String, FlopTuiState>,
}

impl TuiMetrics {
    pub fn new(total_sprs: u32, total_flops: u32) -> Self {
        Self {
            current_spr: AtomicU32::new(0),
            total_sprs: AtomicU32::new(total_sprs),
            flops_completed: AtomicU32::new(0),
            total_flops: AtomicU32::new(total_flops),
            spr_phase: AtomicU32::new(0),
            phase_start_ms: AtomicU64::new(0),
            equity_tables_completed: AtomicU32::new(0),
            equity_tables_total: AtomicU32::new(0),
            flop_states: DashMap::new(),
        }
    }

    /// Set the current SPR phase and record when it started.
    pub fn set_phase(&self, phase: u32) {
        self.spr_phase.store(phase, Ordering::Relaxed);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.phase_start_ms.store(now_ms, Ordering::Relaxed);
    }

    /// Get seconds elapsed since the current phase started.
    pub fn phase_elapsed_secs(&self) -> f64 {
        let start_ms = self.phase_start_ms.load(Ordering::Relaxed);
        if start_ms == 0 {
            return 0.0;
        }
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        now_ms.saturating_sub(start_ms) as f64 / 1000.0
    }

    /// Reset per-SPR counters when starting a new SPR solve.
    pub fn start_spr(&self, spr_index: u32, total_flops: u32) {
        self.current_spr.store(spr_index, Ordering::Relaxed);
        self.flops_completed.store(0, Ordering::Relaxed);
        self.total_flops.store(total_flops, Ordering::Relaxed);
        self.spr_phase.store(0, Ordering::Relaxed);
        self.phase_start_ms.store(0, Ordering::Relaxed);
        self.flop_states.clear();
    }

    /// Update per-flop exploitability state (called from on_progress).
    pub fn update_flop(
        &self,
        flop_name: &str,
        iteration: usize,
        max_iterations: usize,
        exploitability: f64,
        pct_actions_pruned: f64,
        max_positive_regret: f64,
        min_negative_regret: f64,
    ) {
        let mut entry = self
            .flop_states
            .entry(flop_name.to_string())
            .or_insert_with(|| FlopTuiState {
                exploitability_history: Vec::new(),
                iteration: 0,
                max_iterations,
                pct_actions_pruned: 0.0,
                max_positive_regret: 0.0,
                min_negative_regret: 0.0,
            });
        entry.iteration = iteration;
        entry.max_iterations = max_iterations;
        entry.exploitability_history.push(exploitability);
        entry.pct_actions_pruned = pct_actions_pruned;
        entry.max_positive_regret = max_positive_regret;
        entry.min_negative_regret = min_negative_regret;
    }

    /// Remove a flop entry when it finishes (called from on_progress).
    pub fn remove_flop(&self, flop_name: &str) {
        self.flop_states.remove(flop_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    #[test]
    fn tui_metrics_init() {
        let m = TuiMetrics::new(7, 200);
        assert_eq!(m.total_sprs.load(Ordering::Relaxed), 7);
        assert_eq!(m.total_flops.load(Ordering::Relaxed), 200);
    }

    #[test]
    fn flop_state_lifecycle() {
        let m = TuiMetrics::new(1, 10);
        m.update_flop("AhKs2d", 0, 50, 120.5, 0.0, 0.0, 0.0);
        m.update_flop("AhKs2d", 1, 50, 95.3, 12.5, 0.0, 0.0);
        {
            let entry = m.flop_states.get("AhKs2d").unwrap();
            assert_eq!(entry.iteration, 1);
            assert_eq!(entry.exploitability_history.len(), 2);
        }
        m.remove_flop("AhKs2d");
        assert!(m.flop_states.get("AhKs2d").is_none());
    }

    #[test]
    fn start_spr_resets_counters() {
        let m = TuiMetrics::new(3, 100);
        m.flops_completed.store(50, Ordering::Relaxed);
        m.update_flop("test", 10, 50, 42.0, 0.0, 0.0, 0.0);

        m.start_spr(1, 200);

        assert_eq!(m.flops_completed.load(Ordering::Relaxed), 0);
        assert_eq!(m.total_flops.load(Ordering::Relaxed), 200);
        assert_eq!(m.current_spr.load(Ordering::Relaxed), 1);
        assert!(m.flop_states.is_empty());
    }
}
