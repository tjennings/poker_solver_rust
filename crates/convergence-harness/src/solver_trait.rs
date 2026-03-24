use std::collections::BTreeMap;

/// Per-node strategy: maps node_id to flat Vec<f32> (action_idx * num_hands + hand_idx).
pub type StrategyMap = BTreeMap<u64, Vec<f32>>;

/// Per-node, per-player combo EVs: maps node_id to [oop_evs, ip_evs].
pub type ComboEvMap = BTreeMap<u64, [Vec<f32>; 2]>;

/// Algorithm-specific metrics reported by the solver.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SolverMetrics {
    /// Key-value pairs of algorithm-specific metrics (e.g., "avg_regret" -> 0.05)
    pub values: BTreeMap<String, f64>,
}

/// Trait for pluggable CFR solver algorithms.
pub trait ConvergenceSolver {
    /// Human-readable name (e.g., "Exhaustive DCFR", "MCCFR 500bkt")
    fn name(&self) -> &str;

    /// Run one iteration (or batch). The harness calls this in a loop.
    fn solve_step(&mut self);

    /// Current iteration count.
    fn iterations(&self) -> u64;

    /// Extract the current average strategy at every decision node.
    /// Keys are node IDs (unique per decision point), values are flat strategy vectors.
    fn average_strategy(&self) -> StrategyMap;

    /// Extract per-combo EVs at every decision node for both players.
    fn combo_evs(&self) -> ComboEvMap;

    /// Algorithm-specific metrics (avg regret, strategy delta, etc.)
    fn self_reported_metrics(&self) -> SolverMetrics;
}
