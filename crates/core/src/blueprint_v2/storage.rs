//! Flat-buffer storage for regrets and strategy sums, indexed by
//! (decision node, bucket, action).
//!
//! For each `Decision` node in the tree we store
//! `bucket_count * num_actions` entries contiguously.
//! Element index: `layout[decision_idx].offset + bucket * num_actions + action_idx`.

// Arena indices are u32 and action counts fit in u16. Truncation is safe
// for any practical game tree. Precision loss on len-to-f64 casts is
// acceptable (action counts are tiny).
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};
use std::sync::Arc;

use super::game_tree::{GameNode, GameTree};
use crate::cfr::optimizer::CfrOptimizer;

fn humanize_bytes(bytes: usize) -> String {
    if bytes >= 1_073_741_824 {
        format!("{:.1} GB", bytes as f64 / 1_073_741_824.0)
    } else if bytes >= 1_048_576 {
        format!("{:.1} MB", bytes as f64 / 1_048_576.0)
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// Flat-buffer storage for regrets and strategy sums.
pub struct BlueprintStorage {
    /// Cumulative regrets: one `AtomicI32` per (decision node, bucket, action).
    pub regrets: Vec<AtomicI32>,
    /// Strategy sums: one `AtomicI64` per (decision node, bucket, action).
    pub strategy_sums: Vec<AtomicI64>,
    /// Optional baseline buffer for VR-MCCFR variance reduction.
    /// Same layout as regrets. Stores running-average counterfactual
    /// values scaled by x1000.
    pub(crate) baselines: Option<Vec<AtomicI32>>,
    /// Optional prediction buffer for SAPCFR+. Same layout as regrets.
    /// Stores instantaneous regrets from the previous iteration.
    pub(crate) predictions: Option<Vec<AtomicI32>>,
    /// Optional pluggable optimizer (DCFR, SAPCFR+, etc.).
    pub(crate) optimizer: Option<Arc<dyn CfrOptimizer>>,
    /// Number of buckets per street `[preflop, flop, turn, river]`.
    pub bucket_counts: [u16; 4],
    /// Per-node layout metadata. Non-decision nodes use the `Default`
    /// sentinel (`offset=0`, `num_actions=0`) and must never be queried.
    layout: Vec<NodeLayout>,
}

#[derive(Clone, Copy, Default)]
struct NodeLayout {
    offset: usize,
    num_actions: u16,
    /// 0 = preflop, 1 = flop, 2 = turn, 3 = river.
    street_idx: u8,
}

impl BlueprintStorage {
    /// Build storage for a given tree and per-street bucket counts.
    ///
    /// Pre-allocates zeroed flat buffers sized to cover every
    /// (decision node, bucket, action) triple.
    #[must_use]
    pub fn new(tree: &GameTree, bucket_counts: [u16; 4]) -> Self {
        let mut layout = vec![NodeLayout::default(); tree.nodes.len()];
        let mut total: usize = 0;

        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision {
                street, actions, ..
            } = node
            {
                let street_idx = *street as u8;
                let buckets = bucket_counts[street_idx as usize] as usize;
                let num_actions = actions.len();

                layout[i] = NodeLayout {
                    offset: total,
                    num_actions: num_actions as u16,
                    street_idx,
                };

                total += buckets * num_actions;
            }
        }

        let regret_bytes = total * 4;
        let strategy_bytes = total * 8;
        eprintln!(
            "  Storage: {} slots ({} regret + {} strategy = {} total)",
            total,
            humanize_bytes(regret_bytes),
            humanize_bytes(strategy_bytes),
            humanize_bytes(regret_bytes + strategy_bytes),
        );

        Self {
            regrets: (0..total).map(|_| AtomicI32::new(0)).collect(),
            strategy_sums: (0..total).map(|_| AtomicI64::new(0)).collect(),
            baselines: None,
            predictions: None,
            optimizer: None,
            bucket_counts,
            layout,
        }
    }

    /// Build storage with optional baseline buffer for VR-MCCFR.
    ///
    /// When `use_baselines` is `true`, allocates a zeroed baseline buffer
    /// with the same layout as regrets.
    #[must_use]
    pub fn new_with_baselines(tree: &GameTree, bucket_counts: [u16; 4], use_baselines: bool) -> Self {
        let mut storage = Self::new(tree, bucket_counts);
        if use_baselines {
            let total = storage.regrets.len();
            storage.baselines = Some((0..total).map(|_| AtomicI32::new(0)).collect());
        }
        storage
    }

    /// Flat-buffer index for a given (node, bucket, action) triple.
    #[inline]
    fn slot_index(&self, node_idx: u32, bucket: u16, action: usize) -> usize {
        let nl = &self.layout[node_idx as usize];
        Self::slot_offset(nl, bucket) + action
    }

    /// Read the baseline value for a (node, bucket, action) slot.
    ///
    /// Returns `0.0` when baselines are disabled (`None`), which causes
    /// the VR-MCCFR corrected formula to degenerate to standard sampling.
    #[inline]
    #[must_use]
    pub fn get_baseline(&self, node_idx: u32, bucket: u16, action: usize) -> f64 {
        self.baselines
            .as_ref()
            .map(|b| {
                f64::from(b[self.slot_index(node_idx, bucket, action)].load(Ordering::Relaxed))
                    / 1000.0
            })
            .unwrap_or(0.0)
    }

    /// Update the baseline with an exponential moving average.
    ///
    /// `alpha` controls the learning rate: `new = (1 - alpha) * old + alpha * value`.
    /// No-op when baselines are disabled (`None`).
    #[inline]
    pub fn update_baseline(
        &self,
        node_idx: u32,
        bucket: u16,
        action: usize,
        value: f64,
        alpha: f64,
    ) {
        if let Some(ref b) = self.baselines {
            let idx = self.slot_index(node_idx, bucket, action);
            let old = f64::from(b[idx].load(Ordering::Relaxed)) / 1000.0;
            let new_val = old * (1.0 - alpha) + value * alpha;
            b[idx].store((new_val * 1000.0) as i32, Ordering::Relaxed);
        }
    }

    /// Read a single regret value atomically.
    #[inline]
    #[must_use]
    pub fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        self.regrets[idx].load(Ordering::Relaxed)
    }

    /// Add a delta to a single regret value atomically.
    /// Panics if the result would overflow i32 — this indicates the discount
    /// interval is too large and regrets are accumulating without being scaled.
    #[inline]
    pub fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        let old = self.regrets[idx].fetch_add(delta, Ordering::Relaxed);
        // Check for overflow: if signs match and result flipped, we overflowed.
        if delta > 0 && old > 0 && old.checked_add(delta).is_none() {
            panic!(
                "Regret overflow at node={node_idx} bucket={bucket} action={action}: \
                 old={old} + delta={delta} exceeds i32::MAX. \
                 Reduce lcfr_discount_interval to apply discounting more frequently."
            );
        }
    }

    /// Read a single strategy sum value atomically.
    #[inline]
    #[must_use]
    pub fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        self.strategy_sums[idx].load(Ordering::Relaxed)
    }

    /// Add a delta to a single strategy sum value atomically.
    #[inline]
    pub fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64) {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        self.strategy_sums[idx].fetch_add(delta, Ordering::Relaxed);
    }

    /// Set the pluggable optimizer for this storage.
    pub fn set_optimizer(&mut self, optimizer: Arc<dyn CfrOptimizer>) {
        self.optimizer = Some(optimizer);
    }

    /// Allocate the prediction buffer (same size as regrets, zeroed).
    pub fn enable_predictions(&mut self) {
        let total = self.regrets.len();
        self.predictions = Some((0..total).map(|_| AtomicI32::new(0)).collect());
    }

    /// Read a prediction value for a (node, bucket, action) slot.
    /// Returns 0 when predictions are disabled.
    #[inline]
    #[must_use]
    pub fn get_prediction(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
        self.predictions
            .as_ref()
            .map_or(0, |p| p[self.slot_index(node_idx, bucket, action)].load(Ordering::Relaxed))
    }

    /// Write a prediction value for a (node, bucket, action) slot.
    /// No-op when predictions are disabled.
    #[inline]
    pub fn set_prediction(&self, node_idx: u32, bucket: u16, action: usize, value: i32) {
        if let Some(ref p) = self.predictions {
            p[self.slot_index(node_idx, bucket, action)].store(value, Ordering::Relaxed);
        }
    }

    /// Flat-buffer offset for a given (node, bucket) pair.
    ///
    /// Exposed for the optimizer trait to use when computing strategies.
    #[inline]
    #[must_use]
    pub fn slot_offset_for(&self, node_idx: u32, bucket: u16) -> usize {
        let nl = &self.layout[node_idx as usize];
        Self::slot_offset(nl, bucket)
    }

    /// Current strategy via regret matching.
    ///
    /// Returns action probabilities summing to 1.0. When all regrets are
    /// non-positive the uniform distribution is returned.
    #[must_use]
    pub fn current_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
        let nl = &self.layout[node_idx as usize];
        let num_actions = nl.num_actions as usize;
        let start = Self::slot_offset(nl, bucket);

        let mut positive_sum = 0.0_f64;
        let mut positives = Vec::with_capacity(num_actions);
        for i in 0..num_actions {
            let r = f64::from(self.regrets[start + i].load(Ordering::Relaxed).max(0));
            positives.push(r);
            positive_sum += r;
        }

        if positive_sum > 0.0 {
            positives.iter().map(|&r| r / positive_sum).collect()
        } else {
            vec![1.0 / num_actions as f64; num_actions]
        }
    }

    /// Write the current regret-matched strategy into a caller-supplied buffer.
    ///
    /// Same semantics as [`current_strategy`](Self::current_strategy) but
    /// avoids heap allocation on every call — critical for the MCCFR hot path.
    ///
    /// When a pluggable optimizer is set, delegates to its `current_strategy`
    /// method (which may use predictions for SAPCFR+). Otherwise, falls back
    /// to standard regret matching.
    ///
    /// `out` must have length >= `num_actions` for this node; only the first
    /// `num_actions` entries are written.
    #[inline]
    pub fn current_strategy_into(&self, node_idx: u32, bucket: u16, out: &mut [f64]) {
        let nl = &self.layout[node_idx as usize];
        let num_actions = nl.num_actions as usize;
        debug_assert!(
            out.len() >= num_actions,
            "buffer too small: {} < {num_actions}",
            out.len()
        );
        let out = &mut out[..num_actions];
        let start = Self::slot_offset(nl, bucket);

        if let Some(ref opt) = self.optimizer {
            opt.current_strategy(
                &self.regrets,
                self.predictions.as_deref(),
                start,
                num_actions,
                out,
            );
            return;
        }

        let mut positive_sum = 0.0_f64;
        for (i, slot) in out.iter_mut().enumerate() {
            let r = self.regrets[start + i].load(Ordering::Relaxed).max(0);
            *slot = f64::from(r);
            positive_sum += *slot;
        }
        if positive_sum > 0.0 {
            for o in out.iter_mut() {
                *o /= positive_sum;
            }
        } else {
            out.fill(1.0 / num_actions as f64);
        }
    }

    /// Average strategy from strategy sums (the final output).
    ///
    /// Returns action probabilities summing to 1.0. Falls back to
    /// uniform when no strategy mass has been accumulated.
    #[must_use]
    pub fn average_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
        let nl = &self.layout[node_idx as usize];
        let num_actions = nl.num_actions as usize;
        let start = Self::slot_offset(nl, bucket);

        let mut total = 0.0_f64;
        let mut sums = Vec::with_capacity(num_actions);
        for i in 0..num_actions {
            let s = self.strategy_sums[start + i].load(Ordering::Relaxed) as f64;
            sums.push(s);
            total += s;
        }

        if total > 0.0 {
            sums.iter().map(|&s| s / total).collect()
        } else {
            vec![1.0 / num_actions as f64; num_actions]
        }
    }

    /// Average strategy with low-frequency actions zeroed and renormalized.
    ///
    /// Any action with probability below `threshold` (e.g. 0.03 for 3%) is
    /// set to zero. The remaining probabilities are renormalized to sum to 1.
    /// If all actions fall below the threshold, returns uniform.
    #[must_use]
    pub fn purified_average_strategy(
        &self,
        node_idx: u32,
        bucket: u16,
        threshold: f64,
    ) -> Vec<f64> {
        let mut strat = self.average_strategy(node_idx, bucket);
        for p in &mut strat {
            if *p < threshold {
                *p = 0.0;
            }
        }
        let sum: f64 = strat.iter().sum();
        if sum > 0.0 {
            for p in &mut strat {
                *p /= sum;
            }
        } else {
            let uniform = 1.0 / strat.len() as f64;
            strat.fill(uniform);
        }
        strat
    }

    /// Number of actions at a decision node.
    #[inline]
    #[must_use]
    pub fn num_actions(&self, node_idx: u32) -> u16 {
        self.layout[node_idx as usize].num_actions
    }

    /// Street index for a decision node (0=preflop .. 3=river).
    #[inline]
    #[must_use]
    pub fn street_idx(&self, node_idx: u32) -> u8 {
        self.layout[node_idx as usize].street_idx
    }

    /// Compute mean absolute strategy change vs a previous strategy-sum snapshot.
    ///
    /// `prev_sums` must have the same length as `self.strategy_sums`.
    /// For each (node, bucket) group, normalises both old and new sums to
    /// probability distributions, then takes the max absolute difference
    /// across actions.
    ///
    /// Returns `(mean_delta, pct_moving)` where `mean_delta` is the mean of
    /// per-group max deltas (near 0 = stabilised) and `pct_moving` is the
    /// fraction of groups whose max delta exceeds 0.20.
    ///
    /// # Panics
    /// Panics if `prev_sums.len()` differs from `self.strategy_sums.len()`.
    #[must_use]
    pub fn strategy_delta(&self, prev_sums: &[i64]) -> (f64, f64) {
        assert_eq!(prev_sums.len(), self.strategy_sums.len());
        let mut total_delta = 0.0_f64;
        let mut num_groups = 0_u64;
        let mut moving_count = 0_u64;

        for nl in &self.layout {
            if nl.num_actions == 0 {
                continue;
            }
            let n = nl.num_actions as usize;
            let buckets = self.bucket_counts[nl.street_idx as usize] as usize;
            for b in 0..buckets {
                let start = nl.offset + b * n;

                // Normalise previous sums.
                let prev_total: f64 = (0..n).map(|i| prev_sums[start + i] as f64).sum();
                // Normalise current sums.
                let curr_total: f64 = (0..n)
                    .map(|i| self.strategy_sums[start + i].load(Ordering::Relaxed) as f64)
                    .sum();

                let mut max_diff = 0.0_f64;
                for i in 0..n {
                    let p = if prev_total > 0.0 {
                        prev_sums[start + i] as f64 / prev_total
                    } else {
                        1.0 / n as f64
                    };
                    let c = if curr_total > 0.0 {
                        self.strategy_sums[start + i].load(Ordering::Relaxed) as f64 / curr_total
                    } else {
                        1.0 / n as f64
                    };
                    max_diff = max_diff.max((p - c).abs());
                }
                total_delta += max_diff;
                if max_diff > 0.01 {
                    moving_count += 1;
                }
                num_groups += 1;
            }
        }

        if num_groups > 0 {
            (
                total_delta / num_groups as f64,
                moving_count as f64 / num_groups as f64,
            )
        } else {
            (0.0, 0.0)
        }
    }

    /// Snapshot the current strategy sums as plain `i64` values.
    #[must_use]
    pub fn snapshot_strategy_sums(&self) -> Vec<i64> {
        self.strategy_sums
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect()
    }

    /// Serialize regrets and strategy sums to a binary file.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file cannot be created or written.
    pub fn save_regrets(&self, path: &Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        let regrets_plain: Vec<i32> = self
            .regrets
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect();
        let sums_plain: Vec<i64> = self
            .strategy_sums
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect();

        let payload = (&self.bucket_counts, &regrets_plain, &sums_plain);
        bincode::serialize_into(&mut writer, &payload)
            .map_err(|e| std::io::Error::other(e.to_string()))?;

        writer.flush()
    }

    /// Deserialize regrets and strategy sums, rebuilding layout from the tree.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file cannot be opened, parsed, or if bucket
    /// counts or buffer lengths do not match expectations.
    pub fn load_regrets(
        path: &Path,
        tree: &GameTree,
        bucket_counts: [u16; 4],
    ) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let reader = std::io::BufReader::new(file);

        let (stored_counts, regrets_plain, sums_plain): ([u16; 4], Vec<i32>, Vec<i64>) =
            bincode::deserialize_from(reader)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if stored_counts != bucket_counts {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "bucket count mismatch: file has {stored_counts:?}, expected {bucket_counts:?}"
                ),
            ));
        }

        // Rebuild layout from tree (the layout is not serialized).
        let storage = Self::new(tree, bucket_counts);

        if regrets_plain.len() != storage.regrets.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "regret buffer length mismatch: file has {}, expected {}",
                    regrets_plain.len(),
                    storage.regrets.len()
                ),
            ));
        }
        if sums_plain.len() != storage.strategy_sums.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "strategy_sums buffer length mismatch: file has {}, expected {}",
                    sums_plain.len(),
                    storage.strategy_sums.len()
                ),
            ));
        }

        for (atom, &val) in storage.regrets.iter().zip(regrets_plain.iter()) {
            atom.store(val, Ordering::Relaxed);
        }
        for (atom, &val) in storage.strategy_sums.iter().zip(sums_plain.iter()) {
            atom.store(val, Ordering::Relaxed);
        }

        Ok(storage)
    }

    /// Byte offset into the flat buffer for a given node layout and bucket.
    #[inline]
    fn slot_offset(nl: &NodeLayout, bucket: u16) -> usize {
        nl.offset + (bucket as usize) * (nl.num_actions as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::game_tree::GameTree;

    fn toy_tree() -> GameTree {
        GameTree::build(
            10.0,
            0.5,
            1.0,
            &[vec!["2.5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    #[test]
    fn storage_creation() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);
        assert!(!storage.regrets.is_empty());
        assert_eq!(storage.regrets.len(), storage.strategy_sums.len());
    }

    #[test]
    fn current_strategy_uniform_initially() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        let (node_idx, num_actions) = tree
            .nodes
            .iter()
            .enumerate()
            .find_map(|(i, n)| match n {
                GameNode::Decision { actions, .. } => Some((i as u32, actions.len())),
                _ => None,
            })
            .expect("tree should have at least one decision node");

        let strategy = storage.current_strategy(node_idx, 0);
        assert_eq!(strategy.len(), num_actions);
        let expected = 1.0 / num_actions as f64;
        for &p in &strategy {
            assert!(
                (p - expected).abs() < 1e-10,
                "expected uniform {expected}, got {p}"
            );
        }
    }

    #[test]
    fn regret_update_changes_strategy() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        let num_actions = storage.num_actions(node_idx) as usize;
        if num_actions >= 2 {
            storage.add_regret(node_idx, 0, 0, 1000);
            // Actions 1..n stay at 0
        }

        let strategy = storage.current_strategy(node_idx, 0);
        assert!(
            (strategy[0] - 1.0).abs() < 1e-10,
            "action 0 should have probability 1.0, got {}",
            strategy[0]
        );
    }

    #[test]
    fn average_strategy_proportional() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        let num_actions = storage.num_actions(node_idx) as usize;
        if num_actions >= 2 {
            storage.add_strategy_sum(node_idx, 0, 0, 300);
            storage.add_strategy_sum(node_idx, 0, 1, 700);
            // Actions 2..n stay at 0
        }

        let avg = storage.average_strategy(node_idx, 0);
        assert!((avg[0] - 0.3).abs() < 1e-10);
        assert!((avg[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn save_load_round_trip() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        storage.add_regret(node_idx, 5, 0, 42);
        storage.add_strategy_sum(node_idx, 5, 0, 999);

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("storage.bin");
        storage.save_regrets(&path).expect("save");

        let loaded =
            BlueprintStorage::load_regrets(&path, &tree, [50, 50, 50, 50]).expect("load");

        assert_eq!(loaded.get_regret(node_idx, 5, 0), 42);
        assert_eq!(loaded.get_strategy_sum(node_idx, 5, 0), 999);
    }

    #[test]
    fn load_rejects_mismatched_bucket_counts() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("storage.bin");
        storage.save_regrets(&path).expect("save");

        let result = BlueprintStorage::load_regrets(&path, &tree, [100, 50, 50, 50]);
        assert!(result.is_err(), "should reject mismatched bucket counts");
    }

    #[test]
    fn different_buckets_are_independent() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        storage.add_regret(node_idx, 0, 0, 100);
        storage.add_regret(node_idx, 1, 0, 200);

        assert_eq!(storage.get_regret(node_idx, 0, 0), 100);
        assert_eq!(storage.get_regret(node_idx, 1, 0), 200);
    }

    #[test]
    fn num_actions_matches_tree() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision { actions, .. } = node {
                assert_eq!(
                    storage.num_actions(i as u32) as usize,
                    actions.len(),
                    "num_actions mismatch at node {i}"
                );
            }
        }
    }

    #[test]
    fn strategy_delta_zero_for_identical() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        let snap = storage.snapshot_strategy_sums();
        let (delta, pct) = storage.strategy_delta(&snap);
        assert!((delta - 0.0).abs() < 1e-10, "identical snapshots → delta 0, got {delta}");
        assert!((pct - 0.0).abs() < 1e-10, "identical snapshots → pct_moving 0, got {pct}");
    }

    #[test]
    fn strategy_delta_detects_change() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        // Snapshot with initial strategy sums (all zero → uniform).
        let snap = storage.snapshot_strategy_sums();

        // Now change one action to dominate.
        storage.add_strategy_sum(node_idx, 0, 0, 1000);

        let (delta, _pct) = storage.strategy_delta(&snap);
        assert!(delta > 0.0, "changed strategy should have delta > 0, got {delta}");
    }

    // --- Baseline buffer tests ---

    #[test]
    fn baseline_returns_zero_when_disabled() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        assert!((storage.get_baseline(node_idx, 0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn baseline_update_is_noop_when_disabled() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        storage.update_baseline(node_idx, 0, 0, 5.0, 0.1);
        assert!((storage.get_baseline(node_idx, 0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn baseline_enabled_initial_zero() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        assert!((storage.get_baseline(node_idx, 0, 0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn baseline_ema_single_update() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        // EMA: new = (1 - alpha) * old + alpha * value
        // old = 0.0, alpha = 0.5, value = 10.0 => new = 5.0
        storage.update_baseline(node_idx, 0, 0, 10.0, 0.5);
        let b = storage.get_baseline(node_idx, 0, 0);
        assert!((b - 5.0).abs() < 0.01, "expected ~5.0 after EMA update, got {b}");
    }

    #[test]
    fn baseline_ema_multiple_updates() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        let alpha = 0.1;
        // Update 1: new = 0.0 * 0.9 + 10.0 * 0.1 = 1.0
        storage.update_baseline(node_idx, 0, 0, 10.0, alpha);
        // Update 2: new = 1.0 * 0.9 + 10.0 * 0.1 = 1.9
        storage.update_baseline(node_idx, 0, 0, 10.0, alpha);
        // Update 3: new = 1.9 * 0.9 + 10.0 * 0.1 = 2.71
        storage.update_baseline(node_idx, 0, 0, 10.0, alpha);
        let b = storage.get_baseline(node_idx, 0, 0);
        assert!((b - 2.71).abs() < 0.05, "expected ~2.71 after 3 EMA updates, got {b}");
    }

    #[test]
    fn baseline_different_actions_independent() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        let num_actions = storage.num_actions(node_idx) as usize;
        if num_actions >= 2 {
            storage.update_baseline(node_idx, 0, 0, 10.0, 1.0);
            storage.update_baseline(node_idx, 0, 1, 20.0, 1.0);
            let b0 = storage.get_baseline(node_idx, 0, 0);
            let b1 = storage.get_baseline(node_idx, 0, 1);
            assert!((b0 - 10.0).abs() < 0.01, "action 0 baseline: expected ~10.0, got {b0}");
            assert!((b1 - 20.0).abs() < 0.01, "action 1 baseline: expected ~20.0, got {b1}");
        }
    }

    #[test]
    fn baseline_different_buckets_independent() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        storage.update_baseline(node_idx, 0, 0, 10.0, 1.0);
        storage.update_baseline(node_idx, 1, 0, 30.0, 1.0);
        let b0 = storage.get_baseline(node_idx, 0, 0);
        let b1 = storage.get_baseline(node_idx, 1, 0);
        assert!((b0 - 10.0).abs() < 0.01, "bucket 0: expected ~10.0, got {b0}");
        assert!((b1 - 30.0).abs() < 0.01, "bucket 1: expected ~30.0, got {b1}");
    }

    #[test]
    fn baseline_alpha_one_replaces_completely() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        storage.update_baseline(node_idx, 0, 0, 7.5, 1.0);
        let b = storage.get_baseline(node_idx, 0, 0);
        assert!((b - 7.5).abs() < 0.01, "alpha=1 should fully replace: expected 7.5, got {b}");
        storage.update_baseline(node_idx, 0, 0, -3.0, 1.0);
        let b = storage.get_baseline(node_idx, 0, 0);
        assert!((b - (-3.0)).abs() < 0.01, "alpha=1 second update: expected -3.0, got {b}");
    }

    #[test]
    fn baseline_alpha_zero_keeps_old_value() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new_with_baselines(&tree, [50, 50, 50, 50], true);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        storage.update_baseline(node_idx, 0, 0, 5.0, 1.0);
        storage.update_baseline(node_idx, 0, 0, 99.0, 0.0);
        let b = storage.get_baseline(node_idx, 0, 0);
        assert!((b - 5.0).abs() < 0.01, "alpha=0 should keep old: expected 5.0, got {b}");
    }

    // --- Prediction buffer tests ---

    #[test]
    fn predictions_none_by_default() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        assert!(storage.predictions.is_none());
    }

    #[test]
    fn enable_predictions_allocates_buffer() {
        let tree = toy_tree();
        let mut storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        storage.enable_predictions();
        let preds = storage.predictions.as_ref().expect("predictions should be Some");
        assert_eq!(preds.len(), storage.regrets.len());
        assert!(preds.iter().all(|a| a.load(Ordering::Relaxed) == 0));
    }

    #[test]
    fn prediction_get_set_round_trip() {
        let tree = toy_tree();
        let mut storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        storage.enable_predictions();
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        storage.set_prediction(node_idx, 0, 0, 42);
        assert_eq!(storage.get_prediction(node_idx, 0, 0), 42);
    }

    #[test]
    fn prediction_get_returns_zero_when_none() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        assert_eq!(storage.get_prediction(node_idx, 0, 0), 0);
    }

    #[test]
    fn prediction_set_is_noop_when_none() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        // Should not panic or allocate.
        storage.set_prediction(node_idx, 0, 0, 99);
        assert_eq!(storage.get_prediction(node_idx, 0, 0), 0);
    }

    #[test]
    fn slot_offset_for_consistency() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);
        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;
        // slot_offset_for should give the same base as what slot_index uses
        let base = storage.slot_offset_for(node_idx, 3);
        let idx = storage.slot_index(node_idx, 3, 0);
        assert_eq!(base, idx, "slot_offset_for + 0 should equal slot_index");
    }
}
