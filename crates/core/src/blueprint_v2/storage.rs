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

use super::game_tree::{GameNode, GameTree};

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

/// Trait for regret/strategy storage used by MCCFR traversal.
pub trait RegretStorage: Sync {
    fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32;
    fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32);
    fn current_strategy_into(&self, node_idx: u32, bucket: u16, out: &mut [f64]);
    fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64;
    fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64);
    fn num_actions(&self, node_idx: u32) -> usize;
}

/// Flat-buffer storage for regrets and strategy sums.
pub struct BlueprintStorage {
    /// Cumulative regrets: one `AtomicI32` per (decision node, bucket, action).
    pub regrets: Vec<AtomicI32>,
    /// Strategy sums: one `AtomicI64` per (decision node, bucket, action).
    pub strategy_sums: Vec<AtomicI64>,
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


        Self {
            regrets: (0..total).map(|_| AtomicI32::new(0)).collect(),
            strategy_sums: (0..total).map(|_| AtomicI64::new(0)).collect(),
            bucket_counts,
            layout,
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
    #[inline]
    pub fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        self.regrets[idx].fetch_add(delta, Ordering::Relaxed);
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

impl RegretStorage for BlueprintStorage {
    #[inline]
    fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
        self.get_regret(node_idx, bucket, action)
    }

    #[inline]
    fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
        self.add_regret(node_idx, bucket, action, delta);
    }

    #[inline]
    fn current_strategy_into(&self, node_idx: u32, bucket: u16, out: &mut [f64]) {
        self.current_strategy_into(node_idx, bucket, out);
    }

    #[inline]
    fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
        self.get_strategy_sum(node_idx, bucket, action)
    }

    #[inline]
    fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64) {
        self.add_strategy_sum(node_idx, bucket, action, delta);
    }

    #[inline]
    fn num_actions(&self, node_idx: u32) -> usize {
        self.num_actions(node_idx) as usize
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

    #[test]
    fn blueprint_storage_implements_regret_storage_trait() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        // Use the trait through a dynamic reference.
        let dyn_storage: &dyn RegretStorage = &storage;

        // num_actions should match
        let num_actions = dyn_storage.num_actions(node_idx);
        assert!(num_actions > 0, "should have at least one action");

        // Initial regrets should be zero.
        assert_eq!(dyn_storage.get_regret(node_idx, 0, 0), 0);

        // Add regret via trait, read back.
        dyn_storage.add_regret(node_idx, 0, 0, 500);
        assert_eq!(dyn_storage.get_regret(node_idx, 0, 0), 500);

        // Initial strategy sums should be zero.
        assert_eq!(dyn_storage.get_strategy_sum(node_idx, 0, 0), 0);

        // Add strategy sum via trait, read back.
        dyn_storage.add_strategy_sum(node_idx, 0, 0, 1000);
        assert_eq!(dyn_storage.get_strategy_sum(node_idx, 0, 0), 1000);

        // current_strategy_into via trait.
        let mut buf = vec![0.0f64; num_actions];
        dyn_storage.current_strategy_into(node_idx, 0, &mut buf);
        // After adding 500 regret to action 0, action 0 should have probability 1.0.
        assert!(
            (buf[0] - 1.0).abs() < 1e-10,
            "action 0 should have prob 1.0, got {}",
            buf[0]
        );
    }
}
