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

use super::game_tree::{GameNode, GameTree};

/// Flat-buffer storage for regrets and strategy sums.
pub struct BlueprintStorage {
    /// Cumulative regrets: one `i32` per (decision node, bucket, action).
    pub regrets: Vec<i32>,
    /// Strategy sums: one `i64` per (decision node, bucket, action).
    pub strategy_sums: Vec<i64>,
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
            regrets: vec![0i32; total],
            strategy_sums: vec![0i64; total],
            bucket_counts,
            layout,
        }
    }

    /// Regret slice for a decision node and bucket (length = `num_actions`).
    #[inline]
    #[must_use]
    pub fn get_regrets(&self, node_idx: u32, bucket: u16) -> &[i32] {
        let nl = &self.layout[node_idx as usize];
        let start = Self::slot_offset(nl, bucket);
        &self.regrets[start..start + nl.num_actions as usize]
    }

    /// Mutable regret slice for a decision node and bucket.
    #[inline]
    pub fn get_regrets_mut(&mut self, node_idx: u32, bucket: u16) -> &mut [i32] {
        let nl = self.layout[node_idx as usize];
        let start = Self::slot_offset(&nl, bucket);
        let end = start + nl.num_actions as usize;
        &mut self.regrets[start..end]
    }

    /// Strategy-sum slice for a decision node and bucket.
    #[inline]
    #[must_use]
    pub fn get_strategy_sums(&self, node_idx: u32, bucket: u16) -> &[i64] {
        let nl = &self.layout[node_idx as usize];
        let start = Self::slot_offset(nl, bucket);
        &self.strategy_sums[start..start + nl.num_actions as usize]
    }

    /// Mutable strategy-sum slice for a decision node and bucket.
    #[inline]
    pub fn get_strategy_sums_mut(&mut self, node_idx: u32, bucket: u16) -> &mut [i64] {
        let nl = self.layout[node_idx as usize];
        let start = Self::slot_offset(&nl, bucket);
        let end = start + nl.num_actions as usize;
        &mut self.strategy_sums[start..end]
    }

    /// Current strategy via regret matching.
    ///
    /// Returns action probabilities summing to 1.0. When all regrets are
    /// non-positive the uniform distribution is returned.
    #[must_use]
    pub fn current_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
        let regrets = self.get_regrets(node_idx, bucket);
        let positive_sum: f64 = regrets.iter().map(|&r| f64::from(r.max(0))).sum();
        if positive_sum > 0.0 {
            regrets
                .iter()
                .map(|&r| f64::from(r.max(0)) / positive_sum)
                .collect()
        } else {
            let n = regrets.len() as f64;
            vec![1.0 / n; regrets.len()]
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
        let regrets = self.get_regrets(node_idx, bucket);
        let num_actions = regrets.len();
        debug_assert!(
            out.len() >= num_actions,
            "buffer too small: {} < {num_actions}",
            out.len()
        );
        let out = &mut out[..num_actions];

        let positive_sum: f64 = regrets.iter().map(|&r| f64::from(r.max(0))).sum();
        if positive_sum > 0.0 {
            for (o, &r) in out.iter_mut().zip(regrets) {
                *o = f64::from(r.max(0)) / positive_sum;
            }
        } else {
            let u = 1.0 / num_actions as f64;
            out.fill(u);
        }
    }

    /// Average strategy from strategy sums (the final output).
    ///
    /// Returns action probabilities summing to 1.0. Falls back to
    /// uniform when no strategy mass has been accumulated.
    #[must_use]
    pub fn average_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
        let sums = self.get_strategy_sums(node_idx, bucket);
        let total: f64 = sums.iter().map(|&s| s as f64).sum();
        if total > 0.0 {
            sums.iter().map(|&s| s as f64 / total).collect()
        } else {
            let n = sums.len() as f64;
            vec![1.0 / n; sums.len()]
        }
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

    /// Serialize regrets and strategy sums to a binary file.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the file cannot be created or written.
    pub fn save_regrets(&self, path: &Path) -> std::io::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);

        let payload = (
            &self.bucket_counts,
            &self.regrets,
            &self.strategy_sums,
        );
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

        let (stored_counts, regrets, strategy_sums): ([u16; 4], Vec<i32>, Vec<i64>) =
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
        let mut storage = Self::new(tree, bucket_counts);

        if regrets.len() != storage.regrets.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "regret buffer length mismatch: file has {}, expected {}",
                    regrets.len(),
                    storage.regrets.len()
                ),
            ));
        }
        if strategy_sums.len() != storage.strategy_sums.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "strategy_sums buffer length mismatch: file has {}, expected {}",
                    strategy_sums.len(),
                    storage.strategy_sums.len()
                ),
            ));
        }

        storage.regrets = regrets;
        storage.strategy_sums = strategy_sums;
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
            1,
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
        let mut storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        let regrets = storage.get_regrets_mut(node_idx, 0);
        if regrets.len() >= 2 {
            regrets[0] = 1000;
            regrets[1] = 0;
            for r in regrets.iter_mut().skip(2) {
                *r = 0;
            }
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
        let mut storage = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        let sums = storage.get_strategy_sums_mut(node_idx, 0);
        if sums.len() >= 2 {
            sums[0] = 300;
            sums[1] = 700;
            for s in sums.iter_mut().skip(2) {
                *s = 0;
            }
        }

        let avg = storage.average_strategy(node_idx, 0);
        assert!((avg[0] - 0.3).abs() < 1e-10);
        assert!((avg[1] - 0.7).abs() < 1e-10);
    }

    #[test]
    fn save_load_round_trip() {
        let tree = toy_tree();
        let mut storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        storage.get_regrets_mut(node_idx, 5)[0] = 42;
        storage.get_strategy_sums_mut(node_idx, 5)[0] = 999;

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("storage.bin");
        storage.save_regrets(&path).expect("save");

        let loaded =
            BlueprintStorage::load_regrets(&path, &tree, [50, 50, 50, 50]).expect("load");

        assert_eq!(loaded.get_regrets(node_idx, 5)[0], 42);
        assert_eq!(loaded.get_strategy_sums(node_idx, 5)[0], 999);
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
        let mut storage = BlueprintStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        storage.get_regrets_mut(node_idx, 0)[0] = 100;
        storage.get_regrets_mut(node_idx, 1)[0] = 200;

        assert_eq!(storage.get_regrets(node_idx, 0)[0], 100);
        assert_eq!(storage.get_regrets(node_idx, 1)[0], 200);
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
}
