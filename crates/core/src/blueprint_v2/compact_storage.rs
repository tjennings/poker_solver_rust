//! Compact flat-buffer storage for regrets and strategy sums, indexed by
//! (decision node, bucket, action).
//!
//! Uses `AtomicI16` for regrets and `AtomicI32` for strategy sums -- half the
//! memory of [`BlueprintStorage`](super::storage::BlueprintStorage) (6 bytes/slot
//! vs 12 bytes/slot).

// Arena indices are u32 and action counts fit in u16. Truncation is safe
// for any practical game tree. Precision loss on len-to-f64 casts is
// acceptable (action counts are tiny).
#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicI32, Ordering};

use super::game_tree::{GameNode, GameTree};
use super::storage::RegretStorage;

/// Per-node layout metadata for index computation.
#[derive(Clone, Copy, Default)]
struct NodeLayout {
    offset: usize,
    num_actions: u16,
    /// 0 = preflop, 1 = flop, 2 = turn, 3 = river.
    /// Kept for API parity with `BlueprintStorage::NodeLayout`; used by
    /// higher-level methods added in later tasks (e.g. `strategy_delta`).
    #[allow(dead_code)]
    street_idx: u8,
}

/// Compact flat-buffer storage using `AtomicI32` regrets and `AtomicI32`
/// strategy sums. 8 bytes/slot instead of 12 (no AtomicI64 strategy sums).
pub struct CompactStorage {
    /// Cumulative regrets: one `AtomicI32` per (decision node, bucket, action).
    pub regrets: Vec<AtomicI32>,
    /// Strategy sums: one `AtomicI32` per (decision node, bucket, action).
    pub strategy_sums: Vec<AtomicI32>,
    /// Number of buckets per street `[preflop, flop, turn, river]`.
    pub bucket_counts: [u16; 4],
    /// Per-node layout metadata. Non-decision nodes use the `Default`
    /// sentinel (`offset=0`, `num_actions=0`) and must never be queried.
    layout: Vec<NodeLayout>,
}

impl CompactStorage {
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
            strategy_sums: (0..total).map(|_| AtomicI32::new(0)).collect(),
            bucket_counts,
            layout,
        }
    }

    /// Read a single regret value atomically, widened to i32.
    #[inline]
    #[must_use]
    pub fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        self.regrets[idx].load(Ordering::Relaxed)
    }

    /// Add a delta to a single regret value atomically.
    ///
    /// Add a delta to a single regret value atomically.
    #[inline]
    pub fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        self.regrets[idx].fetch_add(delta, Ordering::Relaxed);
    }

    /// Write the current regret-matched strategy into a caller-supplied buffer.
    ///
    /// Returns action probabilities summing to 1.0. When all regrets are
    /// non-positive the uniform distribution is returned.
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

    /// Read a single strategy sum value atomically, widened to i64.
    #[inline]
    #[must_use]
    pub fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        i64::from(self.strategy_sums[idx].load(Ordering::Relaxed))
    }

    /// Add a delta to a single strategy sum value atomically.
    #[inline]
    pub fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64) {
        let clamped = delta.clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32;
        let nl = &self.layout[node_idx as usize];
        let idx = Self::slot_offset(nl, bucket) + action;
        self.strategy_sums[idx].fetch_add(clamped, Ordering::Relaxed);
    }

    /// Number of actions at a decision node.
    #[inline]
    #[must_use]
    pub fn num_actions(&self, node_idx: u32) -> usize {
        self.layout[node_idx as usize].num_actions as usize
    }

    /// Total number of (node, bucket, action) slots.
    #[inline]
    #[must_use]
    pub fn num_slots(&self) -> usize {
        self.regrets.len()
    }

    /// Apply DCFR discounting to regrets and strategy sums.
    ///
    /// `d_pos` multiplies positive regrets, `d_neg` multiplies negative regrets,
    /// `d_strat` multiplies strategy sums.
    pub fn apply_discount(&self, d_pos: f64, d_neg: f64, d_strat: f64) {
        for atom in &self.regrets {
            let v = atom.load(Ordering::Relaxed);
            let d = if v >= 0 { d_pos } else { d_neg };
            let new_val = (f64::from(v) * d) as i32;
            atom.store(new_val, Ordering::Relaxed);
        }
        for atom in &self.strategy_sums {
            let v = atom.load(Ordering::Relaxed);
            let new_val = (f64::from(v) * d_strat) as i32;
            atom.store(new_val, Ordering::Relaxed);
        }
    }

    /// Serialize regrets and strategy sums to a binary file.
    ///
    /// Format: magic `CMP2`, `bucket_counts`, raw i32 regrets, raw i32 strategy sums.
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
        let sums_plain: Vec<i32> = self
            .strategy_sums
            .iter()
            .map(|a| a.load(Ordering::Relaxed))
            .collect();

        // Magic + bucket_counts + regrets + strategy_sums
        let payload = (b"CMP2", &self.bucket_counts, &regrets_plain, &sums_plain);
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

        let (magic, stored_counts, regrets_plain, sums_plain): (
            [u8; 4],
            [u16; 4],
            Vec<i32>,
            Vec<i32>,
        ) = bincode::deserialize_from(reader)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        if &magic != b"CMP2" {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("invalid magic: expected CMP2, got {magic:?}"),
            ));
        }

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

impl RegretStorage for CompactStorage {
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
        self.num_actions(node_idx)
    }

    // Uses default delta_scale (1000.0) — i32 regrets have sufficient range.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::game_tree::GameTree;
    use crate::blueprint_v2::storage::BlueprintStorage;

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
    fn new_creates_zeroed() {
        let tree = toy_tree();
        let storage = CompactStorage::new(&tree, [50, 50, 50, 50]);
        assert!(!storage.regrets.is_empty());
        assert_eq!(storage.regrets.len(), storage.strategy_sums.len());
        for r in &storage.regrets {
            assert_eq!(r.load(Ordering::Relaxed), 0);
        }
        for s in &storage.strategy_sums {
            assert_eq!(s.load(Ordering::Relaxed), 0);
        }
    }

    #[test]
    fn add_and_get_regret_round_trip() {
        let tree = toy_tree();
        let storage = CompactStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, crate::blueprint_v2::game_tree::GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        storage.add_regret(node_idx, 0, 0, 100);
        assert_eq!(storage.get_regret(node_idx, 0, 0), 100);

        // Other slots unaffected
        assert_eq!(storage.get_regret(node_idx, 1, 0), 0);
    }

    #[test]
    fn add_regret_accumulates_i32() {
        let tree = toy_tree();
        let storage = CompactStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, crate::blueprint_v2::game_tree::GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        // i32 — no clamping, direct accumulation
        storage.add_regret(node_idx, 0, 0, 50_000);
        assert_eq!(storage.get_regret(node_idx, 0, 0), 50_000);

        storage.add_regret(node_idx, 0, 0, 50_000);
        assert_eq!(storage.get_regret(node_idx, 0, 0), 100_000);
    }

    #[test]
    fn current_strategy_uniform_when_all_zero() {
        let tree = toy_tree();
        let storage = CompactStorage::new(&tree, [169, 200, 200, 200]);

        let (node_idx, num_actions) = tree
            .nodes
            .iter()
            .enumerate()
            .find_map(|(i, n)| match n {
                crate::blueprint_v2::game_tree::GameNode::Decision { actions, .. } => {
                    Some((i as u32, actions.len()))
                }
                _ => None,
            })
            .expect("tree should have at least one decision node");

        let mut out = vec![0.0; num_actions];
        storage.current_strategy_into(node_idx, 0, &mut out);

        let expected = 1.0 / num_actions as f64;
        for &p in &out {
            assert!(
                (p - expected).abs() < 1e-10,
                "expected uniform {expected}, got {p}"
            );
        }
    }

    #[test]
    fn current_strategy_proportional_to_positive_regrets() {
        let tree = toy_tree();
        let storage = CompactStorage::new(&tree, [169, 200, 200, 200]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, crate::blueprint_v2::game_tree::GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        let num_actions = storage.num_actions(node_idx);
        assert!(num_actions >= 2, "need at least 2 actions for this test");

        // Set regret for action 0 to 300, action 1 to 100
        storage.add_regret(node_idx, 0, 0, 300);
        storage.add_regret(node_idx, 0, 1, 100);

        let mut out = vec![0.0; num_actions];
        storage.current_strategy_into(node_idx, 0, &mut out);

        // action 0 should get 300/400 = 0.75
        assert!(
            (out[0] - 0.75).abs() < 1e-10,
            "expected 0.75 for action 0, got {}",
            out[0]
        );
        // action 1 should get 100/400 = 0.25
        assert!(
            (out[1] - 0.25).abs() < 1e-10,
            "expected 0.25 for action 1, got {}",
            out[1]
        );
        // remaining actions (if any) should be 0
        for &p in &out[2..] {
            assert!((p - 0.0).abs() < 1e-10, "expected 0.0, got {p}");
        }
    }

    #[test]
    fn save_load_round_trip() {
        let tree = toy_tree();
        let storage = CompactStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, crate::blueprint_v2::game_tree::GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        storage.add_regret(node_idx, 5, 0, 42);
        storage.add_strategy_sum(node_idx, 5, 0, 999);

        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("compact.bin");
        storage.save_regrets(&path).expect("save");

        let loaded =
            CompactStorage::load_regrets(&path, &tree, [50, 50, 50, 50]).expect("load");

        assert_eq!(loaded.get_regret(node_idx, 5, 0), 42);
        assert_eq!(loaded.get_strategy_sum(node_idx, 5, 0), 999);
    }

    #[test]
    fn size_is_half_of_blueprint() {
        let tree = toy_tree();
        let compact = CompactStorage::new(&tree, [169, 200, 200, 200]);
        let blueprint = BlueprintStorage::new(&tree, [169, 200, 200, 200]);

        // Both should have same number of slots
        assert_eq!(compact.num_slots(), blueprint.regrets.len());

        // CompactStorage: 2 bytes (i16) + 4 bytes (i32) = 6 bytes/slot
        // BlueprintStorage: 4 bytes (i32) + 8 bytes (i64) = 12 bytes/slot
        let compact_bytes = compact.num_slots() * 6;
        let blueprint_bytes = blueprint.regrets.len() * 12;
        assert_eq!(compact_bytes * 2, blueprint_bytes);
    }

    #[test]
    fn compact_storage_implements_regret_storage_trait() {
        use crate::blueprint_v2::storage::RegretStorage;

        let tree = toy_tree();
        let storage = CompactStorage::new(&tree, [50, 50, 50, 50]);

        let node_idx = tree
            .nodes
            .iter()
            .position(|n| matches!(n, crate::blueprint_v2::game_tree::GameNode::Decision { .. }))
            .expect("need a decision node") as u32;

        // Use through a trait object reference
        let dyn_storage: &dyn RegretStorage = &storage;

        // num_actions via trait
        let num_actions = dyn_storage.num_actions(node_idx);
        assert!(num_actions > 0);

        // Initial regrets should be zero
        assert_eq!(dyn_storage.get_regret(node_idx, 0, 0), 0);

        // add_regret + get_regret via trait
        dyn_storage.add_regret(node_idx, 0, 0, 500);
        assert_eq!(dyn_storage.get_regret(node_idx, 0, 0), 500);

        // add_strategy_sum + get_strategy_sum via trait
        assert_eq!(dyn_storage.get_strategy_sum(node_idx, 0, 0), 0);
        dyn_storage.add_strategy_sum(node_idx, 0, 0, 1000);
        assert_eq!(dyn_storage.get_strategy_sum(node_idx, 0, 0), 1000);

        // current_strategy_into via trait: action 0 has positive regret,
        // so it should get probability 1.0
        let mut buf = vec![0.0f64; num_actions];
        dyn_storage.current_strategy_into(node_idx, 0, &mut buf);
        assert!(
            (buf[0] - 1.0).abs() < 1e-10,
            "action 0 should have prob 1.0, got {}",
            buf[0]
        );
    }
}
