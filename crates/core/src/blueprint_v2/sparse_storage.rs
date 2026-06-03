//! Sparse row-backed CFR storage for blueprint_v2 MCCFR.
//!
//! Missing rows have the same semantics as a dense all-zero row: regrets,
//! strategy sums, predictions, and baselines read as zero, while current and
//! average strategy fall back to uniform. Reads do not realize rows; writes do.
//! Dense projection validates the supplied tree against the layout captured at
//! construction time before interpreting sparse row keys.

#![allow(clippy::cast_possible_truncation)]

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::RwLock;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicI64, AtomicU64, Ordering};

use super::game_tree::{GameNode, GameTree};
use super::storage::{
    BlueprintCfrStorage, BlueprintStorage, CfrStorageStats, REGRET_SCALE, action_schema_fingerprint,
};
use crate::cfr::optimizer::CfrOptimizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparseStorageError {
    NonDecisionNode {
        node_idx: u32,
    },
    BucketOutOfRange {
        node_idx: u32,
        bucket: u16,
        bucket_count: u16,
    },
    TreeShapeMismatch {
        expected_nodes: usize,
        actual_nodes: usize,
    },
    SchemaMismatch {
        node_idx: u32,
        expected_fingerprint: u64,
        actual_fingerprint: u64,
        expected_num_actions: u16,
        actual_num_actions: u16,
    },
}

impl fmt::Display for SparseStorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::NonDecisionNode { node_idx } => {
                write!(f, "node {node_idx} is not a decision node")
            }
            Self::BucketOutOfRange {
                node_idx,
                bucket,
                bucket_count,
            } => write!(
                f,
                "bucket {bucket} is out of range for node {node_idx}; bucket_count={bucket_count}"
            ),
            Self::TreeShapeMismatch {
                expected_nodes,
                actual_nodes,
            } => write!(
                f,
                "tree node-count mismatch: expected {expected_nodes}, got {actual_nodes}"
            ),
            Self::SchemaMismatch {
                node_idx,
                expected_fingerprint,
                actual_fingerprint,
                expected_num_actions,
                actual_num_actions,
            } => write!(
                f,
                "schema mismatch for node {node_idx}: expected fingerprint={expected_fingerprint:#x} \
                 actions={expected_num_actions}, got fingerprint={actual_fingerprint:#x} \
                 actions={actual_num_actions}"
            ),
        }
    }
}

impl std::error::Error for SparseStorageError {}

/// Sparse in-memory CFR storage for heads-up blueprint_v2.
pub struct SparseBlueprintStorage {
    rows: RwLock<HashMap<SparseRowKey, Arc<SparseCfrRow>>>,
    layout: Vec<SparseNodeLayout>,
    pub bucket_counts: [u16; 4],
    dense_equivalent_slots: usize,
    baselines_enabled: AtomicBool,
    predictions_enabled: AtomicBool,
    predictions_locked: AtomicBool,
    regret_floor: AtomicI32,
    optimizer: Option<Arc<dyn CfrOptimizer>>,
    stats: SparseStats,
}

#[derive(Clone, Copy, Debug, Default)]
struct SparseNodeLayout {
    num_actions: u16,
    street_idx: u8,
    schema_fingerprint: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SparseRowKey {
    pub node_idx: u32,
    pub bucket: u16,
    pub schema_fingerprint: u64,
}

struct SparseCfrRow {
    num_actions: u16,
    regrets: Vec<AtomicI32>,
    strategy_sums: Vec<AtomicI64>,
    baselines: Vec<AtomicI32>,
    predictions: Vec<AtomicI32>,
}

#[derive(Default)]
struct SparseStats {
    inserts: AtomicU64,
    realized_slots: AtomicU64,
    read_probes: AtomicU64,
    read_hits: AtomicU64,
    write_probes: AtomicU64,
    write_hits: AtomicU64,
}

impl SparseBlueprintStorage {
    #[must_use]
    pub fn new(tree: &GameTree, bucket_counts: [u16; 4]) -> Self {
        Self::with_options(tree, bucket_counts, false)
    }

    #[must_use]
    pub fn new_with_baselines(
        tree: &GameTree,
        bucket_counts: [u16; 4],
        use_baselines: bool,
    ) -> Self {
        Self::with_options(tree, bucket_counts, use_baselines)
    }

    fn with_options(tree: &GameTree, bucket_counts: [u16; 4], use_baselines: bool) -> Self {
        let mut layout = vec![SparseNodeLayout::default(); tree.nodes.len()];
        let mut dense_equivalent_slots = 0usize;
        for (node_idx, node) in tree.nodes.iter().enumerate() {
            let GameNode::Decision {
                street, actions, ..
            } = node
            else {
                continue;
            };
            let num_actions = actions.len() as u16;
            let street_idx = *street as u8;
            layout[node_idx] = SparseNodeLayout {
                num_actions,
                street_idx,
                schema_fingerprint: action_schema_fingerprint(actions),
            };
            dense_equivalent_slots +=
                bucket_counts[street_idx as usize] as usize * usize::from(num_actions);
        }

        Self {
            rows: RwLock::new(HashMap::new()),
            layout,
            bucket_counts,
            dense_equivalent_slots,
            baselines_enabled: AtomicBool::new(use_baselines),
            predictions_enabled: AtomicBool::new(false),
            predictions_locked: AtomicBool::new(false),
            regret_floor: AtomicI32::new(i32::MIN),
            optimizer: None,
            stats: SparseStats::default(),
        }
    }

    pub fn set_optimizer(&mut self, optimizer: Arc<dyn CfrOptimizer>) {
        self.optimizer = Some(optimizer);
    }

    /// Set the lower bound applied to cumulative regret updates.
    pub fn set_regret_floor(&self, floor: i32) {
        self.regret_floor.store(floor, Ordering::Relaxed);
    }

    pub fn enable_predictions(&self) {
        self.predictions_enabled.store(true, Ordering::Relaxed);
    }

    pub fn lock_predictions(&self) {
        self.predictions_locked.store(true, Ordering::Relaxed);
    }

    pub fn unlock_predictions(&self) {
        self.predictions_locked.store(false, Ordering::Relaxed);
    }

    pub fn zero_predictions(&self) {
        let rows = self.rows.read().expect("sparse storage rows lock");
        for row in rows.values() {
            for prediction in &row.predictions {
                prediction.store(0, Ordering::Relaxed);
            }
        }
    }

    #[must_use]
    pub fn schema_fingerprint(&self, node_idx: u32) -> u64 {
        self.layout[node_idx as usize].schema_fingerprint
    }

    pub fn realize_row(&self, node_idx: u32, bucket: u16) -> Result<bool, SparseStorageError> {
        let layout = self.checked_layout(node_idx, bucket)?;
        self.realize_row_with_schema(
            node_idx,
            bucket,
            layout.schema_fingerprint,
            layout.num_actions,
        )
    }

    pub fn realize_row_with_schema(
        &self,
        node_idx: u32,
        bucket: u16,
        schema_fingerprint: u64,
        num_actions: u16,
    ) -> Result<bool, SparseStorageError> {
        let layout = self.checked_layout(node_idx, bucket)?;
        if layout.schema_fingerprint != schema_fingerprint || layout.num_actions != num_actions {
            return Err(SparseStorageError::SchemaMismatch {
                node_idx,
                expected_fingerprint: layout.schema_fingerprint,
                actual_fingerprint: schema_fingerprint,
                expected_num_actions: layout.num_actions,
                actual_num_actions: num_actions,
            });
        }
        Ok(self.get_or_insert_row(node_idx, bucket, layout).1)
    }

    #[must_use]
    pub fn is_realized(&self, node_idx: u32, bucket: u16) -> bool {
        let layout = self.layout[node_idx as usize];
        if layout.num_actions == 0 {
            return false;
        }
        let key = Self::row_key(node_idx, bucket, layout);
        self.rows
            .read()
            .expect("sparse storage rows lock")
            .contains_key(&key)
    }

    #[must_use]
    pub fn dense_regrets_and_sums(&self, tree: &GameTree) -> (Vec<i32>, Vec<i64>) {
        self.project_dense_regrets_and_sums(tree)
    }

    pub fn try_dense_regrets_and_sums(
        &self,
        tree: &GameTree,
    ) -> Result<(Vec<i32>, Vec<i64>), SparseStorageError> {
        self.validate_tree_schema(tree)?;
        Ok(self.project_dense_regrets_and_sums_unchecked(tree))
    }

    #[must_use]
    pub fn to_dense_storage(&self, tree: &GameTree) -> BlueprintStorage {
        self.validate_tree_schema(tree)
            .unwrap_or_else(|err| panic!("{err}"));
        let dense = BlueprintStorage::new(tree, self.bucket_counts);
        let rows = self.rows.read().expect("sparse storage rows lock");
        for (key, row) in rows.iter() {
            for action_idx in 0..usize::from(row.num_actions) {
                let slot = dense.slot_offset_for(key.node_idx, key.bucket) + action_idx;
                dense.regrets[slot].store(
                    row.regrets[action_idx].load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
                dense.strategy_sums[slot].store(
                    row.strategy_sums[action_idx].load(Ordering::Relaxed),
                    Ordering::Relaxed,
                );
            }
        }
        dense
    }

    fn validate_tree_schema(&self, tree: &GameTree) -> Result<(), SparseStorageError> {
        if tree.nodes.len() != self.layout.len() {
            return Err(SparseStorageError::TreeShapeMismatch {
                expected_nodes: self.layout.len(),
                actual_nodes: tree.nodes.len(),
            });
        }
        for (node_idx, node) in tree.nodes.iter().enumerate() {
            let layout = self.layout[node_idx];
            match node {
                GameNode::Decision {
                    street, actions, ..
                } => {
                    let actual_fingerprint = action_schema_fingerprint(actions);
                    let actual_num_actions = actions.len() as u16;
                    if layout.num_actions != actual_num_actions
                        || layout.street_idx != *street as u8
                        || layout.schema_fingerprint != actual_fingerprint
                    {
                        return Err(SparseStorageError::SchemaMismatch {
                            node_idx: node_idx as u32,
                            expected_fingerprint: layout.schema_fingerprint,
                            actual_fingerprint,
                            expected_num_actions: layout.num_actions,
                            actual_num_actions,
                        });
                    }
                }
                GameNode::Chance { .. } | GameNode::Terminal { .. } => {
                    if layout.num_actions != 0 {
                        return Err(SparseStorageError::NonDecisionNode {
                            node_idx: node_idx as u32,
                        });
                    }
                }
            }
        }
        Ok(())
    }

    fn checked_layout(
        &self,
        node_idx: u32,
        bucket: u16,
    ) -> Result<SparseNodeLayout, SparseStorageError> {
        let layout = self
            .layout
            .get(node_idx as usize)
            .copied()
            .ok_or(SparseStorageError::NonDecisionNode { node_idx })?;
        if layout.num_actions == 0 {
            return Err(SparseStorageError::NonDecisionNode { node_idx });
        }
        let bucket_count = self.bucket_counts[layout.street_idx as usize];
        if bucket >= bucket_count {
            return Err(SparseStorageError::BucketOutOfRange {
                node_idx,
                bucket,
                bucket_count,
            });
        }
        Ok(layout)
    }

    fn layout_or_panic(&self, node_idx: u32, bucket: u16) -> SparseNodeLayout {
        self.checked_layout(node_idx, bucket)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn row_key(node_idx: u32, bucket: u16, layout: SparseNodeLayout) -> SparseRowKey {
        SparseRowKey {
            node_idx,
            bucket,
            schema_fingerprint: layout.schema_fingerprint,
        }
    }

    fn get_row(
        &self,
        node_idx: u32,
        bucket: u16,
        layout: SparseNodeLayout,
    ) -> Option<Arc<SparseCfrRow>> {
        self.stats.read_probes.fetch_add(1, Ordering::Relaxed);
        let key = Self::row_key(node_idx, bucket, layout);
        let row = self
            .rows
            .read()
            .expect("sparse storage rows lock")
            .get(&key)
            .cloned();
        if row.is_some() {
            self.stats.read_hits.fetch_add(1, Ordering::Relaxed);
        }
        row
    }

    fn get_or_insert_row(
        &self,
        node_idx: u32,
        bucket: u16,
        layout: SparseNodeLayout,
    ) -> (Arc<SparseCfrRow>, bool) {
        self.stats.write_probes.fetch_add(1, Ordering::Relaxed);
        let key = Self::row_key(node_idx, bucket, layout);
        if let Some(row) = self
            .rows
            .read()
            .expect("sparse storage rows lock")
            .get(&key)
            .cloned()
        {
            self.stats.write_hits.fetch_add(1, Ordering::Relaxed);
            return (row, false);
        }

        let mut rows = self.rows.write().expect("sparse storage rows lock");
        if let Some(row) = rows.get(&key).cloned() {
            self.stats.write_hits.fetch_add(1, Ordering::Relaxed);
            return (row, false);
        }

        let row = Arc::new(SparseCfrRow::new(layout.num_actions));
        rows.insert(key, Arc::clone(&row));
        self.stats.inserts.fetch_add(1, Ordering::Relaxed);
        self.stats
            .realized_slots
            .fetch_add(u64::from(layout.num_actions), Ordering::Relaxed);
        (row, true)
    }

    fn dense_equivalent_bytes(&self) -> usize {
        let slot_bytes = std::mem::size_of::<AtomicI32>() + std::mem::size_of::<AtomicI64>();
        let baseline_bytes = if self.baselines_enabled.load(Ordering::Relaxed) {
            std::mem::size_of::<AtomicI32>()
        } else {
            0
        };
        let prediction_bytes = if self.predictions_enabled.load(Ordering::Relaxed) {
            std::mem::size_of::<AtomicI32>()
        } else {
            0
        };
        self.dense_equivalent_slots * (slot_bytes + baseline_bytes + prediction_bytes)
    }

    fn sparse_resident_bytes(&self, realized_rows: usize, realized_slots: usize) -> usize {
        let row_slot_bytes =
            std::mem::size_of::<AtomicI32>() * 3 + std::mem::size_of::<AtomicI64>();
        let row_overhead =
            std::mem::size_of::<SparseRowKey>() + std::mem::size_of::<SparseCfrRow>() + 64;
        realized_slots * row_slot_bytes + realized_rows * row_overhead
    }
}

impl SparseCfrRow {
    fn new(num_actions: u16) -> Self {
        let n = usize::from(num_actions);
        Self {
            num_actions,
            regrets: (0..n).map(|_| AtomicI32::new(0)).collect(),
            strategy_sums: (0..n).map(|_| AtomicI64::new(0)).collect(),
            baselines: (0..n).map(|_| AtomicI32::new(0)).collect(),
            predictions: (0..n).map(|_| AtomicI32::new(0)).collect(),
        }
    }
}

impl BlueprintCfrStorage for SparseBlueprintStorage {
    fn bucket_counts(&self) -> [u16; 4] {
        self.bucket_counts
    }

    fn num_actions(&self, node_idx: u32) -> u16 {
        self.layout[node_idx as usize].num_actions
    }

    fn street_idx(&self, node_idx: u32) -> u8 {
        self.layout[node_idx as usize].street_idx
    }

    fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
        let layout = self.layout_or_panic(node_idx, bucket);
        self.get_row(node_idx, bucket, layout)
            .map_or(0, |row| row.regrets[action].load(Ordering::Relaxed))
    }

    fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
        let layout = self.layout_or_panic(node_idx, bucket);
        let (row, _) = self.get_or_insert_row(node_idx, bucket, layout);
        let atom = &row.regrets[action];
        let floor = self.regret_floor.load(Ordering::Relaxed);
        if floor == i32::MIN {
            atom.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                Some(old.saturating_add(delta))
            })
            .ok();
        } else {
            atom.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                Some(old.saturating_add(delta).max(floor))
            })
            .ok();
        }
    }

    fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
        let layout = self.layout_or_panic(node_idx, bucket);
        self.get_row(node_idx, bucket, layout)
            .map_or(0, |row| row.strategy_sums[action].load(Ordering::Relaxed))
    }

    fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64) {
        let layout = self.layout_or_panic(node_idx, bucket);
        let (row, _) = self.get_or_insert_row(node_idx, bucket, layout);
        row.strategy_sums[action].fetch_add(delta, Ordering::Relaxed);
    }

    fn get_baseline(&self, node_idx: u32, bucket: u16, action: usize) -> f64 {
        if !self.baselines_enabled.load(Ordering::Relaxed) {
            return 0.0;
        }
        let layout = self.layout_or_panic(node_idx, bucket);
        self.get_row(node_idx, bucket, layout).map_or(0.0, |row| {
            row.baselines[action].load(Ordering::Relaxed) as f64 / REGRET_SCALE
        })
    }

    fn update_baseline(&self, node_idx: u32, bucket: u16, action: usize, value: f64, alpha: f64) {
        if !self.baselines_enabled.load(Ordering::Relaxed) {
            return;
        }
        let layout = self.layout_or_panic(node_idx, bucket);
        let (row, _) = self.get_or_insert_row(node_idx, bucket, layout);
        let old = row.baselines[action].load(Ordering::Relaxed) as f64 / REGRET_SCALE;
        let new_val = old * (1.0 - alpha) + value * alpha;
        row.baselines[action].store((new_val * REGRET_SCALE) as i32, Ordering::Relaxed);
    }

    fn get_prediction(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
        if !self.predictions_enabled.load(Ordering::Relaxed) {
            return 0;
        }
        let layout = self.layout_or_panic(node_idx, bucket);
        self.get_row(node_idx, bucket, layout)
            .map_or(0, |row| row.predictions[action].load(Ordering::Relaxed))
    }

    fn set_prediction(&self, node_idx: u32, bucket: u16, action: usize, value: i32) {
        if !self.predictions_enabled.load(Ordering::Relaxed)
            || self.predictions_locked.load(Ordering::Relaxed)
        {
            return;
        }
        let layout = self.layout_or_panic(node_idx, bucket);
        let (row, _) = self.get_or_insert_row(node_idx, bucket, layout);
        row.predictions[action].store(value, Ordering::Relaxed);
    }

    fn add_prediction(&self, node_idx: u32, bucket: u16, action: usize, value: i32) {
        if !self.predictions_enabled.load(Ordering::Relaxed)
            || self.predictions_locked.load(Ordering::Relaxed)
        {
            return;
        }
        let layout = self.layout_or_panic(node_idx, bucket);
        let (row, _) = self.get_or_insert_row(node_idx, bucket, layout);
        row.predictions[action].fetch_add(value, Ordering::Relaxed);
    }

    fn current_strategy_into(&self, node_idx: u32, bucket: u16, out: &mut [f64]) {
        let layout = self.layout_or_panic(node_idx, bucket);
        let num_actions = usize::from(layout.num_actions);
        debug_assert!(
            out.len() >= num_actions,
            "buffer too small: {} < {num_actions}",
            out.len()
        );
        let out = &mut out[..num_actions];
        let Some(row) = self.get_row(node_idx, bucket, layout) else {
            out.fill(1.0 / num_actions as f64);
            return;
        };

        if let Some(ref opt) = self.optimizer {
            let predictions = self
                .predictions_enabled
                .load(Ordering::Relaxed)
                .then_some(row.predictions.as_slice());
            opt.current_strategy(&row.regrets, predictions, 0, num_actions, out);
            return;
        }

        let mut positive_sum = 0.0;
        for (action_idx, slot) in out.iter_mut().enumerate() {
            let r = row.regrets[action_idx].load(Ordering::Relaxed).max(0) as f64;
            *slot = r;
            positive_sum += r;
        }
        if positive_sum > 0.0 {
            for slot in out.iter_mut() {
                *slot /= positive_sum;
            }
        } else {
            out.fill(1.0 / num_actions as f64);
        }
    }

    fn average_strategy(&self, node_idx: u32, bucket: u16) -> Vec<f64> {
        let layout = self.layout_or_panic(node_idx, bucket);
        let num_actions = usize::from(layout.num_actions);
        let Some(row) = self.get_row(node_idx, bucket, layout) else {
            return vec![1.0 / num_actions as f64; num_actions];
        };

        let mut sums = Vec::with_capacity(num_actions);
        let mut total = 0.0;
        for action_idx in 0..num_actions {
            let sum = row.strategy_sums[action_idx].load(Ordering::Relaxed) as f64;
            sums.push(sum);
            total += sum;
        }
        if total > 0.0 {
            sums.iter().map(|sum| sum / total).collect()
        } else {
            vec![1.0 / num_actions as f64; num_actions]
        }
    }

    fn storage_stats(&self) -> CfrStorageStats {
        let rows = self.rows.read().expect("sparse storage rows lock");
        let realized_rows = rows.len();
        let realized_slots = self.stats.realized_slots.load(Ordering::Relaxed) as usize;
        CfrStorageStats {
            realized_rows,
            realized_slots,
            inserts: self.stats.inserts.load(Ordering::Relaxed),
            read_probes: self.stats.read_probes.load(Ordering::Relaxed),
            read_hits: self.stats.read_hits.load(Ordering::Relaxed),
            write_probes: self.stats.write_probes.load(Ordering::Relaxed),
            write_hits: self.stats.write_hits.load(Ordering::Relaxed),
            dense_equivalent_slots: self.dense_equivalent_slots,
            dense_equivalent_bytes: self.dense_equivalent_bytes(),
            sparse_resident_bytes: self.sparse_resident_bytes(realized_rows, realized_slots),
        }
    }

    fn project_dense_regrets_and_sums(&self, tree: &GameTree) -> (Vec<i32>, Vec<i64>) {
        self.validate_tree_schema(tree)
            .unwrap_or_else(|err| panic!("{err}"));
        self.project_dense_regrets_and_sums_unchecked(tree)
    }
}

impl SparseBlueprintStorage {
    fn project_dense_regrets_and_sums_unchecked(&self, tree: &GameTree) -> (Vec<i32>, Vec<i64>) {
        let mut regrets = Vec::with_capacity(self.dense_equivalent_slots);
        let mut sums = Vec::with_capacity(self.dense_equivalent_slots);
        let rows = self.rows.read().expect("sparse storage rows lock");

        for (node_idx, node) in tree.nodes.iter().enumerate() {
            let GameNode::Decision {
                street, actions, ..
            } = node
            else {
                continue;
            };
            let layout = self.layout[node_idx];
            for bucket in 0..self.bucket_counts[*street as usize] {
                let key = Self::row_key(node_idx as u32, bucket, layout);
                if let Some(row) = rows.get(&key) {
                    for action_idx in 0..actions.len() {
                        regrets.push(row.regrets[action_idx].load(Ordering::Relaxed));
                        sums.push(row.strategy_sums[action_idx].load(Ordering::Relaxed));
                    }
                } else {
                    regrets.extend(std::iter::repeat_n(0, actions.len()));
                    sums.extend(std::iter::repeat_n(0, actions.len()));
                }
            }
        }

        (regrets, sums)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::game_tree::GameTree;
    use crate::cfr::optimizer::SapcfrPlusOptimizer;
    use std::sync::Arc;

    fn toy_tree() -> GameTree {
        GameTree::build(
            20.0,
            1.0,
            2.0,
            &[vec!["5bb".into()]],
            &[vec![1.0]],
            &[vec![1.0]],
            &[vec![1.0]],
        )
    }

    fn first_decision(tree: &GameTree) -> u32 {
        tree.nodes
            .iter()
            .position(|node| matches!(node, GameNode::Decision { .. }))
            .expect("decision node") as u32
    }

    #[test]
    fn missing_row_uses_uniform_and_zero_without_allocating() {
        let tree = toy_tree();
        let storage = SparseBlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let node_idx = first_decision(&tree);
        let actions = storage.num_actions(node_idx) as usize;

        let current = storage.current_strategy(node_idx, 0);
        let average = storage.average_strategy(node_idx, 0);
        assert_eq!(current, vec![1.0 / actions as f64; actions]);
        assert_eq!(average, vec![1.0 / actions as f64; actions]);
        assert_eq!(storage.get_regret(node_idx, 0, 0), 0);
        assert_eq!(storage.get_strategy_sum(node_idx, 0, 0), 0);
        assert_eq!(storage.get_baseline(node_idx, 0, 0), 0.0);
        assert_eq!(storage.get_prediction(node_idx, 0, 0), 0);
        assert!(!storage.is_realized(node_idx, 0));
        assert_eq!(storage.storage_stats().realized_rows, 0);
    }

    #[test]
    fn row_realization_is_idempotent() {
        let tree = toy_tree();
        let storage = SparseBlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let node_idx = first_decision(&tree);

        assert!(storage.realize_row(node_idx, 0).expect("realize"));
        assert!(!storage.realize_row(node_idx, 0).expect("realize again"));
        let stats = storage.storage_stats();
        assert_eq!(stats.realized_rows, 1);
        assert_eq!(stats.inserts, 1);
        assert_eq!(stats.write_probes, 2);
        assert_eq!(stats.write_hits, 1);
    }

    #[test]
    fn schema_mismatch_is_rejected() {
        let tree = toy_tree();
        let storage = SparseBlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let node_idx = first_decision(&tree);
        let fp = storage.schema_fingerprint(node_idx);
        let actions = storage.num_actions(node_idx);

        let err = storage
            .realize_row_with_schema(node_idx, 0, fp ^ 0x55AA, actions)
            .expect_err("schema mismatch should reject");
        assert!(matches!(err, SparseStorageError::SchemaMismatch { .. }));
        assert!(!storage.is_realized(node_idx, 0));
    }

    #[test]
    fn dense_projection_is_deterministic() {
        let tree = toy_tree();
        let sparse = SparseBlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let dense = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let node_idx = first_decision(&tree);

        sparse.add_regret(node_idx, 0, 0, 100);
        sparse.add_strategy_sum(node_idx, 0, 0, 300);
        sparse.add_strategy_sum(node_idx, 0, 1, 700);

        dense.add_regret(node_idx, 0, 0, 100);
        dense.add_strategy_sum(node_idx, 0, 0, 300);
        dense.add_strategy_sum(node_idx, 0, 1, 700);

        let expected = dense.project_dense_regrets_and_sums(&tree);
        let first = sparse.dense_regrets_and_sums(&tree);
        let second = sparse.dense_regrets_and_sums(&tree);
        assert_eq!(first, expected);
        assert_eq!(second, expected);
    }

    #[test]
    fn prediction_aware_optimizer_matches_dense_strategy() {
        let tree = toy_tree();
        let mut dense = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let mut sparse = SparseBlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let node_idx = first_decision(&tree);
        assert!(dense.num_actions(node_idx) >= 2);

        dense.enable_predictions();
        sparse.enable_predictions();
        dense.set_optimizer(Arc::new(SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 1.0,
        }));
        sparse.set_optimizer(Arc::new(SapcfrPlusOptimizer {
            alpha: 1.5,
            gamma: 2.0,
            eta: 1.0,
        }));

        dense.add_regret(node_idx, 0, 0, 100);
        sparse.add_regret(node_idx, 0, 0, 100);
        dense.set_prediction(node_idx, 0, 1, 200);
        sparse.set_prediction(node_idx, 0, 1, 200);

        assert_eq!(sparse.get_prediction(node_idx, 0, 1), 200);
        assert_eq!(
            dense.current_strategy(node_idx, 0),
            sparse.current_strategy(node_idx, 0)
        );
    }

    #[test]
    fn regret_floor_clamps_sparse_updates() {
        let tree = toy_tree();
        let storage = SparseBlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let node_idx = first_decision(&tree);

        storage.set_regret_floor(-1_000);
        storage.add_regret(node_idx, 0, 0, -5_000);

        assert_eq!(storage.get_regret(node_idx, 0, 0), -1_000);
    }

    #[test]
    fn dense_projection_rejects_mismatched_tree_schema() {
        let tree = toy_tree();
        let storage = SparseBlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let node_idx = first_decision(&tree);
        storage.add_regret(node_idx, 0, 0, 100);

        let mut mismatched = tree.clone();
        let GameNode::Decision { actions, .. } = &mut mismatched.nodes[node_idx as usize] else {
            panic!("expected decision node");
        };
        assert!(actions.len() >= 2);
        actions.swap(0, 1);

        let err = storage
            .try_dense_regrets_and_sums(&mismatched)
            .expect_err("projection should reject action schema mismatch");
        assert!(matches!(err, SparseStorageError::SchemaMismatch { .. }));
    }
}
