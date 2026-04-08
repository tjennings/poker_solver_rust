//! Flat-buffer storage for regrets and strategy sums, indexed by
//! (decision node, bucket, action).
//!
//! Mirrors the design of `blueprint_v2::storage` but simplified for
//! the multiplayer blueprint solver. Each `Decision` node stores
//! `bucket_count * num_actions` entries contiguously.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::atomic::{AtomicI32, AtomicI64, Ordering};

use super::game_tree::{MpGameNode, MpGameTree};

/// Fixed-point scaling factor for regret values.
///
/// Regret deltas are stored as `(chip_value * REGRET_SCALE) as i32`.
pub const REGRET_SCALE: f64 = 1_000.0;

/// Flat-buffer storage for regrets and strategy sums.
pub struct MpStorage {
    /// Cumulative regrets: one `AtomicI32` per (node, bucket, action).
    pub regrets: Vec<AtomicI32>,
    /// Strategy sums: one `AtomicI64` per (node, bucket, action).
    pub strategy_sums: Vec<AtomicI64>,
    /// Number of buckets per street `[preflop, flop, turn, river]`.
    pub bucket_counts: [u16; 4],
    /// Per-node layout metadata.
    layout: Vec<NodeLayout>,
    /// Floor for cumulative regret. `i32::MIN` = no floor (default).
    pub regret_floor: i32,
}

#[derive(Clone, Copy, Default)]
struct NodeLayout {
    offset: usize,
    num_actions: u16,
    _street_idx: u8,
}

impl MpStorage {
    /// Build storage for a given tree and per-street bucket counts.
    #[must_use]
    pub fn new(tree: &MpGameTree, bucket_counts: [u16; 4]) -> Self {
        let (layout, total) = build_layout(&tree.nodes, bucket_counts);
        Self {
            regrets: (0..total).map(|_| AtomicI32::new(0)).collect(),
            strategy_sums: (0..total).map(|_| AtomicI64::new(0)).collect(),
            bucket_counts,
            layout,
            regret_floor: i32::MIN,
        }
    }

    /// Number of actions at a decision node.
    #[inline]
    #[must_use]
    pub fn num_actions(&self, node_idx: u32) -> usize {
        self.layout[node_idx as usize].num_actions as usize
    }

    /// Read a single regret value atomically (raw scaled i32).
    #[inline]
    #[must_use]
    pub fn get_regret(&self, node_idx: u32, bucket: u16, action: usize) -> i32 {
        self.regrets[self.slot(node_idx, bucket, action)].load(Ordering::Relaxed)
    }

    /// Add a delta to a single regret value atomically with saturation.
    #[inline]
    pub fn add_regret(&self, node_idx: u32, bucket: u16, action: usize, delta: i32) {
        let idx = self.slot(node_idx, bucket, action);
        let floor = self.regret_floor;
        self.regrets[idx]
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
                let val = old.saturating_add(delta);
                Some(if floor == i32::MIN { val } else { val.max(floor) })
            })
            .ok();
    }

    /// Read a single strategy sum value atomically.
    #[inline]
    #[must_use]
    pub fn get_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize) -> i64 {
        self.strategy_sums[self.slot(node_idx, bucket, action)].load(Ordering::Relaxed)
    }

    /// Add a delta to a single strategy sum value atomically.
    #[inline]
    pub fn add_strategy_sum(&self, node_idx: u32, bucket: u16, action: usize, delta: i64) {
        self.strategy_sums[self.slot(node_idx, bucket, action)]
            .fetch_add(delta, Ordering::Relaxed);
    }

    /// Current strategy via regret matching, written into `out`.
    pub fn regret_matched_strategy(
        &self,
        node_idx: u32,
        bucket: u16,
        num_actions: usize,
        out: &mut [f64],
    ) {
        let start = self.slot(node_idx, bucket, 0);
        let mut positive_sum = 0.0_f64;
        for (i, slot) in out[..num_actions].iter_mut().enumerate() {
            let r = f64::from(self.regrets[start + i].load(Ordering::Relaxed).max(0));
            *slot = r;
            positive_sum += r;
        }
        normalize_or_uniform(out, num_actions, positive_sum);
    }

    /// Average strategy from strategy sums, written into `out`.
    pub fn average_strategy(
        &self,
        node_idx: u32,
        bucket: u16,
        num_actions: usize,
        out: &mut [f64],
    ) {
        let start = self.slot(node_idx, bucket, 0);
        let mut total = 0.0_f64;
        for (i, slot) in out[..num_actions].iter_mut().enumerate() {
            let s = self.strategy_sums[start + i].load(Ordering::Relaxed) as f64;
            *slot = s;
            total += s;
        }
        normalize_or_uniform(out, num_actions, total);
    }

    /// Flat-buffer index for (node, bucket, action).
    #[inline]
    fn slot(&self, node_idx: u32, bucket: u16, action: usize) -> usize {
        let nl = &self.layout[node_idx as usize];
        nl.offset + (bucket as usize) * (nl.num_actions as usize) + action
    }
}

/// Build layout metadata from tree nodes, returning `(layout, total_slots)`.
fn build_layout(nodes: &[MpGameNode], bucket_counts: [u16; 4]) -> (Vec<NodeLayout>, usize) {
    let mut layout = vec![NodeLayout::default(); nodes.len()];
    let mut total: usize = 0;
    for (i, node) in nodes.iter().enumerate() {
        if let MpGameNode::Decision {
            street, actions, ..
        } = node
        {
            let street_idx = street.index() as u8;
            let buckets = bucket_counts[street_idx as usize] as usize;
            let num_actions = actions.len();
            layout[i] = NodeLayout {
                offset: total,
                num_actions: num_actions as u16,
                _street_idx: street_idx,
            };
            total += buckets * num_actions;
        }
    }
    (layout, total)
}

/// Normalize `out[..n]` by `sum`, or fill with uniform if sum is zero.
fn normalize_or_uniform(out: &mut [f64], n: usize, sum: f64) {
    if sum > 0.0 {
        for o in &mut out[..n] {
            *o /= sum;
        }
    } else {
        out[..n].fill(1.0 / n as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_mp::config::{
        ForcedBet, ForcedBetKind, MpActionAbstractionConfig, MpGameConfig, MpStreetSizes,
    };
    use crate::blueprint_mp::game_tree::MpGameTree;
    use test_macros::timed_test;

    fn minimal_tree(num_players: u8) -> MpGameTree {
        let blinds = vec![
            ForcedBet {
                seat: 0,
                kind: ForcedBetKind::SmallBlind,
                amount: 1.0,
            },
            ForcedBet {
                seat: 1,
                kind: ForcedBetKind::BigBlind,
                amount: 2.0,
            },
        ];
        let game = MpGameConfig {
            name: format!("{num_players}-player storage test"),
            num_players,
            stack_depth: 20.0,
            blinds,
            rake_rate: 0.0,
            rake_cap: 0.0,
        };
        let empty = MpStreetSizes {
            lead: vec![],
            raise: vec![],
        };
        let action = MpActionAbstractionConfig {
            preflop: empty.clone(),
            flop: empty.clone(),
            turn: empty.clone(),
            river: empty,
        };
        MpGameTree::build(&game, &action)
    }

    fn first_decision_node(tree: &MpGameTree) -> u32 {
        tree.nodes
            .iter()
            .position(|n| {
                matches!(
                    n,
                    crate::blueprint_mp::game_tree::MpGameNode::Decision { .. }
                )
            })
            .expect("tree should have a decision node") as u32
    }

    #[timed_test]
    fn storage_regret_round_trip() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);

        assert_eq!(storage.get_regret(node, 0, 0), 0);
        storage.add_regret(node, 0, 0, 42);
        assert_eq!(storage.get_regret(node, 0, 0), 42);
        storage.add_regret(node, 0, 0, -10);
        assert_eq!(storage.get_regret(node, 0, 0), 32);
    }

    #[timed_test]
    fn storage_strategy_sum_round_trip() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);

        assert_eq!(storage.get_strategy_sum(node, 0, 0), 0);
        storage.add_strategy_sum(node, 0, 0, 999);
        assert_eq!(storage.get_strategy_sum(node, 0, 0), 999);
        storage.add_strategy_sum(node, 0, 0, 1);
        assert_eq!(storage.get_strategy_sum(node, 0, 0), 1000);
    }

    #[timed_test]
    fn regret_matched_strategy_uniform_when_zero() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);
        let num_actions = storage.num_actions(node);

        let mut out = vec![0.0; num_actions];
        storage.regret_matched_strategy(node, 0, num_actions, &mut out);

        let expected = 1.0 / num_actions as f64;
        for &p in &out {
            assert!(
                (p - expected).abs() < 1e-10,
                "expected uniform {expected}, got {p}"
            );
        }
    }

    #[timed_test]
    fn regret_matched_strategy_positive_only() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);
        let num_actions = storage.num_actions(node);

        // Set action 0 to positive, action 1 to negative
        storage.add_regret(node, 0, 0, 1000);
        if num_actions >= 2 {
            storage.add_regret(node, 0, 1, -500);
        }

        let mut out = vec![0.0; num_actions];
        storage.regret_matched_strategy(node, 0, num_actions, &mut out);

        assert!(
            (out[0] - 1.0).abs() < 1e-10,
            "positive-only action should get all weight, got {}",
            out[0]
        );
        if num_actions >= 2 {
            assert!(
                out[1].abs() < 1e-10,
                "negative regret should get zero weight, got {}",
                out[1]
            );
        }
    }

    #[timed_test]
    fn average_strategy_uniform_when_zero() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);
        let num_actions = storage.num_actions(node);

        let mut out = vec![0.0; num_actions];
        storage.average_strategy(node, 0, num_actions, &mut out);

        let expected = 1.0 / num_actions as f64;
        for &p in &out {
            assert!(
                (p - expected).abs() < 1e-10,
                "expected uniform {expected}, got {p}"
            );
        }
    }

    #[timed_test]
    fn average_strategy_proportional() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);
        let num_actions = storage.num_actions(node);

        if num_actions >= 2 {
            storage.add_strategy_sum(node, 0, 0, 300);
            storage.add_strategy_sum(node, 0, 1, 700);
        }

        let mut out = vec![0.0; num_actions];
        storage.average_strategy(node, 0, num_actions, &mut out);

        if num_actions >= 2 {
            assert!((out[0] - 0.3).abs() < 1e-10);
            assert!((out[1] - 0.7).abs() < 1e-10);
        }
    }

    #[timed_test]
    fn different_buckets_are_independent() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);

        storage.add_regret(node, 0, 0, 100);
        storage.add_regret(node, 1, 0, 200);

        assert_eq!(storage.get_regret(node, 0, 0), 100);
        assert_eq!(storage.get_regret(node, 1, 0), 200);
    }

    #[timed_test]
    fn storage_buffers_equal_length() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        assert!(!storage.regrets.is_empty());
        assert_eq!(storage.regrets.len(), storage.strategy_sums.len());
    }

    #[timed_test]
    fn regret_saturates_at_i32_max() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);

        storage.add_regret(node, 0, 0, i32::MAX - 10);
        storage.add_regret(node, 0, 0, 100);
        assert_eq!(storage.get_regret(node, 0, 0), i32::MAX);
    }

    #[timed_test]
    fn regret_floor_clamps() {
        let tree = minimal_tree(2);
        let mut storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        storage.regret_floor = -1000;
        let node = first_decision_node(&tree);

        storage.add_regret(node, 0, 0, -5000);
        assert_eq!(storage.get_regret(node, 0, 0), -1000);
    }

    #[timed_test]
    fn strategy_sum_holds_large_i64() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);

        let large: i64 = 3_000_000_000;
        storage.add_strategy_sum(node, 0, 0, large);
        assert_eq!(storage.get_strategy_sum(node, 0, 0), large);
    }

    #[timed_test]
    fn regret_matched_strategy_two_positive() {
        let tree = minimal_tree(2);
        let storage = MpStorage::new(&tree, [10, 10, 10, 10]);
        let node = first_decision_node(&tree);
        let num_actions = storage.num_actions(node);

        if num_actions >= 2 {
            storage.add_regret(node, 0, 0, 300);
            storage.add_regret(node, 0, 1, 700);
        }

        let mut out = vec![0.0; num_actions];
        storage.regret_matched_strategy(node, 0, num_actions, &mut out);

        if num_actions >= 2 {
            assert!((out[0] - 0.3).abs() < 1e-10);
            assert!((out[1] - 0.7).abs() < 1e-10);
        }
    }
}
