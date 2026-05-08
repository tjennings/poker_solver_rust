//! Sparse visited-infoset storage for lazy multiplayer blueprint training.
//!
//! Unlike [`super::storage::MpStorage`], this backend does not allocate a
//! `(public-node, bucket, action)` slab for the full eager tree. Entries are
//! allocated only after an infoset is touched, while missing entries retain the
//! CFR semantics of zero regret and uniform strategy.

#![allow(clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicI32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::{MAX_PLAYERS, Seat, Street};

const DEFAULT_SHARDS: usize = 4096;
pub const SPARSE_INSERT_STREET_COUNT: usize = 4;
pub const SPARSE_INSERT_SPR_BUCKET_COUNT: usize = 32;
pub const SPARSE_INSERT_HISTORY_LEN_BIN_COUNT: usize = 8;
pub const SPARSE_INSERT_ACTION_COUNT_BIN_COUNT: usize = 8;

/// Stable key for one lazy MP infoset.
///
/// `history_hi`/`history_lo` carry the compact action history when it fits in
/// 128 bits. `history_hash` keeps keys stable for longer histories, while
/// `history_len` distinguishes equal packed prefixes of different lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MpInfosetKey {
    pub seat: u8,
    pub street: u8,
    pub bucket: u16,
    pub spr_bucket: u8,
    pub history_hi: u64,
    pub history_lo: u64,
    pub history_hash: u64,
    pub history_len: u16,
}

impl MpInfosetKey {
    /// Create a key from typed MP components.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub const fn from_parts(
        seat: Seat,
        street: Street,
        bucket: u16,
        spr_bucket: u8,
        history_hi: u64,
        history_lo: u64,
        history_hash: u64,
        history_len: u16,
    ) -> Self {
        Self {
            seat: seat.index(),
            street: street as u8,
            bucket,
            spr_bucket,
            history_hi,
            history_lo,
            history_hash,
            history_len,
        }
    }

    /// Return the seat component.
    #[must_use]
    pub const fn seat(self) -> Seat {
        Seat::from_raw(self.seat)
    }

    /// Return the street component when the serialized byte is valid.
    #[must_use]
    pub const fn street(self) -> Option<Street> {
        Street::from_u8(self.street)
    }
}

/// Coarse storage accounting for sparse MP storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SparseStorageStats {
    pub entries: usize,
    pub regret_slots: usize,
    pub strategy_slots: usize,
    pub approx_bytes: usize,
    pub shard_count: usize,
    pub nonempty_shards: usize,
    pub max_entries_per_shard: usize,
}

/// Cumulative sparse storage activity counters for throughput diagnostics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseStorageActivity {
    pub read_probes: u64,
    pub read_hits: u64,
    pub write_probes: u64,
    pub write_hits: u64,
    pub inserts: u64,
}

/// Cumulative attribution for newly inserted sparse infosets.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SparseInsertAttribution {
    pub by_street: [u64; SPARSE_INSERT_STREET_COUNT],
    pub by_seat: [u64; MAX_PLAYERS],
    pub by_spr_bucket: [u64; SPARSE_INSERT_SPR_BUCKET_COUNT],
    pub history_len_bins: [u64; SPARSE_INSERT_HISTORY_LEN_BIN_COUNT],
    pub action_count_bins: [u64; SPARSE_INSERT_ACTION_COUNT_BIN_COUNT],
    pub history_len_max: u64,
    pub action_count_sum: u64,
    pub action_count_max: u64,
}

/// Regret and average-strategy sample for sparse telemetry.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseTelemetrySample {
    pub entries_sampled: usize,
    pub regret_slots_sampled: usize,
    pub max_positive_regret: i32,
    pub max_negative_regret: i32,
    pub avg_positive_regret: f64,
    pub strategy_fingerprint: f64,
}

/// Serializable snapshot of one visited sparse infoset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SparseSnapshotEntry {
    pub key: MpInfosetKey,
    pub num_actions: u8,
    pub regrets: Vec<i32>,
    pub strategy_sums: Vec<u64>,
}

struct SparseNode {
    num_actions: u8,
    regrets: Box<[AtomicI32]>,
    strategy_sums: Box<[AtomicU64]>,
}

impl SparseNode {
    fn new(num_actions: usize) -> Self {
        let num_actions_u8 =
            u8::try_from(num_actions).expect("sparse infoset supports at most 255 actions");
        let regrets = (0..num_actions)
            .map(|_| AtomicI32::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let strategy_sums = (0..num_actions)
            .map(|_| AtomicU64::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            num_actions: num_actions_u8,
            regrets,
            strategy_sums,
        }
    }

    fn action_count(&self) -> usize {
        self.num_actions as usize
    }
}

#[derive(Default)]
struct Shard {
    entries: Mutex<HashMap<MpInfosetKey, Arc<SparseNode>>>,
}

/// Thread-safe sparse regret and average-strategy storage for lazy MP CFR.
pub struct SparseMpStorage {
    shards: Box<[Shard]>,
    shard_entry_counts: Box<[AtomicUsize]>,
    entry_count: AtomicUsize,
    slot_count: AtomicUsize,
    read_probes: AtomicU64,
    read_hits: AtomicU64,
    write_probes: AtomicU64,
    write_hits: AtomicU64,
    inserts: AtomicU64,
    insert_by_street: [AtomicU64; SPARSE_INSERT_STREET_COUNT],
    insert_by_seat: [AtomicU64; MAX_PLAYERS],
    insert_by_spr_bucket: [AtomicU64; SPARSE_INSERT_SPR_BUCKET_COUNT],
    insert_history_len_bins: [AtomicU64; SPARSE_INSERT_HISTORY_LEN_BIN_COUNT],
    insert_action_count_bins: [AtomicU64; SPARSE_INSERT_ACTION_COUNT_BIN_COUNT],
    insert_history_len_max: AtomicU64,
    insert_action_count_sum: AtomicU64,
    insert_action_count_max: AtomicU64,
    regret_floor: AtomicI32,
}

impl Default for SparseMpStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseMpStorage {
    /// Build storage with the default shard count.
    #[must_use]
    pub fn new() -> Self {
        Self::with_shards(DEFAULT_SHARDS)
    }

    /// Build storage with a caller-selected shard count.
    #[must_use]
    pub fn with_shards(num_shards: usize) -> Self {
        let shard_count = num_shards.max(1);
        let shards = (0..shard_count)
            .map(|_| Shard::default())
            .collect::<Vec<_>>()
            .into_boxed_slice();
        let shard_entry_counts = (0..shard_count)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Self {
            shards,
            shard_entry_counts,
            entry_count: AtomicUsize::new(0),
            slot_count: AtomicUsize::new(0),
            read_probes: AtomicU64::new(0),
            read_hits: AtomicU64::new(0),
            write_probes: AtomicU64::new(0),
            write_hits: AtomicU64::new(0),
            inserts: AtomicU64::new(0),
            insert_by_street: std::array::from_fn(|_| AtomicU64::new(0)),
            insert_by_seat: std::array::from_fn(|_| AtomicU64::new(0)),
            insert_by_spr_bucket: std::array::from_fn(|_| AtomicU64::new(0)),
            insert_history_len_bins: std::array::from_fn(|_| AtomicU64::new(0)),
            insert_action_count_bins: std::array::from_fn(|_| AtomicU64::new(0)),
            insert_history_len_max: AtomicU64::new(0),
            insert_action_count_sum: AtomicU64::new(0),
            insert_action_count_max: AtomicU64::new(0),
            regret_floor: AtomicI32::new(i32::MIN),
        }
    }

    /// Build storage from sparse snapshot entries.
    #[must_use]
    pub fn from_snapshot_entries(entries: impl IntoIterator<Item = SparseSnapshotEntry>) -> Self {
        let storage = Self::new();
        storage.load_snapshot_entries(entries);
        storage
    }

    /// Set a lower bound for cumulative regret updates.
    pub fn set_regret_floor(&self, floor: i32) {
        self.regret_floor.store(floor, Ordering::Relaxed);
    }

    /// Number of visited infosets currently allocated.
    #[must_use]
    pub fn entry_count(&self) -> usize {
        self.entry_count.load(Ordering::Relaxed)
    }

    /// Storage accounting for allocated sparse entries.
    #[must_use]
    pub fn stats(&self) -> SparseStorageStats {
        let mut nonempty_shards = 0;
        let mut max_entries_per_shard = 0;
        for shard_entries in &self.shard_entry_counts {
            let shard_entries = shard_entries.load(Ordering::Relaxed);
            if shard_entries > 0 {
                nonempty_shards += 1;
            }
            max_entries_per_shard = max_entries_per_shard.max(shard_entries);
        }
        let entries = self.entry_count.load(Ordering::Relaxed);
        let slots = self.slot_count.load(Ordering::Relaxed);
        let approx_bytes = self
            .shards
            .len()
            .saturating_mul(std::mem::size_of::<Shard>())
            .saturating_add(
                self.shard_entry_counts
                    .len()
                    .saturating_mul(std::mem::size_of::<AtomicUsize>()),
            )
            .saturating_add(
                std::mem::size_of::<AtomicUsize>()
                    .saturating_mul(2)
                    .saturating_add(std::mem::size_of::<AtomicI32>()),
            )
            .saturating_add(entries.saturating_mul(
                std::mem::size_of::<MpInfosetKey>() + std::mem::size_of::<Arc<SparseNode>>(),
            ))
            .saturating_add(slots.saturating_mul(
                std::mem::size_of::<AtomicI32>() + std::mem::size_of::<AtomicU64>(),
            ));
        SparseStorageStats {
            entries,
            regret_slots: slots,
            strategy_slots: slots,
            approx_bytes,
            shard_count: self.shards.len(),
            nonempty_shards,
            max_entries_per_shard,
        }
    }

    /// Cumulative storage access counters for training telemetry.
    #[must_use]
    pub fn activity(&self) -> SparseStorageActivity {
        SparseStorageActivity {
            read_probes: self.read_probes.load(Ordering::Relaxed),
            read_hits: self.read_hits.load(Ordering::Relaxed),
            write_probes: self.write_probes.load(Ordering::Relaxed),
            write_hits: self.write_hits.load(Ordering::Relaxed),
            inserts: self.inserts.load(Ordering::Relaxed),
        }
    }

    /// Cumulative attribution for newly allocated sparse infosets.
    #[must_use]
    pub fn insert_attribution(&self) -> SparseInsertAttribution {
        SparseInsertAttribution {
            by_street: load_atomic_array(&self.insert_by_street),
            by_seat: load_atomic_array(&self.insert_by_seat),
            by_spr_bucket: load_atomic_array(&self.insert_by_spr_bucket),
            history_len_bins: load_atomic_array(&self.insert_history_len_bins),
            action_count_bins: load_atomic_array(&self.insert_action_count_bins),
            history_len_max: self.insert_history_len_max.load(Ordering::Relaxed),
            action_count_sum: self.insert_action_count_sum.load(Ordering::Relaxed),
            action_count_max: self.insert_action_count_max.load(Ordering::Relaxed),
        }
    }

    /// Sample sparse entries for TUI telemetry without scanning the whole store.
    ///
    /// The sample walks shards in order and takes a bounded number of entries
    /// per shard. It is intended for trend telemetry, not exact reporting.
    #[must_use]
    pub fn telemetry_sample(&self, max_entries: usize) -> SparseTelemetrySample {
        if max_entries == 0 {
            return SparseTelemetrySample::default();
        }
        let per_shard = max_entries.div_ceil(self.shards.len()).max(1);
        let mut entries_sampled = 0_usize;
        let mut regret_slots_sampled = 0_usize;
        let mut max_positive_regret = 0_i32;
        let mut max_negative_regret = 0_i32;
        let mut positive_sum = 0_i64;
        let mut positive_count = 0_u64;
        let mut strategy_fingerprint = 0.0_f64;

        for shard in &self.shards {
            if entries_sampled >= max_entries {
                break;
            }
            let guard = lock_entries(shard);
            let remaining = max_entries - entries_sampled;
            for (key, node) in guard.iter().take(per_shard.min(remaining)) {
                entries_sampled += 1;
                regret_slots_sampled += node.action_count();
                let mut strategy_total = 0_u64;
                for (action_idx, regret) in node.regrets.iter().enumerate() {
                    let value = regret.load(Ordering::Relaxed);
                    max_positive_regret = max_positive_regret.max(value);
                    max_negative_regret = max_negative_regret.min(value);
                    if value > 0 {
                        positive_sum += i64::from(value);
                        positive_count += 1;
                    }
                    let strategy_sum = node.strategy_sums[action_idx].load(Ordering::Relaxed);
                    strategy_total = strategy_total.saturating_add(strategy_sum);
                }
                if strategy_total > 0 {
                    let key_weight = sparse_key_fingerprint_weight(*key);
                    for (action_idx, strategy_sum) in node.strategy_sums.iter().enumerate() {
                        let prob =
                            strategy_sum.load(Ordering::Relaxed) as f64 / strategy_total as f64;
                        strategy_fingerprint += key_weight * (action_idx as f64 + 1.0) * prob;
                    }
                }
            }
        }

        let avg_positive_regret = if positive_count > 0 {
            positive_sum as f64 / positive_count as f64
        } else {
            0.0
        };

        SparseTelemetrySample {
            entries_sampled,
            regret_slots_sampled,
            max_positive_regret,
            max_negative_regret,
            avg_positive_regret,
            strategy_fingerprint,
        }
    }

    /// Read a regret value. Missing entries or out-of-range actions return zero.
    #[must_use]
    pub fn get_regret(&self, key: MpInfosetKey, action: usize) -> i32 {
        let Some(node) = self.get_node(key) else {
            return 0;
        };
        node.regrets
            .get(action)
            .map_or(0, |atom| atom.load(Ordering::Relaxed))
    }

    /// Read a strategy-sum value. Missing entries or out-of-range actions return zero.
    #[must_use]
    pub fn get_strategy_sum(&self, key: MpInfosetKey, action: usize) -> u64 {
        let Some(node) = self.get_node(key) else {
            return 0;
        };
        node.strategy_sums
            .get(action)
            .map_or(0, |atom| atom.load(Ordering::Relaxed))
    }

    /// Add a regret delta, allocating the infoset if needed.
    pub fn add_regret(&self, key: MpInfosetKey, num_actions: usize, action: usize, delta: i32) {
        if action >= num_actions {
            return;
        }
        let Some(node) = self.get_or_create_node(key, num_actions) else {
            return;
        };
        let Some(atom) = node.regrets.get(action) else {
            return;
        };
        let floor = self.regret_floor.load(Ordering::Relaxed);
        atom.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
            let value = old.saturating_add(delta);
            Some(if floor == i32::MIN {
                value
            } else {
                value.max(floor)
            })
        })
        .ok();
    }

    /// Add a non-negative strategy-sum delta, allocating the infoset if needed.
    pub fn add_strategy_sum(
        &self,
        key: MpInfosetKey,
        num_actions: usize,
        action: usize,
        delta: i32,
    ) {
        if action >= num_actions {
            return;
        }
        let Ok(delta) = u64::try_from(delta) else {
            return;
        };
        let Some(node) = self.get_or_create_node(key, num_actions) else {
            return;
        };
        let Some(atom) = node.strategy_sums.get(action) else {
            return;
        };
        atom.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |old| {
            Some(old.saturating_add(delta))
        })
        .ok();
    }

    /// Current strategy via regret matching. Missing entries are uniform.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `num_actions`.
    pub fn regret_matched_strategy(&self, key: MpInfosetKey, num_actions: usize, out: &mut [f64]) {
        assert!(
            out.len() >= num_actions,
            "strategy output shorter than action count"
        );
        let Some(node) = self.get_node(key) else {
            fill_uniform(out, num_actions);
            return;
        };
        let mut positive_sum = 0.0;
        for (i, slot) in out[..num_actions].iter_mut().enumerate() {
            let raw = node
                .regrets
                .get(i)
                .map_or(0, |atom| atom.load(Ordering::Relaxed));
            let regret = f64::from(raw.max(0));
            *slot = regret;
            positive_sum += regret;
        }
        normalize_or_uniform(out, num_actions, positive_sum);
    }

    /// Average strategy from strategy sums. Missing or zero-sum entries are uniform.
    ///
    /// # Panics
    ///
    /// Panics if `out` is shorter than `num_actions`.
    pub fn average_strategy(&self, key: MpInfosetKey, num_actions: usize, out: &mut [f64]) {
        assert!(
            out.len() >= num_actions,
            "strategy output shorter than action count"
        );
        let Some(node) = self.get_node(key) else {
            fill_uniform(out, num_actions);
            return;
        };
        let mut total = 0.0;
        for (i, slot) in out[..num_actions].iter_mut().enumerate() {
            let sum = node
                .strategy_sums
                .get(i)
                .map_or(0, |atom| atom.load(Ordering::Relaxed));
            *slot = sum as f64;
            total += *slot;
        }
        normalize_or_uniform(out, num_actions, total);
    }

    /// Apply DCFR/LCFR discounts to visited entries only.
    pub fn discount(&self, d_pos: f64, d_neg: f64, d_strat: f64) {
        self.shards.par_iter().for_each(|shard| {
            let guard = lock_entries(shard);
            for node in guard.values() {
                for regret in &node.regrets {
                    let value = regret.load(Ordering::Relaxed);
                    let factor = if value >= 0 { d_pos } else { d_neg };
                    regret.store(discount_i32(value, factor), Ordering::Relaxed);
                }
                for strategy_sum in &node.strategy_sums {
                    let value = strategy_sum.load(Ordering::Relaxed);
                    strategy_sum.store(discount_u64(value, d_strat), Ordering::Relaxed);
                }
            }
        });
    }

    /// Snapshot all visited entries in deterministic key order.
    #[must_use]
    pub fn snapshot_entries(&self) -> Vec<SparseSnapshotEntry> {
        let mut entries = Vec::with_capacity(self.entry_count());
        for shard in &self.shards {
            let guard = lock_entries(shard);
            for (&key, node) in guard.iter() {
                entries.push(SparseSnapshotEntry {
                    key,
                    num_actions: node.num_actions,
                    regrets: node
                        .regrets
                        .iter()
                        .map(|atom| atom.load(Ordering::Relaxed))
                        .collect(),
                    strategy_sums: node
                        .strategy_sums
                        .iter()
                        .map(|atom| atom.load(Ordering::Relaxed))
                        .collect(),
                });
            }
        }
        entries.sort_unstable_by_key(|entry| entry.key);
        entries
    }

    /// Load sparse snapshot entries, replacing values for matching visited keys.
    pub fn load_snapshot_entries(&self, entries: impl IntoIterator<Item = SparseSnapshotEntry>) {
        for entry in entries {
            let num_actions = usize::from(entry.num_actions);
            let Some(node) = self.get_or_create_node(entry.key, num_actions) else {
                continue;
            };
            for (idx, regret) in node.regrets.iter().enumerate() {
                regret.store(*entry.regrets.get(idx).unwrap_or(&0), Ordering::Relaxed);
            }
            for (idx, strategy_sum) in node.strategy_sums.iter().enumerate() {
                strategy_sum.store(
                    *entry.strategy_sums.get(idx).unwrap_or(&0),
                    Ordering::Relaxed,
                );
            }
        }
    }

    fn get_node(&self, key: MpInfosetKey) -> Option<Arc<SparseNode>> {
        let shard = &self.shards[self.shard_index_for(key)];
        let guard = lock_entries(shard);
        self.read_probes.fetch_add(1, Ordering::Relaxed);
        let node = guard.get(&key).cloned();
        if node.is_some() {
            self.read_hits.fetch_add(1, Ordering::Relaxed);
        }
        node
    }

    fn get_or_create_node(&self, key: MpInfosetKey, num_actions: usize) -> Option<Arc<SparseNode>> {
        if num_actions == 0 {
            return None;
        }
        let shard_idx = self.shard_index_for(key);
        let shard = &self.shards[shard_idx];
        let mut guard = lock_entries(shard);
        self.write_probes.fetch_add(1, Ordering::Relaxed);
        if let Some(node) = guard.get(&key) {
            debug_assert_eq!(
                node.action_count(),
                num_actions,
                "same sparse infoset key used with different action counts"
            );
            self.write_hits.fetch_add(1, Ordering::Relaxed);
            return Some(Arc::clone(node));
        }
        let node = Arc::new(SparseNode::new(num_actions));
        guard.insert(key, Arc::clone(&node));
        self.entry_count.fetch_add(1, Ordering::Relaxed);
        self.slot_count.fetch_add(num_actions, Ordering::Relaxed);
        self.shard_entry_counts[shard_idx].fetch_add(1, Ordering::Relaxed);
        self.inserts.fetch_add(1, Ordering::Relaxed);
        self.record_insert(key, num_actions);
        Some(node)
    }

    fn record_insert(&self, key: MpInfosetKey, num_actions: usize) {
        if let Some(counter) = self.insert_by_street.get(usize::from(key.street)) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        if let Some(counter) = self.insert_by_seat.get(usize::from(key.seat)) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
        let spr_idx = usize::from(key.spr_bucket).min(SPARSE_INSERT_SPR_BUCKET_COUNT - 1);
        self.insert_by_spr_bucket[spr_idx].fetch_add(1, Ordering::Relaxed);

        let history_len = u64::from(key.history_len);
        self.insert_history_len_bins[history_len_bin(key.history_len)]
            .fetch_add(1, Ordering::Relaxed);
        self.insert_history_len_max
            .fetch_max(history_len, Ordering::Relaxed);

        let action_count = u64::try_from(num_actions).unwrap_or(u64::MAX);
        self.insert_action_count_bins[action_count_bin(num_actions)]
            .fetch_add(1, Ordering::Relaxed);
        self.insert_action_count_sum
            .fetch_add(action_count, Ordering::Relaxed);
        self.insert_action_count_max
            .fetch_max(action_count, Ordering::Relaxed);
    }

    fn shard_index_for(&self, key: MpInfosetKey) -> usize {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        let shard_count = u64::try_from(self.shards.len()).expect("shard count fits in u64");
        usize::try_from(hasher.finish() % shard_count)
            .expect("bounded sparse shard index fits in usize")
    }
}

fn load_atomic_array<const N: usize>(atoms: &[AtomicU64; N]) -> [u64; N] {
    std::array::from_fn(|idx| atoms[idx].load(Ordering::Relaxed))
}

fn history_len_bin(history_len: u16) -> usize {
    match history_len {
        0 => 0,
        1 => 1,
        2..=3 => 2,
        4..=7 => 3,
        8..=15 => 4,
        16..=31 => 5,
        32..=63 => 6,
        _ => 7,
    }
}

fn action_count_bin(action_count: usize) -> usize {
    match action_count {
        0 | 1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        33..=64 => 6,
        _ => 7,
    }
}

fn sparse_key_fingerprint_weight(key: MpInfosetKey) -> f64 {
    let mixed = key
        .history_hash
        .wrapping_add(key.history_lo.rotate_left(17))
        .wrapping_add(key.history_hi.rotate_left(31))
        .wrapping_add(u64::from(key.bucket) << 8)
        .wrapping_add(u64::from(key.seat) << 3)
        .wrapping_add(u64::from(key.street));
    (mixed % 1_000_003) as f64 / 1_000_003.0
}

fn lock_entries(shard: &Shard) -> MutexGuard<'_, HashMap<MpInfosetKey, Arc<SparseNode>>> {
    shard
        .entries
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn normalize_or_uniform(out: &mut [f64], n: usize, sum: f64) {
    if n == 0 {
        return;
    }
    if sum > 0.0 {
        for value in &mut out[..n] {
            *value /= sum;
        }
    } else {
        fill_uniform(out, n);
    }
}

fn fill_uniform(out: &mut [f64], n: usize) {
    if n == 0 {
        return;
    }
    out[..n].fill(1.0 / n as f64);
}

#[allow(clippy::cast_possible_truncation)]
fn discount_i32(value: i32, factor: f64) -> i32 {
    (f64::from(value) * factor)
        .round()
        .clamp(f64::from(i32::MIN), f64::from(i32::MAX)) as i32
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn discount_u64(value: u64, factor: f64) -> u64 {
    ((value as f64) * factor).clamp(0.0, u64::MAX as f64) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    fn key(bucket: u16) -> MpInfosetKey {
        MpInfosetKey::from_parts(
            Seat::from_raw(2),
            Street::Flop,
            bucket,
            3,
            0xAA,
            0xBB,
            0xCC,
            4,
        )
    }

    fn assert_strategy_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual_value, expected_value) in actual.iter().zip(expected) {
            assert!(
                (actual_value - expected_value).abs() < 1e-12,
                "strategy mismatch: actual={actual:?} expected={expected:?}"
            );
        }
    }

    #[timed_test]
    fn missing_entries_read_as_zero_and_uniform() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(7);

        assert_eq!(storage.get_regret(k, 0), 0);
        assert_eq!(storage.get_strategy_sum(k, 1), 0);

        let mut strategy = [0.0; 3];
        storage.regret_matched_strategy(k, 3, &mut strategy);
        assert_strategy_close(&strategy, &[1.0 / 3.0; 3]);

        storage.average_strategy(k, 3, &mut strategy);
        assert_strategy_close(&strategy, &[1.0 / 3.0; 3]);
        assert_eq!(storage.entry_count(), 0);
    }

    #[timed_test]
    fn regret_round_trip_and_regret_matching() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(2);

        storage.add_regret(k, 3, 0, -500);
        storage.add_regret(k, 3, 1, 100);
        storage.add_regret(k, 3, 2, 300);

        assert_eq!(storage.get_regret(k, 0), -500);
        assert_eq!(storage.get_regret(k, 1), 100);
        assert_eq!(storage.get_regret(k, 2), 300);

        let mut strategy = [0.0; 3];
        storage.regret_matched_strategy(k, 3, &mut strategy);
        assert_strategy_close(&strategy, &[0.0, 0.25, 0.75]);
    }

    #[timed_test]
    fn strategy_sum_round_trip_and_average_strategy() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(2);

        storage.add_strategy_sum(k, 2, 0, 300);
        storage.add_strategy_sum(k, 2, 1, 700);
        storage.add_strategy_sum(k, 2, 1, -10);

        assert_eq!(storage.get_strategy_sum(k, 0), 300);
        assert_eq!(storage.get_strategy_sum(k, 1), 700);

        let mut strategy = [0.0; 2];
        storage.average_strategy(k, 2, &mut strategy);
        assert_strategy_close(&strategy, &[0.3, 0.7]);
    }

    #[timed_test]
    fn out_of_range_updates_are_ignored() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(9);

        storage.add_regret(k, 2, 2, 100);
        storage.add_strategy_sum(k, 2, 3, 100);

        assert_eq!(storage.entry_count(), 0);
        assert_eq!(storage.get_regret(k, 2), 0);
        assert_eq!(storage.get_strategy_sum(k, 3), 0);
    }

    #[timed_test]
    fn regret_floor_is_applied_to_updates() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(1);

        storage.set_regret_floor(-1_000);
        storage.add_regret(k, 2, 0, -5_000);

        assert_eq!(storage.get_regret(k, 0), -1_000);
    }

    #[timed_test]
    fn discount_updates_visited_entries_only() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(3);

        storage.add_regret(k, 2, 0, 101);
        storage.add_regret(k, 2, 1, -101);
        storage.add_strategy_sum(k, 2, 0, 1_000);
        storage.add_strategy_sum(k, 2, 1, 2_000);

        storage.discount(0.5, 0.25, 0.1);

        assert_eq!(storage.get_regret(k, 0), 51);
        assert_eq!(storage.get_regret(k, 1), -25);
        assert_eq!(storage.get_strategy_sum(k, 0), 100);
        assert_eq!(storage.get_strategy_sum(k, 1), 200);
        assert_eq!(storage.entry_count(), 1);
    }

    #[timed_test]
    fn discount_updates_entries_across_many_shards() {
        let storage = SparseMpStorage::with_shards(64);
        for bucket in 0..512 {
            let k = key(bucket);
            storage.add_regret(k, 2, 0, 100);
            storage.add_regret(k, 2, 1, -100);
            storage.add_strategy_sum(k, 2, 0, 1_000);
            storage.add_strategy_sum(k, 2, 1, 2_000);
        }

        let before = storage.stats();
        assert!(
            before.nonempty_shards > 1,
            "test should cover multiple shards"
        );

        storage.discount(0.5, 0.25, 0.1);

        for bucket in 0..512 {
            let k = key(bucket);
            assert_eq!(storage.get_regret(k, 0), 50);
            assert_eq!(storage.get_regret(k, 1), -25);
            assert_eq!(storage.get_strategy_sum(k, 0), 100);
            assert_eq!(storage.get_strategy_sum(k, 1), 200);
        }
        assert_eq!(storage.entry_count(), 512);
    }

    #[timed_test]
    fn stats_count_entries_and_slots() {
        let storage = SparseMpStorage::with_shards(4);

        storage.add_regret(key(1), 2, 0, 1);
        storage.add_strategy_sum(key(2), 3, 1, 1);

        let stats = storage.stats();
        assert_eq!(stats.entries, 2);
        assert_eq!(stats.regret_slots, 5);
        assert_eq!(stats.strategy_slots, 5);
        assert_eq!(stats.shard_count, 4);
        assert!(stats.nonempty_shards > 0);
        assert!(stats.max_entries_per_shard > 0);
        assert!(stats.approx_bytes > 0);
    }

    #[timed_test]
    fn stats_do_not_double_count_existing_entries() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(1);

        storage.add_regret(k, 3, 0, 1);
        storage.add_regret(k, 3, 1, 1);
        storage.add_strategy_sum(k, 3, 2, 1);

        let stats = storage.stats();
        assert_eq!(storage.entry_count(), 1);
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.regret_slots, 3);
        assert_eq!(stats.strategy_slots, 3);
    }

    #[timed_test]
    fn activity_counts_sparse_probes_hits_and_inserts() {
        let storage = SparseMpStorage::with_shards(4);
        let k = key(1);

        assert_eq!(storage.activity(), SparseStorageActivity::default());

        assert_eq!(storage.get_regret(k, 0), 0);
        let after_missing_read = storage.activity();
        assert_eq!(after_missing_read.read_probes, 1);
        assert_eq!(after_missing_read.read_hits, 0);

        storage.add_regret(k, 2, 0, 1);
        let after_insert = storage.activity();
        assert_eq!(after_insert.write_probes, 1);
        assert_eq!(after_insert.write_hits, 0);
        assert_eq!(after_insert.inserts, 1);

        storage.add_strategy_sum(k, 2, 1, 2);
        let after_existing_write = storage.activity();
        assert_eq!(after_existing_write.write_probes, 2);
        assert_eq!(after_existing_write.write_hits, 1);
        assert_eq!(after_existing_write.inserts, 1);

        assert_eq!(storage.get_strategy_sum(k, 1), 2);
        let after_hit_read = storage.activity();
        assert_eq!(after_hit_read.read_probes, 2);
        assert_eq!(after_hit_read.read_hits, 1);
    }

    #[timed_test]
    fn insert_attribution_counts_new_entries_by_key_shape() {
        let storage = SparseMpStorage::with_shards(4);
        let preflop = MpInfosetKey::from_parts(
            Seat::from_raw(0),
            Street::Preflop,
            1,
            0,
            0,
            0,
            0,
            0,
        );
        let river = MpInfosetKey::from_parts(
            Seat::from_raw(5),
            Street::River,
            2,
            31,
            0,
            0,
            0,
            70,
        );

        storage.add_regret(preflop, 1, 0, 1);
        storage.add_strategy_sum(preflop, 1, 0, 1);
        storage.add_regret(river, 9, 0, 1);

        let attribution = storage.insert_attribution();
        assert_eq!(attribution.by_street[Street::Preflop as usize], 1);
        assert_eq!(attribution.by_street[Street::River as usize], 1);
        assert_eq!(attribution.by_seat[0], 1);
        assert_eq!(attribution.by_seat[5], 1);
        assert_eq!(attribution.by_spr_bucket[0], 1);
        assert_eq!(attribution.by_spr_bucket[31], 1);
        assert_eq!(attribution.history_len_bins[0], 1);
        assert_eq!(attribution.history_len_bins[7], 1);
        assert_eq!(attribution.history_len_max, 70);
        assert_eq!(attribution.action_count_bins[0], 1);
        assert_eq!(attribution.action_count_bins[4], 1);
        assert_eq!(attribution.action_count_sum, 10);
        assert_eq!(attribution.action_count_max, 9);
    }

    #[timed_test]
    fn default_storage_uses_high_shard_count_for_large_lazy_runs() {
        let storage = SparseMpStorage::new();
        let stats = storage.stats();

        assert_eq!(stats.shard_count, DEFAULT_SHARDS);
        assert!(stats.shard_count >= 4096);
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.nonempty_shards, 0);
        assert_eq!(stats.max_entries_per_shard, 0);
    }

    #[timed_test]
    fn snapshot_entries_are_sorted_and_complete() {
        let storage = SparseMpStorage::with_shards(4);

        storage.add_regret(key(9), 2, 1, 22);
        storage.add_strategy_sum(key(9), 2, 0, 11);
        storage.add_regret(key(1), 3, 2, 33);

        let snapshot = storage.snapshot_entries();

        assert_eq!(snapshot.len(), 2);
        assert!(snapshot[0].key < snapshot[1].key);
        assert_eq!(snapshot[0].num_actions, 3);
        assert_eq!(snapshot[0].regrets, vec![0, 0, 33]);
        assert_eq!(snapshot[0].strategy_sums, vec![0, 0, 0]);
        assert_eq!(snapshot[1].num_actions, 2);
        assert_eq!(snapshot[1].regrets, vec![0, 22]);
        assert_eq!(snapshot[1].strategy_sums, vec![11, 0]);
    }

    #[timed_test]
    fn telemetry_sample_reports_sparse_regret_and_strategy_movement_signal() {
        let storage = SparseMpStorage::with_shards(4);

        storage.add_regret(key(3), 3, 0, 30);
        storage.add_regret(key(3), 3, 1, -20);
        storage.add_strategy_sum(key(3), 3, 0, 10);
        storage.add_strategy_sum(key(3), 3, 1, 30);
        storage.add_regret(key(7), 2, 1, 15);

        let sample = storage.telemetry_sample(16);

        assert_eq!(sample.entries_sampled, 2);
        assert_eq!(sample.regret_slots_sampled, 5);
        assert_eq!(sample.max_positive_regret, 30);
        assert_eq!(sample.max_negative_regret, -20);
        assert!((sample.avg_positive_regret - 22.5).abs() < 1e-9);
        assert!(
            sample.strategy_fingerprint > 0.0,
            "strategy fingerprint should move once strategy sums exist"
        );
    }

    #[timed_test]
    fn snapshot_entries_can_restore_storage() {
        let storage = SparseMpStorage::with_shards(4);

        storage.add_regret(key(4), 2, 0, 44);
        storage.add_strategy_sum(key(4), 2, 1, 55);

        let restored = SparseMpStorage::from_snapshot_entries(storage.snapshot_entries());

        assert_eq!(restored.entry_count(), 1);
        let stats = restored.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.regret_slots, 2);
        assert_eq!(stats.strategy_slots, 2);
        assert_eq!(restored.get_regret(key(4), 0), 44);
        assert_eq!(restored.get_regret(key(4), 1), 0);
        assert_eq!(restored.get_strategy_sum(key(4), 0), 0);
        assert_eq!(restored.get_strategy_sum(key(4), 1), 55);
    }
}
