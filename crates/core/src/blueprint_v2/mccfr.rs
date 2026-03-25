//! External-sampling MCCFR traversal for the blueprint strategy.
//!
//! Implements the core counterfactual regret minimisation loop:
//! at the traverser's decision nodes **all** actions are explored;
//! at the opponent's decision nodes **one** action is sampled
//! according to the current regret-matched strategy.

// Arena indices are u32, bucket indices u16. Truncation is safe for
// any practical game tree. Precision loss on cast to f64 is acceptable
// (action counts and bucket counts are small).
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss
)]

use std::cmp::Ordering;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, RwLock};

use rand::Rng;

/// Global diagnostic counters for prune-hit measurement.
/// Updated once per batch (not per action) and read by the trainer for TUI.
pub static PRUNE_HITS: AtomicU64 = AtomicU64::new(0);
pub static PRUNE_TOTAL: AtomicU64 = AtomicU64::new(0);

/// Prune statistics accumulated during a single traversal.
///
/// Collected locally to avoid global atomic contention, then merged
/// into the global counters after the batch completes.
#[derive(Debug, Default, Clone, Copy)]
pub struct PruneStats {
    pub hits: u64,
    pub total: u64,
}

impl PruneStats {
    pub fn merge(&mut self, other: PruneStats) {
        self.hits += other.hits;
        self.total += other.total;
    }
}

use rustc_hash::FxHashMap;

use super::bucket_file::{BucketFile, PackedBoard};
use super::cluster_pipeline::{canonical_key, combo_index};
use super::game_tree::{GameNode, GameTree, TerminalKind};
use super::per_flop_bucket_file::PerFlopBucketFile;
use super::storage::BlueprintStorage;
use super::Street;
use crate::abstraction::isomorphism::CanonicalBoard;
use crate::poker::{Card, Hand, Rankable};

/// Maximum number of actions at any decision node.
///
/// Covers fold/check/call plus up to 13 bet/raise sizing choices.
/// Used for stack-allocated strategy and value buffers in the MCCFR
/// traversal hot path.
const MAX_ACTIONS: usize = 16;

/// A sampled deal: hole cards for each player plus the full 5-card board.
#[derive(Debug, Clone)]
pub struct Deal {
    pub hole_cards: [[Card; 2]; 2], // [player0, player1]
    pub board: [Card; 5],           // flop(3) + turn(1) + river(1)
}

/// A deal with pre-computed bucket assignments for all streets and players.
///
/// Eliminates repeated `get_bucket()` calls during
/// MCCFR traversal by computing all buckets once up-front.
#[derive(Debug, Clone)]
pub struct DealWithBuckets {
    pub deal: Deal,
    /// Pre-computed bucket indices: `buckets[player][street]`.
    pub buckets: [[u16; 4]; 2],
}

/// Loaded bucket assignments for all 4 streets.
///
/// During MCCFR, we look up buckets by street using the deal's cards.
/// Uses precomputed bucket files for O(1) lookups.
/// In production, panics if no bucket file is available for a postflop street.
/// In `#[cfg(test)]`, falls back to equity-based bucketing.
pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    /// Per-street bucket files produced by the clustering pipeline.
    pub bucket_files: [Option<BucketFile>; 4],
    /// Board index lookup tables for O(1) bucket file lookups.
    board_maps: [Option<FxHashMap<PackedBoard, u32>>; 4],
    /// Directory containing per-flop bucket files (`flop_NNNN.buckets`).
    /// When set, turn/river lookups use per-flop files instead of global
    /// bucket files.
    per_flop_dir: Option<PathBuf>,
    /// Cache of loaded per-flop bucket files, keyed by canonical flop
    /// `PackedBoard`.
    per_flop_cache: RwLock<FxHashMap<PackedBoard, Arc<PerFlopBucketFile>>>,
    /// Map from canonical flop `PackedBoard` to flop file index (the NNNN
    /// in `flop_NNNN.buckets`).
    flop_index_map: Option<FxHashMap<PackedBoard, u16>>,
}

impl AllBuckets {
    #[must_use]
    pub fn new(
        bucket_counts: [u16; 4],
        bucket_files: [Option<BucketFile>; 4],
    ) -> Self {
        let board_maps = std::array::from_fn(|i| {
            bucket_files[i].as_ref().and_then(|bf| {
                if bf.boards.is_empty() {
                    None
                } else {
                    Some(bf.board_index_map())
                }
            })
        });
        Self {
            bucket_counts,
            bucket_files,
            board_maps,
            per_flop_dir: None,
            per_flop_cache: RwLock::new(FxHashMap::default()),
            flop_index_map: None,
        }
    }

    /// Returns `true` if per-flop bucket lookups are enabled.
    #[must_use]
    pub fn has_per_flop_dir(&self) -> bool {
        self.per_flop_dir.is_some()
    }

    /// Enable per-flop bucket lookups for turn and river streets.
    ///
    /// Scans `dir` for `flop_NNNN.buckets` files, loads each header to
    /// extract the canonical flop, and builds an index mapping each flop
    /// `PackedBoard` to its file index.
    ///
    /// # Panics
    ///
    /// Panics if the internal per-flop cache lock is poisoned.
    #[must_use]
    pub fn with_per_flop_dir(mut self, dir: PathBuf) -> Self {
        let mut index_map = FxHashMap::default();

        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = match path.file_name().and_then(|n| n.to_str()) {
                    Some(n) => n.to_string(),
                    None => continue,
                };
                // Match flop_NNNN.buckets pattern
                if !name.starts_with("flop_") || !name.ends_with(".buckets") {
                    continue;
                }
                let idx_str = &name[5..name.len() - 8]; // strip "flop_" and ".buckets"
                let file_index: u16 = match idx_str.parse() {
                    Ok(v) => v,
                    Err(_) => continue,
                };

                // Load the file to get the flop cards
                if let Ok(pf) = PerFlopBucketFile::load(&path) {
                    let Ok(canonical) = CanonicalBoard::from_cards(&pf.flop_cards[..]) else {
                        continue;
                    };
                    let packed = canonical_key(&canonical.cards);
                    index_map.insert(packed, file_index);
                    // Also cache the loaded file
                    self.per_flop_cache
                        .write()
                        .expect("per_flop_cache lock")
                        .insert(packed, Arc::new(pf));
                }
            }
        }

        self.per_flop_dir = Some(dir);
        self.flop_index_map = Some(index_map);
        self
    }

    /// Look up a bucket for a postflop street via bucket file.
    ///
    /// Canonicalizes the board, looks up the board index in the hash map,
    /// applies the same suit permutation to hole cards, and reads the bucket
    /// from the flat array. In production, panics if no bucket file is
    /// available. In `#[cfg(test)]`, falls back to equity-based bucketing.
    ///
    /// For turn (`street_idx`=2) and river (`street_idx`=3), when `per_flop_dir`
    /// is set, uses per-flop bucket files instead of global bucket files.
    fn lookup_bucket(&self, street_idx: usize, hole: [Card; 2], board: &[Card]) -> u16 {
        // Per-flop lookup for turn and river when available
        if (street_idx == 2 || street_idx == 3) && self.per_flop_dir.is_some() {
            if let Some(bucket) = self.lookup_per_flop(street_idx, hole, board) {
                return bucket;
            }
            // In tests, fall back to equity-based bucketing
            #[cfg(not(test))]
            {
                let street_name = if street_idx == 2 { "turn" } else { "river" };
                panic!(
                    "No per-flop bucket file found for {street_name} board {board:?} \
                     with hole cards {hole:?}. Bucket files must be generated before training.",
                );
            }
        } else if let (Some(bf), Some(board_map)) = (
            &self.bucket_files[street_idx],
            &self.board_maps[street_idx],
        ) {
            if let Ok(canonical) = CanonicalBoard::from_cards(board) {
                let packed = canonical_key(&canonical.cards);
                if let Some(&board_idx) = board_map.get(&packed) {
                    let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
                    let ci = combo_index(c0, c1);
                    return bf.get_bucket(board_idx, ci);
                }
            }
        }

        // In tests, fall back to equity-based bucketing.
        // In production, missing bucket files are a fatal configuration error.
        #[cfg(test)]
        {
            let equity = crate::showdown_equity::compute_equity(hole, board);
            let k = self.bucket_counts[street_idx];
            let bucket = (equity * f64::from(k)) as u16;
            return bucket.min(k - 1);
        }
        #[cfg(not(test))]
        {
            let street_name = match street_idx {
                1 => "flop",
                2 => "turn",
                3 => "river",
                _ => "unknown",
            };
            panic!(
                "No bucket file found for {street_name} (street_idx={street_idx}), \
                 board {board:?}, hole {hole:?}. Bucket files must be generated before training.",
            );
        }
    }

    /// Per-flop bucket lookup for turn (`street_idx`=2) or river (`street_idx`=3).
    ///
    /// Canonicalizes the flop portion of the board, looks up the per-flop
    /// file (loading from cache or disk), then finds the turn/river card
    /// indices and returns the bucket. Returns `None` if the flop or
    /// card is not found in the per-flop file.
    fn lookup_per_flop(
        &self,
        street_idx: usize,
        hole: [Card; 2],
        board: &[Card],
    ) -> Option<u16> {
        let index_map = self.flop_index_map.as_ref()?;

        // Canonicalize the flop (first 3 board cards)
        let canonical_flop = CanonicalBoard::from_cards(&board[..3]).ok()?;
        let flop_key = canonical_key(&canonical_flop.cards);

        // Check if we have a per-flop file for this flop
        let _file_idx = index_map.get(&flop_key)?;

        // Get or load the per-flop file from cache
        let pf = self.get_per_flop_file(flop_key)?;

        // Apply the flop's suit mapping to the hole cards for combo index
        let (h0, h1) = canonical_flop.canonicalize_holding(hole[0], hole[1]);
        let ci = combo_index(h0, h1) as usize;

        // Map the turn card through the flop's suit canonicalization
        let canon_turn = canonical_flop.mapping.map_card(board[3]);

        // Find turn card index in the per-flop file
        let turn_idx = pf.turn_cards.iter().position(|&tc| tc == canon_turn)?;

        if street_idx == 2 {
            // Turn bucket
            Some(pf.get_turn_bucket(turn_idx, ci))
        } else {
            // River: also need to find the river card
            let canon_river = canonical_flop.mapping.map_card(board[4]);
            let river_idx = pf.river_cards_per_turn[turn_idx]
                .iter()
                .position(|&rc| rc == canon_river)?;
            Some(pf.get_river_bucket(turn_idx, river_idx, ci))
        }
    }

    /// Get a per-flop file from the cache, loading from disk if needed.
    fn get_per_flop_file(&self, flop_key: PackedBoard) -> Option<Arc<PerFlopBucketFile>> {
        // Fast path: read lock
        {
            let cache = self.per_flop_cache.read().expect("per_flop_cache read lock");
            if let Some(pf) = cache.get(&flop_key) {
                return Some(Arc::clone(pf));
            }
        }

        // Slow path: load from disk, write lock
        let dir = self.per_flop_dir.as_ref()?;
        let index_map = self.flop_index_map.as_ref()?;
        let &file_idx = index_map.get(&flop_key)?;

        let path = dir.join(format!("flop_{file_idx:04}.buckets"));
        let pf = PerFlopBucketFile::load(&path).ok()?;
        let arc = Arc::new(pf);

        let mut cache = self.per_flop_cache.write().expect("per_flop_cache write lock");
        cache.insert(flop_key, Arc::clone(&arc));
        Some(arc)
    }

    /// Pre-compute bucket assignments for all 4 streets x 2 players.
    #[must_use]
    pub fn precompute_buckets(&self, deal: &Deal) -> [[u16; 4]; 2] {
        let mut result = [[0u16; 4]; 2];
        for (player, row) in result.iter_mut().enumerate() {
            let hole = deal.hole_cards[player];
            // Preflop: canonical hand index.
            let hand = crate::hands::CanonicalHand::from_cards(hole[0], hole[1]);
            let idx = hand.index() as u16;
            row[0] = idx.min(self.bucket_counts[0] - 1);
            // Postflop: bucket file lookup (panics if missing).
            row[1] = self.lookup_bucket(1, hole, &deal.board[..3]); // flop
            row[2] = self.lookup_bucket(2, hole, &deal.board[..4]); // turn
            row[3] = self.lookup_bucket(3, hole, &deal.board[..5]); // river
        }
        result
    }

    /// Look up the bucket for a single street.
    ///
    /// Preflop uses canonical hand index. Postflop uses bucket file
    /// lookup (panics in production if no bucket file is available;
    /// falls back to equity in tests).
    #[must_use]
    pub fn get_bucket(&self, street: Street, hole_cards: [Card; 2], board: &[Card]) -> u16 {
        if street == Street::Preflop {
            let hand = crate::hands::CanonicalHand::from_cards(hole_cards[0], hole_cards[1]);
            let idx = hand.index() as u16;
            return idx.min(self.bucket_counts[0] - 1);
        }
        self.lookup_bucket(street as usize, hole_cards, board)
    }

    /// Return the visible board slice for a given street.
    #[must_use]
    pub fn board_for_street(board: &[Card; 5], street: Street) -> &[Card] {
        match street {
            Street::Preflop => &[],
            Street::Flop => &board[..3],
            Street::Turn => &board[..4],
            Street::River => &board[..5],
        }
    }
}

/// Full-tree EV accumulator that tracks EVs at ALL decision nodes.
///
/// Maps every arena index to a dense decision-node index via `decision_map`.
/// Each decision node gets a `[player][169]` pair of sum/count arrays.
/// Persisted as `hand_ev.bin` in snapshots for resume support.
pub struct FullTreeEvTracker {
    num_nodes: usize,
    /// `decision_map[arena_idx]` -> decision_node_idx (`u32::MAX` if not a decision).
    decision_map: Vec<u32>,
    /// `ev_sum[decision_idx][player][hand_index]` -- sum * 1000.
    pub ev_sum: Vec<[[AtomicI64; 169]; 2]>,
    /// `ev_count[decision_idx][player][hand_index]`.
    pub ev_count: Vec<[[AtomicU64; 169]; 2]>,
}

impl FullTreeEvTracker {
    /// Create a new tracker for all decision nodes in the tree.
    #[must_use]
    pub fn new(tree: &super::game_tree::GameTree) -> Self {
        let decision_map = tree.decision_index_map();
        let num_nodes = decision_map.iter().filter(|&&d| d != u32::MAX).count();
        let ev_sum = (0..num_nodes)
            .map(|_| {
                [
                    std::array::from_fn(|_| AtomicI64::new(0)),
                    std::array::from_fn(|_| AtomicI64::new(0)),
                ]
            })
            .collect();
        let ev_count = (0..num_nodes)
            .map(|_| {
                [
                    std::array::from_fn(|_| AtomicU64::new(0)),
                    std::array::from_fn(|_| AtomicU64::new(0)),
                ]
            })
            .collect();
        Self {
            num_nodes,
            decision_map,
            ev_sum,
            ev_count,
        }
    }

    /// Accumulate an EV sample at an arena node. No-op if the node is not a decision.
    pub fn accumulate(&self, arena_idx: u32, player: usize, hand_index: usize, ev: f64) {
        let dec_idx = self.decision_map[arena_idx as usize];
        if dec_idx == u32::MAX {
            return;
        }
        let idx = dec_idx as usize;
        self.ev_sum[idx][player][hand_index]
            .fetch_add((ev * 1000.0) as i64, AtomicOrdering::Relaxed);
        self.ev_count[idx][player][hand_index]
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Compute average EV per hand for a decision node and player.
    #[must_use]
    pub fn hand_ev_array(&self, decision_idx: usize, player: usize) -> [f64; 169] {
        std::array::from_fn(|i| {
            let sum =
                self.ev_sum[decision_idx][player][i].load(AtomicOrdering::Relaxed) as f64 / 1000.0;
            let count =
                self.ev_count[decision_idx][player][i].load(AtomicOrdering::Relaxed);
            if count > 0 {
                sum / count as f64
            } else {
                0.0
            }
        })
    }

    /// Number of decision nodes tracked.
    #[must_use]
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// Save to binary file: `[u32 num_nodes][per node: per player: per hand: i64 sum, u64 count]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or written.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(self.num_nodes as u32).to_le_bytes())?;
        for node_idx in 0..self.num_nodes {
            for player in 0..2 {
                for hand in 0..169 {
                    let sum =
                        self.ev_sum[node_idx][player][hand].load(AtomicOrdering::Relaxed);
                    let count =
                        self.ev_count[node_idx][player][hand].load(AtomicOrdering::Relaxed);
                    f.write_all(&sum.to_le_bytes())?;
                    f.write_all(&count.to_le_bytes())?;
                }
            }
        }
        Ok(())
    }

    /// Load from binary file. Handles mismatched node counts gracefully:
    /// loads min(saved, current) nodes, ignores extras.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or read.
    pub fn load(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Read;
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut buf4 = [0u8; 4];
        f.read_exact(&mut buf4)?;
        let saved_nodes = u32::from_le_bytes(buf4) as usize;
        let nodes_to_load = saved_nodes.min(self.num_nodes);
        let mut buf8 = [0u8; 8];
        for node_idx in 0..saved_nodes {
            for player in 0..2 {
                for hand in 0..169 {
                    f.read_exact(&mut buf8)?;
                    let sum = i64::from_le_bytes(buf8);
                    f.read_exact(&mut buf8)?;
                    let count = u64::from_le_bytes(buf8);
                    if node_idx < nodes_to_load {
                        self.ev_sum[node_idx][player][hand]
                            .store(sum, AtomicOrdering::Relaxed);
                        self.ev_count[node_idx][player][hand]
                            .store(count, AtomicOrdering::Relaxed);
                    }
                }
            }
        }
        Ok(())
    }

    /// Reset all accumulators to zero.
    pub fn reset(&self) {
        for node in &self.ev_sum {
            for player in node {
                for val in player {
                    val.store(0, AtomicOrdering::Relaxed);
                }
            }
        }
        for node in &self.ev_count {
            for player in node {
                for val in player {
                    val.store(0, AtomicOrdering::Relaxed);
                }
            }
        }
    }
}

/// Per-scenario-node EV accumulator for TUI display.
///
/// Each scenario node gets a `[player][169]` pair of sum/count arrays.
/// Updated during MCCFR traversal when visiting tracked nodes.
pub struct ScenarioEvTracker {
    /// Node indices being tracked.
    pub node_indices: Vec<u32>,
    /// `ev_sum[scenario_idx][player][hand_index]` -- sum * 1000.
    pub ev_sum: Vec<[[AtomicI64; 169]; 2]>,
    /// `ev_count[scenario_idx][player][hand_index]`.
    pub ev_count: Vec<[[AtomicU64; 169]; 2]>,
}

impl ScenarioEvTracker {
    /// Create a new tracker for the given node indices.
    #[must_use]
    pub fn new(node_indices: Vec<u32>) -> Self {
        let len = node_indices.len();
        let ev_sum = (0..len)
            .map(|_| [std::array::from_fn(|_| AtomicI64::new(0)), std::array::from_fn(|_| AtomicI64::new(0))])
            .collect();
        let ev_count = (0..len)
            .map(|_| [std::array::from_fn(|_| AtomicU64::new(0)), std::array::from_fn(|_| AtomicU64::new(0))])
            .collect();
        Self {
            node_indices,
            ev_sum,
            ev_count,
        }
    }

    /// Reinitialize the tracker with new node indices.
    pub fn set_nodes(&mut self, node_indices: Vec<u32>) {
        *self = Self::new(node_indices);
    }

    /// Find the scenario index for a given node index (linear scan).
    #[must_use]
    pub fn find_scenario(&self, node_idx: u32) -> Option<usize> {
        self.node_indices.iter().position(|&n| n == node_idx)
    }

    /// Accumulate an EV sample for a scenario node.
    pub fn accumulate(&self, scenario_idx: usize, player: usize, hand_index: usize, ev: f64) {
        self.ev_sum[scenario_idx][player][hand_index]
            .fetch_add((ev * 1000.0) as i64, AtomicOrdering::Relaxed);
        self.ev_count[scenario_idx][player][hand_index]
            .fetch_add(1, AtomicOrdering::Relaxed);
    }

    /// Compute average EV per hand for a scenario and player.
    #[must_use]
    pub fn hand_ev_array(&self, scenario_idx: usize, player: usize) -> [f64; 169] {
        std::array::from_fn(|i| {
            let sum = self.ev_sum[scenario_idx][player][i].load(AtomicOrdering::Relaxed) as f64 / 1000.0;
            let count = self.ev_count[scenario_idx][player][i].load(AtomicOrdering::Relaxed);
            if count > 0 { sum / count as f64 } else { 0.0 }
        })
    }

    /// Zero all accumulators.
    pub fn reset(&self) {
        for scenario in &self.ev_sum {
            for player in scenario {
                for v in player {
                    v.store(0, AtomicOrdering::Relaxed);
                }
            }
        }
        for scenario in &self.ev_count {
            for player in scenario {
                for v in player {
                    v.store(0, AtomicOrdering::Relaxed);
                }
            }
        }
    }
}

/// Run one external-sampling MCCFR iteration for the given traverser.
///
/// Returns the expected value for the traverser at the root (in BB
/// units, representing the net amount won/lost from the traverser's
/// invested chips).
///
/// External sampling: at the traverser's decision nodes, **all**
/// actions are explored. At the opponent's decision nodes, **one**
/// action is sampled according to the current strategy.
///
/// # Arguments
/// * `tree` - The game tree
/// * `storage` - Atomic regret/strategy storage (shared across threads)
/// * `deal` - The sampled deal with pre-computed bucket assignments
/// * `traverser` - Which player is traversing (0 or 1)
/// * `node_idx` - Current node in the tree
/// * `prune` - Whether negative-regret pruning is active
/// * `prune_threshold` - Regret threshold below which actions are skipped
/// * `rng` - Random number generator for opponent sampling
/// * `rake_rate` - Fraction of pot taken as rake (0.0 = no rake)
/// * `rake_cap` - Maximum rake in chip units (0.0 = no cap)
/// * `ev_tracker` - Optional per-scenario EV tracker for TUI display
/// * `full_ev_tracker` - Optional full-tree EV tracker for all decision nodes
#[allow(clippy::too_many_arguments)]
pub fn traverse_external(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
    ev_tracker: Option<&ScenarioEvTracker>,
    full_ev_tracker: Option<&FullTreeEvTracker>,
    baseline_alpha: f64,
) -> (f64, PruneStats) {
    match &tree.nodes[node_idx as usize] {
        GameNode::Terminal { kind, pot, stacks } => {
            (terminal_value(*kind, *pot, stacks, tree.starting_stack, traverser, &deal.deal, rake_rate, rake_cap), PruneStats::default())
        }

        GameNode::Chance { child, .. } => {
            // Board cards are pre-dealt in the Deal; just recurse.
            traverse_external(
                tree, storage, deal, traverser, *child, prune, prune_threshold, rng,
                rake_rate, rake_cap, ev_tracker, full_ev_tracker, baseline_alpha,
            )
        }

        GameNode::Decision {
            player,
            street,
            children,
            ..
        } => {
            let player = *player;
            let street = *street;
            let num_actions = children.len();

            let bucket = deal.buckets[player as usize][street as usize];

            if player == traverser {
                traverse_traverser(
                    tree,
                    storage,
                    deal,
                    traverser,
                    node_idx,
                    bucket,
                    children,
                    num_actions,
                    prune,
                    prune_threshold,
                    rng,
                    rake_rate,
                    rake_cap,
                    ev_tracker,
                    full_ev_tracker,
                    baseline_alpha,
                )
            } else {
                traverse_opponent(
                    tree,
                    storage,
                    deal,
                    traverser,
                    node_idx,
                    bucket,
                    children,
                    num_actions,
                    prune,
                    prune_threshold,
                    rng,
                    rake_rate,
                    rake_cap,
                    ev_tracker,
                    full_ev_tracker,
                    baseline_alpha,
                )
            }
        }
    }
}

/// Compute the traverser's payoff at a terminal node using the stacks-based model.
///
/// Each player starts with `starting_stack`. At the terminal node,
/// `stacks[player]` is the remaining stack and `pot` is the accumulated pot.
/// Payoffs are net profit/loss relative to starting stack:
///   - Winner:  (stacks\[t\] + pot - rake) - starting_stack
///   - Loser:   stacks\[t\] - starting_stack
///   - Tie:     (stacks\[t\] + pot/2 - rake/2) - starting_stack
///
/// When rake is enabled (`rake_rate > 0`), the winner pays
/// `min(pot * rake_rate, rake_cap)` from the pot. A `rake_cap` of `0.0` means no cap.
fn terminal_value(
    kind: TerminalKind,
    pot: f64,
    stacks: &[f64; 2],
    starting_stack: f64,
    traverser: u8,
    deal: &Deal,
    rake_rate: f64,
    rake_cap: f64,
) -> f64 {
    let rake = if rake_rate > 0.0 {
        let uncapped = pot * rake_rate;
        if rake_cap > 0.0 {
            uncapped.min(rake_cap)
        } else {
            uncapped
        }
    } else {
        0.0
    };
    let t = traverser as usize;
    match kind {
        TerminalKind::Fold { winner } => {
            if winner == traverser {
                (stacks[t] + pot - rake) - starting_stack
            } else {
                stacks[t] - starting_stack
            }
        }
        TerminalKind::Showdown => {
            let o = 1 - t;
            let rank_t = rank_hand(deal.hole_cards[t], &deal.board);
            let rank_o = rank_hand(deal.hole_cards[o], &deal.board);
            match rank_t.cmp(&rank_o) {
                Ordering::Greater => (stacks[t] + pot - rake) - starting_stack,
                Ordering::Less => stacks[t] - starting_stack,
                Ordering::Equal => (stacks[t] + pot / 2.0 - rake / 2.0) - starting_stack,
            }
        }
        TerminalKind::DepthBoundary => {
            unreachable!("DepthBoundary should not be reached during MCCFR traversal")
        }
    }
}

/// Assert that a CFV is within valid bounds for the game.
/// The maximum a player can win is the opponent's starting stack
/// (stack_depth in BB). The maximum they can lose is their own stack.
#[inline]
pub fn assert_cfv_bounds(cfv: f64, stack_depth: f64, context: &str) {
    assert!(
        cfv.abs() <= stack_depth * 1.01,
        "CFV {cfv:.1} exceeds stack depth {stack_depth:.1} — {context}",
    );
}

/// Traverser's decision node: explore all actions, update regrets and
/// strategy sums.
#[allow(clippy::too_many_arguments)]
fn traverse_traverser(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
    ev_tracker: Option<&ScenarioEvTracker>,
    full_ev_tracker: Option<&FullTreeEvTracker>,
    baseline_alpha: f64,
) -> (f64, PruneStats) {
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy_buf = [0.0f64; MAX_ACTIONS];
    storage.current_strategy_into(node_idx, bucket, &mut strategy_buf[..num_actions]);
    let strategy = &strategy_buf[..num_actions];

    let mut action_values_buf = [0.0f64; MAX_ACTIONS];
    let action_values = &mut action_values_buf[..num_actions];
    let mut pruned_buf = [false; MAX_ACTIONS];
    let pruned = &mut pruned_buf[..num_actions];
    let mut node_value = 0.0f64;
    let mut stats = PruneStats::default();

    for (a, &child_idx) in children.iter().enumerate() {
        if prune {
            stats.total += 1;
            if storage.get_regret(node_idx, bucket, a) < prune_threshold {
                stats.hits += 1;
                pruned[a] = true;
                continue;
            }
        }

        let (child_ev, child_stats) = traverse_external(
            tree, storage, deal, traverser, child_idx, prune, prune_threshold, rng,
            rake_rate, rake_cap, ev_tracker, full_ev_tracker, baseline_alpha,
        );
        action_values[a] = child_ev;
        node_value += strategy[a] * child_ev;
        stats.merge(child_stats);
    }

    // Update regrets: delta = action_value - node_value, scaled to
    // integer by ×1000 for precision. Skip pruned actions.
    // Also store the instantaneous regret as the SAPCFR+ prediction
    // for the next iteration (no-op when predictions are disabled).
    for (a, &av) in action_values.iter().enumerate().take(num_actions) {
        if pruned[a] {
            continue;
        }
        let delta = av - node_value;
        let delta_i32 = (delta * 1000.0) as i32;
        storage.add_regret(node_idx, bucket, a, delta_i32);
    }

    // Accumulate strategy sums (for computing the average strategy).
    for (a, &s) in strategy.iter().enumerate().take(num_actions) {
        storage.add_strategy_sum(node_idx, bucket, a, (s * 1000.0) as i64);
    }

    // Accumulate EV at tracked scenario nodes.
    let hand_index = crate::hands::CanonicalHand::from_cards(
        deal.deal.hole_cards[traverser as usize][0],
        deal.deal.hole_cards[traverser as usize][1],
    )
    .index();

    if let Some(tracker) = ev_tracker {
        if let Some(scenario_idx) = tracker.find_scenario(node_idx) {
            tracker.accumulate(scenario_idx, traverser as usize, hand_index, node_value);
        }
    }

    if let Some(tracker) = full_ev_tracker {
        tracker.accumulate(node_idx, traverser as usize, hand_index, node_value);
    }

    (node_value, stats)
}

/// Opponent's decision node: sample one action according to the
/// current strategy and recurse.
#[allow(clippy::too_many_arguments)]
fn traverse_opponent(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    bucket: u16,
    children: &[u32],
    num_actions: usize,
    prune: bool,
    prune_threshold: i32,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
    ev_tracker: Option<&ScenarioEvTracker>,
    full_ev_tracker: Option<&FullTreeEvTracker>,
    baseline_alpha: f64,
) -> (f64, PruneStats) {
    debug_assert!(num_actions <= MAX_ACTIONS);
    let mut strategy_buf = [0.0f64; MAX_ACTIONS];
    storage.current_strategy_into(node_idx, bucket, &mut strategy_buf[..num_actions]);
    let strategy = &strategy_buf[..num_actions];

    let r: f64 = rng.random();
    let mut cumulative = 0.0;
    let mut chosen = num_actions - 1;
    for (a, &prob) in strategy.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            chosen = a;
            break;
        }
    }

    let (v_sampled, child_stats) = traverse_external(
        tree, storage, deal, traverser, children[chosen], prune, prune_threshold, rng,
        rake_rate, rake_cap, ev_tracker, full_ev_tracker, baseline_alpha,
    );

    // Read baselines BEFORE updating (use old values for correction).
    let mut baseline_buf = [0.0f64; MAX_ACTIONS];
    for (a, slot) in baseline_buf[..num_actions].iter_mut().enumerate() {
        *slot = storage.get_baseline(node_idx, bucket, a);
    }

    // Baseline-corrected value: v = sum(s[a]*b[a]) + (v_sampled - b[chosen])
    // When baselines are disabled, get_baseline returns 0.0 and this
    // degenerates to the standard external-sampling value (v_sampled).
    let v = baseline_corrected_value(strategy, &baseline_buf[..num_actions], chosen, v_sampled);

    // Update the baseline for the sampled action with the new observation.
    storage.update_baseline(node_idx, bucket, chosen, v_sampled, baseline_alpha);

    (v, child_stats)
}

/// Compute the baseline-corrected value for VR-MCCFR (Schmid et al., AAAI 2019).
///
/// Formula: `v = sum(strategy[a] * baseline[a]) + (v_sampled - baseline[a_sampled])`
///
/// When baselines are all zero (disabled), this reduces to `v_sampled`.
fn baseline_corrected_value(strategy: &[f64], baselines: &[f64], a_sampled: usize, v_sampled: f64) -> f64 {
    let mut v = 0.0;
    for (&s, &b) in strategy.iter().zip(baselines.iter()) {
        v += s * b;
    }
    v += v_sampled - baselines[a_sampled];
    v
}

/// Rank a hand (hole + board) using `rs_poker`.
fn rank_hand(hole: [Card; 2], board: &[Card; 5]) -> crate::poker::Rank {
    let mut hand = Hand::default();
    hand.insert(hole[0]);
    hand.insert(hole[1]);
    for &c in board {
        hand.insert(c);
    }
    hand.rank()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blueprint_v2::game_tree::GameTree;
    use crate::blueprint_v2::storage::BlueprintStorage;

    // --- ScenarioEvTracker tests ---

    #[test]
    fn scenario_tracker_new_empty() {
        let tracker = ScenarioEvTracker::new(vec![]);
        assert!(tracker.node_indices.is_empty());
        assert!(tracker.ev_sum.is_empty());
        assert!(tracker.ev_count.is_empty());
    }

    #[test]
    fn scenario_tracker_new_with_nodes() {
        let tracker = ScenarioEvTracker::new(vec![3, 7]);
        assert_eq!(tracker.node_indices.len(), 2);
        assert_eq!(tracker.ev_sum.len(), 2);
        assert_eq!(tracker.ev_count.len(), 2);
    }

    #[test]
    fn scenario_tracker_find_scenario() {
        let tracker = ScenarioEvTracker::new(vec![3, 7, 12]);
        assert_eq!(tracker.find_scenario(3), Some(0));
        assert_eq!(tracker.find_scenario(7), Some(1));
        assert_eq!(tracker.find_scenario(12), Some(2));
        assert_eq!(tracker.find_scenario(99), None);
    }

    #[test]
    fn scenario_tracker_accumulate_and_read() {
        let tracker = ScenarioEvTracker::new(vec![5]);
        tracker.accumulate(0, 0, 42, 2.5);
        tracker.accumulate(0, 0, 42, 3.5);
        let evs = tracker.hand_ev_array(0, 0);
        assert!((evs[42] - 3.0).abs() < 1e-3, "expected avg 3.0, got {}", evs[42]);
        // Unaccumulated hands should be 0.
        assert!((evs[0] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn scenario_tracker_accumulate_both_players() {
        let tracker = ScenarioEvTracker::new(vec![5]);
        tracker.accumulate(0, 0, 10, 1.0);
        tracker.accumulate(0, 1, 10, -1.0);
        let evs_p0 = tracker.hand_ev_array(0, 0);
        let evs_p1 = tracker.hand_ev_array(0, 1);
        assert!((evs_p0[10] - 1.0).abs() < 1e-3);
        assert!((evs_p1[10] - (-1.0)).abs() < 1e-3);
    }

    #[test]
    fn scenario_tracker_reset_zeroes_all() {
        let tracker = ScenarioEvTracker::new(vec![5, 10]);
        tracker.accumulate(0, 0, 0, 5.0);
        tracker.accumulate(1, 1, 100, -3.0);
        tracker.reset();
        let evs0 = tracker.hand_ev_array(0, 0);
        let evs1 = tracker.hand_ev_array(1, 1);
        assert!((evs0[0] - 0.0).abs() < 1e-10);
        assert!((evs1[100] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn scenario_tracker_set_nodes_reinitializes() {
        let mut tracker = ScenarioEvTracker::new(vec![5]);
        tracker.accumulate(0, 0, 42, 2.5);
        tracker.set_nodes(vec![10, 20]);
        assert_eq!(tracker.node_indices.len(), 2);
        assert_eq!(tracker.ev_sum.len(), 2);
        // Old data should be gone.
        let evs = tracker.hand_ev_array(0, 0);
        assert!((evs[42] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn scenario_tracker_hand_ev_array_no_samples() {
        let tracker = ScenarioEvTracker::new(vec![5]);
        let evs = tracker.hand_ev_array(0, 0);
        for &v in &evs {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    // --- FullTreeEvTracker tests ---

    #[test]
    fn full_tree_ev_tracker_new_counts_decision_nodes() {
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        // The toy tree has decision nodes; num_nodes should match.
        let (decision_count, _, _) = tree.node_counts();
        assert_eq!(tracker.num_nodes(), decision_count);
    }

    #[test]
    fn full_tree_ev_tracker_accumulates() {
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        tracker.accumulate(tree.root, 0, 42, 2.5);
        tracker.accumulate(tree.root, 0, 42, 3.5);
        let dec_idx = tree.decision_index_map()[tree.root as usize];
        let evs = tracker.hand_ev_array(dec_idx as usize, 0);
        assert!((evs[42] - 3.0).abs() < 0.01, "expected avg 3.0, got {}", evs[42]);
    }

    #[test]
    fn full_tree_ev_tracker_accumulate_non_decision_is_noop() {
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        // Find a terminal node (non-decision).
        let decision_map = tree.decision_index_map();
        let terminal_idx = decision_map.iter().position(|&d| d == u32::MAX).unwrap();
        // Accumulating at a non-decision node should not panic.
        tracker.accumulate(terminal_idx as u32, 0, 0, 1.0);
    }

    #[test]
    fn full_tree_ev_tracker_both_players() {
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        let dec_idx = tree.decision_index_map()[tree.root as usize];
        tracker.accumulate(tree.root, 0, 10, 5.0);
        tracker.accumulate(tree.root, 1, 10, -3.0);
        let evs_p0 = tracker.hand_ev_array(dec_idx as usize, 0);
        let evs_p1 = tracker.hand_ev_array(dec_idx as usize, 1);
        assert!((evs_p0[10] - 5.0).abs() < 0.01);
        assert!((evs_p1[10] - (-3.0)).abs() < 0.01);
    }

    #[test]
    fn full_tree_ev_tracker_no_samples_returns_zero() {
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        let dec_idx = tree.decision_index_map()[tree.root as usize];
        let evs = tracker.hand_ev_array(dec_idx as usize, 0);
        for &v in &evs {
            assert!((v - 0.0).abs() < 1e-10);
        }
    }

    #[test]
    fn full_tree_ev_tracker_reset_zeroes_all() {
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        tracker.accumulate(tree.root, 0, 42, 10.0);
        tracker.reset();
        let dec_idx = tree.decision_index_map()[tree.root as usize];
        let evs = tracker.hand_ev_array(dec_idx as usize, 0);
        assert!((evs[42] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn full_tree_ev_tracker_save_load_roundtrip() {
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        tracker.accumulate(tree.root, 0, 10, 5.0);
        tracker.accumulate(tree.root, 1, 100, -2.5);
        let tmp = std::env::temp_dir().join("test_hand_ev_roundtrip.bin");
        tracker.save(&tmp).unwrap();
        let tracker2 = FullTreeEvTracker::new(&tree);
        tracker2.load(&tmp).unwrap();
        let dec_idx = tree.decision_index_map()[tree.root as usize];
        let evs_p0 = tracker2.hand_ev_array(dec_idx as usize, 0);
        let evs_p1 = tracker2.hand_ev_array(dec_idx as usize, 1);
        assert!((evs_p0[10] - 5.0).abs() < 0.01, "p0 hand 10 should be 5.0, got {}", evs_p0[10]);
        assert!((evs_p1[100] - (-2.5)).abs() < 0.01, "p1 hand 100 should be -2.5, got {}", evs_p1[100]);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn full_tree_ev_tracker_load_smaller_file() {
        // Build a small tree, save, then load into a bigger tracker.
        let tree = toy_tree();
        let tracker = FullTreeEvTracker::new(&tree);
        tracker.accumulate(tree.root, 0, 5, 7.0);
        let tmp = std::env::temp_dir().join("test_hand_ev_smaller.bin");
        tracker.save(&tmp).unwrap();
        // Load into the same tree — should work fine.
        let tracker2 = FullTreeEvTracker::new(&tree);
        tracker2.load(&tmp).unwrap();
        let dec_idx = tree.decision_index_map()[tree.root as usize];
        let evs = tracker2.hand_ev_array(dec_idx as usize, 0);
        assert!((evs[5] - 7.0).abs() < 0.01);
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn traverse_with_full_tracker_accumulates_ev() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new([10, 10, 10, 10], [None, None, None, None]);
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let tracker = FullTreeEvTracker::new(&tree);

        for _ in 0..20 {
            traverse_external(
                &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000,
                &mut rng, 0.0, 0.0, None, Some(&tracker), 0.0,
            );
        }

        // The root node is a decision node for player 0 (SB),
        // so EVs for player 0 should have samples.
        let hand_idx = crate::hands::CanonicalHand::from_cards(
            precomputed.deal.hole_cards[0][0],
            precomputed.deal.hole_cards[0][1],
        ).index();
        let dec_idx = tree.decision_index_map()[tree.root as usize];
        let evs = tracker.hand_ev_array(dec_idx as usize, 0);
        let count = tracker.ev_count[dec_idx as usize][0][hand_idx].load(Ordering::Relaxed);
        assert!(count > 0, "should have accumulated samples for traverser hand");
        assert!(evs[hand_idx].is_finite(), "EV should be finite");
    }

    #[test]
    fn traverse_with_tracker_accumulates_ev() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new([10, 10, 10, 10], [None, None, None, None]);
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        // Track the root node.
        let tracker = ScenarioEvTracker::new(vec![tree.root]);

        for _ in 0..20 {
            traverse_external(
                &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000,
                &mut rng, 0.0, 0.0, Some(&tracker), None, 0.0,
            );
        }

        // The root node is a decision node for player 0 (SB),
        // so scenario EVs for player 0 should have samples.
        let hand_idx = crate::hands::CanonicalHand::from_cards(
            precomputed.deal.hole_cards[0][0],
            precomputed.deal.hole_cards[0][1],
        ).index();
        let evs = tracker.hand_ev_array(0, 0);
        let count = tracker.ev_count[0][0][hand_idx].load(Ordering::Relaxed);
        assert!(count > 0, "should have accumulated samples for traverser hand");
        assert!(evs[hand_idx].is_finite(), "EV should be finite");
    }

    #[test]
    fn traverse_without_tracker_still_works() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new([10, 10, 10, 10], [None, None, None, None]);
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let (ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000,
            &mut rng, 0.0, 0.0, None, None, 0.0,
        );
        assert!(ev.is_finite(), "EV should be finite without tracker");
    }
    use crate::poker::{Card, Suit, Value};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::atomic::Ordering;

    fn make_deal() -> Deal {
        Deal {
            hole_cards: [
                [
                    Card::new(Value::Ace, Suit::Spade),
                    Card::new(Value::King, Suit::Spade),
                ],
                [
                    Card::new(Value::Queen, Suit::Heart),
                    Card::new(Value::Jack, Suit::Heart),
                ],
            ],
            board: [
                Card::new(Value::Two, Suit::Club),
                Card::new(Value::Three, Suit::Diamond),
                Card::new(Value::Four, Suit::Club),
                Card::new(Value::Five, Suit::Diamond),
                Card::new(Value::Six, Suit::Heart),
            ],
        }
    }

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

    fn make_precomputed(buckets: &AllBuckets, deal: Deal) -> DealWithBuckets {
        let b = buckets.precompute_buckets(&deal);
        DealWithBuckets { deal, buckets: b }
    }

    #[test]
    fn traverse_returns_finite() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let (ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0, None, None, 0.0,
        );
        assert!(ev.is_finite(), "EV should be finite, got {ev}");
    }

    #[test]
    fn traverse_both_players() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let (ev0, _) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0, None, None, 0.0,
        );
        let (ev1, _) = traverse_external(
            &tree, &storage, &precomputed, 1, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0, None, None, 0.0,
        );

        assert!(ev0.is_finite());
        assert!(ev1.is_finite());
    }

    #[test]
    fn traverse_updates_regrets() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        assert!(storage.regrets.iter().all(|r| r.load(Ordering::Relaxed) == 0));

        let (_ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0, None, None, 0.0,
        );

        assert!(
            storage.regrets.iter().any(|r| r.load(Ordering::Relaxed) != 0),
            "regrets should be updated after traversal"
        );
    }

    #[test]
    fn traverse_updates_strategy_sums() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        let (_ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
            0.0, 0.0, None, None, 0.0,
        );

        assert!(
            storage.strategy_sums.iter().any(|s| s.load(Ordering::Relaxed) != 0),
            "strategy sums should be updated after traversal"
        );
    }

    #[test]
    fn multiple_iterations_change_strategy() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let deal = make_deal();
        let precomputed = make_precomputed(&buckets, deal);
        let mut rng = StdRng::seed_from_u64(42);

        for _ in 0..50 {
            let _ = traverse_external(
                &tree, &storage, &precomputed, 0, tree.root, false, -310_000_000, &mut rng,
                0.0, 0.0, None, None, 0.0,
            );
            let _ = traverse_external(
                &tree, &storage, &precomputed, 1, tree.root, false, -310_000_000, &mut rng,
                0.0, 0.0, None, None, 0.0,
            );
        }

        // After 50 iterations, at least one traverser decision node
        // should have a non-uniform current strategy.
        for (i, node) in tree.nodes.iter().enumerate() {
            if let GameNode::Decision { player: 0, street, .. } = node {
                let visible = AllBuckets::board_for_street(&precomputed.deal.board, *street);
                let bucket = buckets.get_bucket(*street, precomputed.deal.hole_cards[0], visible);
                let strategy = storage.current_strategy(i as u32, bucket);
                let is_uniform = strategy.iter().all(|&p| (p - strategy[0]).abs() < 1e-10);
                if strategy.len() > 1 && !is_uniform {
                    return; // pass
                }
            }
        }
        // Uniform everywhere is acceptable in degenerate cases.
    }

    #[test]
    fn traverse_with_pruning() {
        let tree = toy_tree();
        let storage = BlueprintStorage::new(&tree, [10, 10, 10, 10]);
        let buckets = AllBuckets::new(
            [10, 10, 10, 10],
            [None, None, None, None],
        );
        let precomputed = make_precomputed(&buckets, make_deal());
        let mut rng = StdRng::seed_from_u64(42);

        // Force some very negative regrets.
        for r in storage.regrets.iter().step_by(3) {
            r.store(-400_000_000, Ordering::Relaxed);
        }

        let (ev, _stats) = traverse_external(
            &tree, &storage, &precomputed, 0, tree.root, true, -310_000_000, &mut rng,
            0.0, 0.0, None, None, 0.0,
        );
        assert!(ev.is_finite());
    }

    #[test]
    fn terminal_fold_payoff() {
        let deal = make_deal();

        // Using stacks=[0,0], starting_stack=0 => pot-only semantics.
        // winner gets pot, loser gets 0.
        let v = terminal_value(TerminalKind::Fold { winner: 1 }, 1.5, &[0.0; 2], 0.0, 0, &deal, 0.0, 0.0);
        assert!(v.abs() < 1e-10, "Loser gets 0 on fold, got {v}");

        let v = terminal_value(TerminalKind::Fold { winner: 1 }, 1.5, &[0.0; 2], 0.0, 1, &deal, 0.0, 0.0);
        assert!((v - 1.5).abs() < 1e-10, "Winner gets pot=1.5 on fold, got {v}");
    }

    #[test]
    fn terminal_showdown_payoff() {
        let deal = make_deal();

        // Both have a straight on 2-3-4-5-6; AKs (6-high straight) vs
        // QJh (6-high straight). Both make the same board straight, so
        // this should be a tie (chop). stacks=[0,0], starting_stack=0 => tie = pot/2.
        let v = terminal_value(TerminalKind::Showdown, 4.0, &[0.0; 2], 0.0, 0, &deal, 0.0, 0.0);
        assert!(
            (v - 2.0).abs() < 1e-10,
            "Equal hands should chop (EV=pot/2=2.0), got {v}"
        );
    }

    // --- Rake tests ---

    /// Deal where player 0 wins at showdown: AA vs KK on a low board.
    fn make_deal_p0_wins() -> Deal {
        Deal {
            hole_cards: [
                [
                    Card::new(Value::Ace, Suit::Spade),
                    Card::new(Value::Ace, Suit::Heart),
                ],
                [
                    Card::new(Value::King, Suit::Spade),
                    Card::new(Value::King, Suit::Heart),
                ],
            ],
            board: [
                Card::new(Value::Two, Suit::Club),
                Card::new(Value::Three, Suit::Diamond),
                Card::new(Value::Four, Suit::Club),
                Card::new(Value::Seven, Suit::Diamond),
                Card::new(Value::Eight, Suit::Club),
            ],
        }
    }

    #[test]
    fn terminal_fold_with_rake() {
        let deal = make_deal();
        // stacks=[0,0], starting_stack=0 => pot-only semantics.
        // pot=10, 5% rake, cap 1.0 -> rake = min(10*0.05, 1.0) = 0.5
        // winner = pot - rake = 10 - 0.5 = 9.5
        let v = terminal_value(TerminalKind::Fold { winner: 1 }, 10.0, &[0.0; 2], 0.0, 1, &deal, 0.05, 1.0);
        assert!((v - 9.5).abs() < 1e-10, "Winner gets pot - rake = 9.5, got {v}");

        // Loser gets 0
        let v = terminal_value(TerminalKind::Fold { winner: 1 }, 10.0, &[0.0; 2], 0.0, 0, &deal, 0.05, 1.0);
        assert!(v.abs() < 1e-10, "Loser gets 0, got {v}");
    }

    #[test]
    fn terminal_showdown_with_rake_winner() {
        let deal = make_deal_p0_wins();
        // stacks=[0,0], starting_stack=0 => pot-only semantics.
        // pot=10, 5% rake, no cap -> rake = 0.5
        // winner = pot - rake = 10 - 0.5 = 9.5
        let v = terminal_value(TerminalKind::Showdown, 10.0, &[0.0; 2], 0.0, 0, &deal, 0.05, 0.0);
        assert!((v - 9.5).abs() < 1e-10, "Winner gets pot - rake = 9.5, got {v}");

        // Loser gets 0
        let v = terminal_value(TerminalKind::Showdown, 10.0, &[0.0; 2], 0.0, 1, &deal, 0.05, 0.0);
        assert!(v.abs() < 1e-10, "Loser gets 0, got {v}");
    }

    #[test]
    fn terminal_tie_with_rake() {
        let deal = make_deal(); // equal hands (tie via board straight)
        // stacks=[0,0], starting_stack=0 => pot-only semantics.
        // pot=10, 5% rake, no cap -> rake = 0.5
        // tie: pot/2 - rake/2 = 5 - 0.25 = 4.75
        let v = terminal_value(TerminalKind::Showdown, 10.0, &[0.0; 2], 0.0, 0, &deal, 0.05, 0.0);
        assert!((v - 4.75).abs() < 1e-10, "Tie: pot/2 - rake/2 = 4.75, got {v}");
    }

    #[test]
    fn terminal_rake_cap_applied() {
        let deal = make_deal(); // tie
        // stacks=[0,0], starting_stack=0 => pot-only semantics.
        // pot=100, 5% rake, cap 3.0 -> rake = min(5.0, 3.0) = 3.0 (capped)
        // tie: pot/2 - rake/2 = 50 - 1.5 = 48.5
        let v = terminal_value(TerminalKind::Showdown, 100.0, &[0.0; 2], 0.0, 0, &deal, 0.05, 3.0);
        assert!((v - 48.5).abs() < 1e-10, "Capped rake tie = 48.5, got {v}");
    }

    #[test]
    fn terminal_rake_zero_rate_is_noop() {
        let deal = make_deal_p0_wins();
        // stacks=[0,0], starting_stack=0 => pot-only semantics.
        // pot=10, rate=0 -> no rake. Winner gets pot = 10.
        let v = terminal_value(TerminalKind::Showdown, 10.0, &[0.0; 2], 0.0, 0, &deal, 0.0, 3.0);
        assert!((v - 10.0).abs() < 1e-10, "Zero rate means no rake, got {v}");
    }

    #[test]
    fn terminal_rake_uncapped() {
        let deal = make_deal_p0_wins();
        // stacks=[0,0], starting_stack=0 => pot-only semantics.
        // pot=100, 10% rake, no cap -> rake = 10.0
        // winner = pot - rake = 100 - 10 = 90
        let v = terminal_value(TerminalKind::Showdown, 100.0, &[0.0; 2], 0.0, 0, &deal, 0.10, 0.0);
        assert!((v - 90.0).abs() < 1e-10, "Winner gets pot - rake = 90, got {v}");
    }

    #[test]
    fn terminal_pot_only_model() {
        let deal = make_deal();

        // stacks=[0,0], starting_stack=0 => pot-only semantics.
        // SB fold at root, pot=1.5. Loser=0, winner=1.5.
        let v = terminal_value(TerminalKind::Fold { winner: 1 }, 1.5, &[0.0; 2], 0.0, 0, &deal, 0.0, 0.0);
        assert!(v.abs() < 1e-10, "Loser gets 0 on fold, got {v}");

        let v = terminal_value(TerminalKind::Fold { winner: 1 }, 1.5, &[0.0; 2], 0.0, 1, &deal, 0.0, 0.0);
        assert!((v - 1.5).abs() < 1e-10, "Winner gets pot=1.5, got {v}");

        // After raise: pot=6.0
        let deal_p0_wins = make_deal_p0_wins();
        // Player 0 wins showdown: pot = 6.0
        let v = terminal_value(TerminalKind::Showdown, 6.0, &[0.0; 2], 0.0, 0, &deal_p0_wins, 0.0, 0.0);
        assert!((v - 6.0).abs() < 1e-10, "Winner gets pot = 6.0, got {v}");

        // Player 1 loses: 0
        let v = terminal_value(TerminalKind::Showdown, 6.0, &[0.0; 2], 0.0, 1, &deal_p0_wins, 0.0, 0.0);
        assert!(v.abs() < 1e-10, "Loser gets 0, got {v}");
    }

    // ── Stacks-based payoff model tests ─────────────────────────────

    #[test]
    fn terminal_fold_stacks_based_winner() {
        let deal = make_deal();
        // SB raises to 2.5, BB folds.
        // stacks = [47.5, 49.0], pot = 3.5, starting_stack = 50.0
        // Winner (player 0): (47.5 + 3.5 - 0) - 50 = 1.0
        let v = terminal_value(
            TerminalKind::Fold { winner: 0 }, 3.5,
            &[47.5, 49.0], 50.0,
            0, &deal, 0.0, 0.0,
        );
        assert!((v - 1.0).abs() < 1e-10, "Winner EV should be 1.0, got {v}");
    }

    #[test]
    fn terminal_fold_stacks_based_loser() {
        let deal = make_deal();
        // stacks = [47.5, 49.0], pot = 3.5, starting_stack = 50.0
        // Loser (player 1): 49.0 - 50.0 = -1.0
        let v = terminal_value(
            TerminalKind::Fold { winner: 0 }, 3.5,
            &[47.5, 49.0], 50.0,
            1, &deal, 0.0, 0.0,
        );
        assert!((v - (-1.0)).abs() < 1e-10, "Loser EV should be -1.0, got {v}");
    }

    #[test]
    fn terminal_fold_stacks_based_zero_sum() {
        let deal = make_deal();
        // stacks = [47.5, 49.0], pot = 3.5, starting_stack = 50.0
        let winner_ev = terminal_value(
            TerminalKind::Fold { winner: 0 }, 3.5,
            &[47.5, 49.0], 50.0,
            0, &deal, 0.0, 0.0,
        );
        let loser_ev = terminal_value(
            TerminalKind::Fold { winner: 0 }, 3.5,
            &[47.5, 49.0], 50.0,
            1, &deal, 0.0, 0.0,
        );
        assert!(
            (winner_ev + loser_ev).abs() < 1e-10,
            "Zero-sum violated: {winner_ev} + {loser_ev} != 0"
        );
    }

    #[test]
    fn terminal_showdown_stacks_based_winner() {
        let deal = make_deal_p0_wins();
        // Both put in 5.0 each. stacks = [45.0, 45.0], pot = 10.0, starting_stack = 50.0
        // Winner (player 0): (45 + 10 - 0) - 50 = 5.0
        let v = terminal_value(
            TerminalKind::Showdown, 10.0,
            &[45.0, 45.0], 50.0,
            0, &deal, 0.0, 0.0,
        );
        assert!((v - 5.0).abs() < 1e-10, "Winner EV should be 5.0, got {v}");
    }

    #[test]
    fn terminal_showdown_stacks_based_loser() {
        let deal = make_deal_p0_wins();
        // stacks = [45.0, 45.0], pot = 10.0, starting_stack = 50.0
        // Loser (player 1): 45.0 - 50.0 = -5.0
        let v = terminal_value(
            TerminalKind::Showdown, 10.0,
            &[45.0, 45.0], 50.0,
            1, &deal, 0.0, 0.0,
        );
        assert!((v - (-5.0)).abs() < 1e-10, "Loser EV should be -5.0, got {v}");
    }

    #[test]
    fn terminal_tie_stacks_based() {
        let deal = make_deal(); // equal hands (tie via board straight)
        // stacks = [45.0, 45.0], pot = 10.0, starting_stack = 50.0
        // Tie: (45 + 10/2 - 0) - 50 = 0.0
        let v = terminal_value(
            TerminalKind::Showdown, 10.0,
            &[45.0, 45.0], 50.0,
            0, &deal, 0.0, 0.0,
        );
        assert!(v.abs() < 1e-10, "Tie EV should be 0.0, got {v}");
    }

    #[test]
    fn terminal_fold_stacks_based_with_rake() {
        let deal = make_deal();
        // pot = 10.0, 5% rake, cap 1.0 -> rake = min(0.5, 1.0) = 0.5
        // stacks = [45.0, 45.0], starting_stack = 50.0
        // Winner (player 0): (45 + 10 - 0.5) - 50 = 4.5
        let v = terminal_value(
            TerminalKind::Fold { winner: 0 }, 10.0,
            &[45.0, 45.0], 50.0,
            0, &deal, 0.05, 1.0,
        );
        assert!((v - 4.5).abs() < 1e-10, "Winner with rake should be 4.5, got {v}");
    }

    #[test]
    fn terminal_fold_stacks_based_loser_unaffected_by_rake() {
        let deal = make_deal();
        // pot = 10.0, 5% rake, cap 1.0 -> rake = 0.5
        // stacks = [45.0, 45.0], starting_stack = 50.0
        // Loser (player 1): 45.0 - 50.0 = -5.0 (rake does not affect loser)
        let v = terminal_value(
            TerminalKind::Fold { winner: 0 }, 10.0,
            &[45.0, 45.0], 50.0,
            1, &deal, 0.05, 1.0,
        );
        assert!((v - (-5.0)).abs() < 1e-10, "Loser EV should be -5.0 (unaffected by rake), got {v}");
    }

    #[test]
    fn terminal_showdown_stacks_based_with_rake_cap() {
        let deal = make_deal_p0_wins();
        // pot = 100.0, 5% rake, cap 3.0 -> rake = min(5.0, 3.0) = 3.0
        // stacks = [0.0, 0.0], starting_stack = 50.0
        // Winner (player 0): (0 + 100 - 3) - 50 = 47.0
        let v = terminal_value(
            TerminalKind::Showdown, 100.0,
            &[0.0, 0.0], 50.0,
            0, &deal, 0.05, 3.0,
        );
        assert!((v - 47.0).abs() < 1e-10, "Winner with capped rake should be 47.0, got {v}");
    }

    #[test]
    fn terminal_tie_stacks_based_with_rake() {
        let deal = make_deal(); // equal hands (tie)
        // pot = 10.0, 5% rake, no cap -> rake = 0.5
        // stacks = [45.0, 45.0], starting_stack = 50.0
        // Tie: (45 + 5 - 0.25) - 50 = -0.25
        let v = terminal_value(
            TerminalKind::Showdown, 10.0,
            &[45.0, 45.0], 50.0,
            0, &deal, 0.05, 0.0,
        );
        assert!((v - (-0.25)).abs() < 1e-10, "Tie with rake EV should be -0.25, got {v}");
    }

    #[test]
    fn terminal_stacks_asymmetric_fold() {
        let deal = make_deal();
        // Asymmetric: SB raised 3bb (3.0), BB called. stacks differ.
        // stacks = [47.0, 47.0], pot = 6.0, starting_stack = 50.0
        // But then BB bets 4, SB folds.
        // stacks = [47.0, 43.0], pot = 10.0, starting_stack = 50.0
        // Winner (BB = 1): (43 + 10) - 50 = 3.0
        // Loser (SB = 0): 47 - 50 = -3.0
        let v = terminal_value(
            TerminalKind::Fold { winner: 1 }, 10.0,
            &[47.0, 43.0], 50.0,
            1, &deal, 0.0, 0.0,
        );
        assert!((v - 3.0).abs() < 1e-10, "Winner EV should be 3.0, got {v}");
        let v = terminal_value(
            TerminalKind::Fold { winner: 1 }, 10.0,
            &[47.0, 43.0], 50.0,
            0, &deal, 0.0, 0.0,
        );
        assert!((v - (-3.0)).abs() < 1e-10, "Loser EV should be -3.0, got {v}");
    }

    /// Helper: shorthand card constructor.
    fn c(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    // ── Bucket file lookup tests ─────────────────────────────────────

    #[test]
    fn equity_fallback_aa_high_bucket() {
        let all = AllBuckets::new(
            [169, 400, 100, 1000],
            [None, None, None, None],
        );
        let deal = Deal {
            hole_cards: [
                [c(Value::Ace, Suit::Spade), c(Value::Ace, Suit::Heart)],
                [c(Value::Seven, Suit::Club), c(Value::Two, Suit::Diamond)],
            ],
            board: [
                c(Value::King, Suit::Diamond),
                c(Value::Seven, Suit::Spade),
                c(Value::Two, Suit::Heart),
                c(Value::Three, Suit::Club),
                c(Value::Eight, Suit::Diamond),
            ],
        };

        let b = all.precompute_buckets(&deal);
        let flop_bucket_aa = b[0][1];
        let turn_bucket_aa = b[0][2];

        // Buckets should be valid
        assert!(flop_bucket_aa < 400, "flop bucket in range");
        assert!(turn_bucket_aa < 100, "turn bucket in range");
    }

    #[test]
    fn get_bucket_equity_fallback() {
        let all = AllBuckets::new(
            [169, 10, 10, 10],
            [None, None, None, None],
        );

        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Ten, Suit::Club),
        ];

        let bucket = all.get_bucket(
            Street::River,
            [
                Card::new(Value::Two, Suit::Heart),
                Card::new(Value::Three, Suit::Heart),
            ],
            &board,
        );
        // Should return a valid bucket index in [0, 10).
        assert!(bucket < 10, "bucket should be < 10, got {bucket}");
    }

    // ── Bucket file lookup tests ─────────────────────────────────────

    /// Build a minimal BucketFile with known boards and bucket assignments
    /// for testing lookup_bucket.
    fn make_test_bucket_file(
        street: Street,
        board_cards: &[Vec<Card>],
        bucket_count: u16,
    ) -> BucketFile {
        use crate::blueprint_v2::bucket_file::{BucketFileHeader, PackedBoard};
        use crate::blueprint_v2::cluster_pipeline::canonical_key;

        let boards: Vec<PackedBoard> = board_cards
            .iter()
            .map(|b| {
                use crate::abstraction::isomorphism::CanonicalBoard;
                let canonical = CanonicalBoard::from_cards(b).unwrap();
                canonical_key(&canonical.cards)
            })
            .collect();

        let combos_per_board = 1326_u16;
        let total = boards.len() * combos_per_board as usize;
        // Assign deterministic buckets: bucket = (board_idx * 100 + combo_idx) % bucket_count
        let mut buckets = Vec::with_capacity(total);
        for board_idx in 0..boards.len() {
            for combo_idx in 0..combos_per_board as usize {
                buckets.push(((board_idx * 100 + combo_idx) % bucket_count as usize) as u16);
            }
        }

        BucketFile {
            header: BucketFileHeader {
                street,
                bucket_count,
                board_count: boards.len() as u32,
                combos_per_board,
                version: 2,
            },
            boards,
            buckets,
        }
    }

    #[test]
    fn lookup_bucket_returns_bucket_from_file() {
        use crate::blueprint_v2::cluster_pipeline::{canonical_key, combo_index};
        use crate::abstraction::isomorphism::CanonicalBoard;

        let flop_cards = vec![
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let bf = make_test_bucket_file(Street::Flop, &[flop_cards.clone()], 50);

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, Some(bf), None, None],
        );

        let hole = [c(Value::Queen, Suit::Spade), c(Value::Jack, Suit::Club)];
        let bucket = all.lookup_bucket(1, hole, &flop_cards);

        // Verify the bucket matches what the file would return:
        let canonical = CanonicalBoard::from_cards(&flop_cards).unwrap();
        let packed = canonical_key(&canonical.cards);
        let board_map = all.board_maps[1].as_ref().unwrap();
        let board_idx = board_map[&packed];
        let (c0, c1) = canonical.canonicalize_holding(hole[0], hole[1]);
        let ci = combo_index(c0, c1);
        let expected = all.bucket_files[1].as_ref().unwrap().get_bucket(board_idx, ci);

        assert_eq!(bucket, expected, "lookup_bucket should return the bucket file value");
    }

    #[test]
    fn lookup_bucket_falls_back_to_equity_without_file() {
        let all = AllBuckets::new(
            [169, 10, 10, 10],
            [None, None, None, None],
        );

        let flop = [
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let hole = [c(Value::Queen, Suit::Spade), c(Value::Jack, Suit::Club)];

        let bucket = all.lookup_bucket(1, hole, &flop);
        // Should fall back to equity-based bucketing: valid in [0, 10)
        assert!(bucket < 10, "equity fallback bucket should be < 10, got {bucket}");

        // Cross-check with manual equity computation
        let equity = crate::showdown_equity::compute_equity(hole, &flop);
        let expected = ((equity * 10.0) as u16).min(9);
        assert_eq!(bucket, expected, "should match equity-based bucketing");
    }

    #[test]
    fn precompute_buckets_uses_bucket_file() {
        let flop_cards = vec![
            c(Value::Two, Suit::Club),
            c(Value::Three, Suit::Diamond),
            c(Value::Four, Suit::Club),
        ];
        let turn_cards = vec![
            c(Value::Two, Suit::Club),
            c(Value::Three, Suit::Diamond),
            c(Value::Four, Suit::Club),
            c(Value::Five, Suit::Diamond),
        ];
        let river_cards = vec![
            c(Value::Two, Suit::Club),
            c(Value::Three, Suit::Diamond),
            c(Value::Four, Suit::Club),
            c(Value::Five, Suit::Diamond),
            c(Value::Six, Suit::Heart),
        ];

        let flop_bf = make_test_bucket_file(Street::Flop, &[flop_cards], 50);
        let turn_bf = make_test_bucket_file(Street::Turn, &[turn_cards], 30);
        let river_bf = make_test_bucket_file(Street::River, &[river_cards], 20);

        let all = AllBuckets::new(
            [169, 50, 30, 20],
            [None, Some(flop_bf), Some(turn_bf), Some(river_bf)],
        );

        let deal = make_deal();
        let result = all.precompute_buckets(&deal);

        // Preflop should be canonical hand index
        for player in 0..2 {
            let hand = crate::hands::CanonicalHand::from_cards(
                deal.hole_cards[player][0],
                deal.hole_cards[player][1],
            );
            let expected_preflop = (hand.index() as u16).min(168);
            assert_eq!(result[player][0], expected_preflop, "preflop bucket for player {player}");
        }

        // Postflop buckets should be valid
        for player in 0..2 {
            assert!(result[player][1] < 50, "flop bucket valid for player {player}");
            assert!(result[player][2] < 30, "turn bucket valid for player {player}");
            assert!(result[player][3] < 20, "river bucket valid for player {player}");
        }
    }

    #[test]
    fn get_bucket_uses_bucket_file_for_postflop() {
        let flop_cards = vec![
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let bf = make_test_bucket_file(Street::Flop, &[flop_cards.clone()], 50);

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, Some(bf), None, None],
        );

        let hole = [c(Value::Queen, Suit::Spade), c(Value::Jack, Suit::Club)];
        let bucket_via_get = all.get_bucket(Street::Flop, hole, &flop_cards);
        let bucket_via_lookup = all.lookup_bucket(1, hole, &flop_cards);

        assert_eq!(bucket_via_get, bucket_via_lookup,
            "get_bucket and lookup_bucket should agree for flop");
    }

    #[test]
    fn get_bucket_preflop_returns_canonical_hand_index() {
        let all = AllBuckets::new(
            [169, 10, 10, 10],
            [None, None, None, None],
        );

        let hole = [c(Value::Ace, Suit::Spade), c(Value::King, Suit::Spade)];
        let bucket = all.get_bucket(Street::Preflop, hole, &[]);

        let hand = crate::hands::CanonicalHand::from_cards(hole[0], hole[1]);
        let expected = (hand.index() as u16).min(168);
        assert_eq!(bucket, expected);
    }

    #[test]
    fn new_allbuckets_builds_board_maps() {
        let flop_cards = vec![
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let bf = make_test_bucket_file(Street::Flop, &[flop_cards], 50);

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, Some(bf), None, None],
        );

        // Preflop has no bucket file -> no board map
        assert!(all.board_maps[0].is_none());
        // Flop has a bucket file with boards -> has board map
        assert!(all.board_maps[1].is_some());
        assert_eq!(all.board_maps[1].as_ref().unwrap().len(), 1);
        // Turn/River have no bucket files -> no board maps
        assert!(all.board_maps[2].is_none());
        assert!(all.board_maps[3].is_none());
    }

    // ── Per-flop bucket lookup tests ─────────────────────────────────

    /// Helper: build a PerFlopBucketFile for a given canonical flop with
    /// deterministic bucket assignments, save it to the given directory,
    /// and return the flop cards used.
    fn save_test_per_flop_file(
        dir: &std::path::Path,
        flop: [Card; 3],
        turn_cards: Vec<Card>,
        river_cards_per_turn: Vec<Vec<Card>>,
        turn_bucket_count: u16,
        river_bucket_count: u16,
        file_index: usize,
    ) -> crate::blueprint_v2::per_flop_bucket_file::PerFlopBucketFile {
        use crate::blueprint_v2::per_flop_bucket_file::PerFlopBucketFile;

        let num_turns = turn_cards.len();
        // Deterministic turn buckets: bucket = (turn_idx * 100 + combo_idx) % turn_bucket_count
        let mut turn_buckets = vec![0u16; num_turns * 1326];
        for t in 0..num_turns {
            for ci in 0..1326 {
                turn_buckets[t * 1326 + ci] = ((t * 100 + ci) % turn_bucket_count as usize) as u16;
            }
        }

        // Deterministic river buckets
        let mut river_buckets_per_turn = Vec::new();
        for (t, rivers) in river_cards_per_turn.iter().enumerate() {
            let num_rivers = rivers.len();
            let mut rb = vec![0u16; num_rivers * 1326];
            for r in 0..num_rivers {
                for ci in 0..1326 {
                    rb[r * 1326 + ci] = ((t * 50 + r * 10 + ci) % river_bucket_count as usize) as u16;
                }
            }
            river_buckets_per_turn.push(rb);
        }

        let pf = PerFlopBucketFile {
            flop_cards: flop,
            turn_bucket_count,
            river_bucket_count,
            turn_cards,
            turn_buckets,
            river_cards_per_turn,
            river_buckets_per_turn,
        };

        let path = dir.join(format!("flop_{file_index:04}.buckets"));
        pf.save(&path).expect("save per-flop file");
        pf
    }

    #[test]
    fn with_per_flop_dir_scans_files() {
        let dir = tempfile::tempdir().expect("tempdir");
        let flop = [
            c(Value::Queen, Suit::Spade),
            c(Value::Jack, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let turn_card = c(Value::Ace, Suit::Club);
        let river_card = c(Value::Ten, Suit::Club);

        save_test_per_flop_file(
            dir.path(), flop,
            vec![turn_card],
            vec![vec![river_card]],
            10, 10, 0,
        );

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, None, None, None],
        ).with_per_flop_dir(dir.path().to_path_buf());

        assert!(all.per_flop_dir.is_some());
        assert!(all.flop_index_map.is_some());
        let index = all.flop_index_map.as_ref().unwrap();
        assert_eq!(index.len(), 1, "should have 1 flop indexed");
    }

    #[test]
    fn per_flop_turn_bucket_lookup() {
        use crate::abstraction::isomorphism::CanonicalBoard;

        let dir = tempfile::tempdir().expect("tempdir");

        // Use canonical flop cards (already in canonical form for simplicity)
        let flop = [
            c(Value::Queen, Suit::Spade),
            c(Value::Jack, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let turn_card = c(Value::Ace, Suit::Club);
        let river_card = c(Value::Ten, Suit::Club);

        let pf = save_test_per_flop_file(
            dir.path(), flop,
            vec![turn_card],
            vec![vec![river_card]],
            10, 10, 0,
        );

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, None, None, None],
        ).with_per_flop_dir(dir.path().to_path_buf());

        // Build a turn board: flop + turn_card
        let turn_board = [flop[0], flop[1], flop[2], turn_card];

        // Use hole cards that won't conflict with board
        let hole = [c(Value::King, Suit::Club), c(Value::Nine, Suit::Diamond)];

        let bucket = all.get_bucket(Street::Turn, hole, &turn_board);

        // Compute expected: canonicalize the flop, find turn card, get combo_index
        let canonical_flop = CanonicalBoard::from_cards(&flop[..]).unwrap();
        let (h0, h1) = canonical_flop.canonicalize_holding(hole[0], hole[1]);
        let ci = combo_index(h0, h1) as usize;

        // turn_idx=0, expected = (0 * 100 + ci) % 10
        let expected = (ci % 10) as u16;
        assert_eq!(bucket, expected, "turn bucket from per-flop file");
    }

    #[test]
    fn per_flop_river_bucket_lookup() {
        use crate::abstraction::isomorphism::CanonicalBoard;

        let dir = tempfile::tempdir().expect("tempdir");

        let flop = [
            c(Value::Queen, Suit::Spade),
            c(Value::Jack, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        let turn_card = c(Value::Ace, Suit::Club);
        let river_card = c(Value::Ten, Suit::Club);

        let _pf = save_test_per_flop_file(
            dir.path(), flop,
            vec![turn_card],
            vec![vec![river_card]],
            10, 10, 0,
        );

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, None, None, None],
        ).with_per_flop_dir(dir.path().to_path_buf());

        // Build a river board: flop + turn + river
        let river_board = [flop[0], flop[1], flop[2], turn_card, river_card];

        let hole = [c(Value::King, Suit::Club), c(Value::Nine, Suit::Diamond)];

        let bucket = all.get_bucket(Street::River, hole, &river_board);

        // Compute expected
        let canonical_flop = CanonicalBoard::from_cards(&flop[..]).unwrap();
        let (h0, h1) = canonical_flop.canonicalize_holding(hole[0], hole[1]);
        let ci = combo_index(h0, h1) as usize;

        // turn_idx=0, river_idx=0, expected = (0 * 50 + 0 * 10 + ci) % 10
        let expected = (ci % 10) as u16;
        assert_eq!(bucket, expected, "river bucket from per-flop file");
    }

    #[test]
    fn per_flop_precompute_buckets() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Use the same board cards as make_deal()
        let deal = make_deal();
        let flop = [deal.board[0], deal.board[1], deal.board[2]];
        let turn_card = deal.board[3];
        let river_card = deal.board[4];

        save_test_per_flop_file(
            dir.path(), flop,
            vec![turn_card],
            vec![vec![river_card]],
            10, 10, 0,
        );

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, None, None, None],
        ).with_per_flop_dir(dir.path().to_path_buf());

        let result = all.precompute_buckets(&deal);

        // Turn and river buckets should be valid (in [0, 10))
        for player in 0..2 {
            assert!(result[player][2] < 10, "turn bucket valid for player {player}: {}", result[player][2]);
            assert!(result[player][3] < 10, "river bucket valid for player {player}: {}", result[player][3]);
        }
    }

    #[test]
    fn per_flop_flop_still_uses_global_file() {
        let dir = tempfile::tempdir().expect("tempdir");

        let flop = [
            c(Value::Queen, Suit::Spade),
            c(Value::Jack, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];

        // Save a per-flop file
        save_test_per_flop_file(
            dir.path(), flop,
            vec![c(Value::Ace, Suit::Club)],
            vec![vec![c(Value::Ten, Suit::Club)]],
            10, 10, 0,
        );

        // Also provide a global flop bucket file
        let flop_bf = make_test_bucket_file(
            Street::Flop,
            &[vec![flop[0], flop[1], flop[2]]],
            50,
        );

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, Some(flop_bf), None, None],
        ).with_per_flop_dir(dir.path().to_path_buf());

        let hole = [c(Value::King, Suit::Club), c(Value::Nine, Suit::Diamond)];

        // Flop lookup should use global bucket file, not per-flop
        let bucket = all.get_bucket(Street::Flop, hole, &[flop[0], flop[1], flop[2]]);
        assert!(bucket < 50, "flop bucket should use global file (< 50), got {bucket}");
    }

    #[test]
    fn per_flop_fallback_when_flop_not_found() {
        let dir = tempfile::tempdir().expect("tempdir");

        // Save a per-flop file for one flop
        let flop = [
            c(Value::Queen, Suit::Spade),
            c(Value::Jack, Suit::Heart),
            c(Value::Two, Suit::Diamond),
        ];
        save_test_per_flop_file(
            dir.path(), flop,
            vec![c(Value::Ace, Suit::Club)],
            vec![vec![c(Value::Ten, Suit::Club)]],
            10, 10, 0,
        );

        let all = AllBuckets::new(
            [169, 50, 10, 10],
            [None, None, None, None],
        ).with_per_flop_dir(dir.path().to_path_buf());

        // Try a different flop that has no per-flop file
        let other_flop = [
            c(Value::Ace, Suit::Spade),
            c(Value::King, Suit::Heart),
            c(Value::Three, Suit::Diamond),
        ];
        let turn_board = [other_flop[0], other_flop[1], other_flop[2], c(Value::Four, Suit::Club)];

        let hole = [c(Value::Seven, Suit::Club), c(Value::Eight, Suit::Diamond)];

        // Should fall back to equity-based bucketing (returns valid bucket in [0,10))
        let bucket = all.get_bucket(Street::Turn, hole, &turn_board);
        assert!(bucket < 10, "should fall back to equity bucketing, got {bucket}");
    }

    // ── baseline_corrected_value unit tests ──────────────────────────────

    #[test]
    fn baseline_corrected_value_zero_baselines_returns_sampled() {
        // With zero baselines, formula degenerates to v_sampled.
        let strategy = &[0.5, 0.3, 0.2];
        let baselines = &[0.0, 0.0, 0.0];
        let v = baseline_corrected_value(strategy, baselines, 1, 5.0);
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn baseline_corrected_value_perfect_baselines() {
        // When baselines exactly equal the sampled value, v = sum(s*b).
        let strategy = &[0.6, 0.4];
        let baselines = &[3.0, 3.0];
        let v = baseline_corrected_value(strategy, baselines, 0, 3.0);
        // sum(s*b) = 0.6*3 + 0.4*3 = 3.0; + (3.0 - 3.0) = 3.0
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn baseline_corrected_value_nonzero_correction() {
        let strategy = &[0.5, 0.5];
        let baselines = &[2.0, 4.0];
        let v = baseline_corrected_value(strategy, baselines, 1, 6.0);
        // sum(s*b) = 0.5*2 + 0.5*4 = 3.0; + (6.0 - 4.0) = 5.0
        assert!((v - 5.0).abs() < 1e-10);
    }

    #[test]
    fn baseline_corrected_value_single_action() {
        // With one action, strategy = [1.0], always sampled.
        let strategy = &[1.0];
        let baselines = &[10.0];
        let v = baseline_corrected_value(strategy, baselines, 0, 7.0);
        // sum(s*b) = 10.0; + (7.0 - 10.0) = 7.0
        assert!((v - 7.0).abs() < 1e-10);
    }
}
