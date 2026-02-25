//! Postflop abstraction: precomputed data + CFR traversal for the abstracted postflop model.
//!
//! Combines board abstraction, hand buckets, and postflop trees into a single
//! structure that the preflop solver queries at showdown terminals.
//!
//! # Architecture
//!
//! ```text
//! PreflopSolver hits Showdown terminal
//!   → PostflopAbstraction::evaluate(spr_idx, hero_hand, opp_hand, pot)
//!     → look up hand buckets per street/texture
//!     → traverse postflop tree with bucket-level regret matching
//!     → return expected value
//! ```

use serde::{Deserialize, Serialize};

use super::equity::EquityTable;
use super::hand_buckets::{BucketEquity, StreetBuckets, StreetEquity, TransitionMatrices};
use super::postflop_bucketed::build_bucketed;
use super::postflop_mccfr::build_mccfr;
use super::postflop_model::{PostflopModelConfig, PostflopSolveType};
use super::postflop_tree::{PostflopNode, PostflopTerminalType, PostflopTree};
use crate::abstraction::Street;
use crate::poker::Card;

/// Number of canonical hole-card pairs (169).
pub const NUM_CANONICAL_HANDS: usize = 169;

/// All precomputed postflop data needed by the preflop solver.
pub struct PostflopAbstraction {
    /// Flop bucket assignments (per-flop, needed at runtime).
    pub buckets: StreetBuckets,
    /// Per-flop equity tables (only populated for diagnostics/cached builds).
    pub street_equity: Option<StreetEquity>,
    /// Per-flop transition matrices (only populated for diagnostics/cached builds).
    pub transitions: Option<TransitionMatrices>,
    /// Single shared tree template at `config.primary_spr()`.
    pub tree: PostflopTree,
    /// Precomputed EV table from solved postflop game.
    pub values: PostflopValues,
    /// Hand-averaged EV: `hand_avg_values[hero_pos * 169 * 169 + hero_hand * 169 + opp_hand]`.
    ///
    /// Precomputed average of `values.get_by_flop` across all flops for each
    /// `(hero_pos, hero_hand, opp_hand)` triple. This turns O(num_flops) showdown
    /// lookups into O(1) during preflop solving.
    pub hand_avg_values: Vec<f64>,
    /// The fixed SPR used for all postflop solves.
    pub spr: f64,
    /// The flop boards used to build this abstraction (for diagnostics/display).
    pub flops: Vec<[Card; 3]>,
}

/// Precomputed EV table: `values[flop_idx][hero_pos][hero_bucket][opp_bucket]` → EV fraction.
///
/// Stored as a flat `Vec<f64>` indexed by
/// `flop_idx * 2 * n * n + hero_pos * n * n + hero_bucket * n + opp_bucket`.
#[derive(Serialize, Deserialize)]
pub struct PostflopValues {
    pub(crate) values: Vec<f64>,
    pub(crate) num_buckets: usize,
    pub(crate) num_flops: usize,
}

impl PostflopValues {
    /// Number of flop slots in the value table.
    #[must_use]
    pub fn num_flops(&self) -> usize {
        self.num_flops
    }

    /// Number of entries in the value table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the value table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Create an empty value table (for testing).
    #[must_use]
    pub fn empty() -> Self {
        Self {
            values: Vec::new(),
            num_buckets: 0,
            num_flops: 0,
        }
    }

    /// Look up postflop EV by flop index.
    #[inline]
    #[must_use]
    pub fn get_by_flop(&self, flop_idx: usize, hero_pos: u8, hero_bucket: u16, opp_bucket: u16) -> f64 {
        let n = self.num_buckets;
        let idx = flop_idx * 2 * n * n + (hero_pos as usize) * n * n + (hero_bucket as usize) * n + opp_bucket as usize;
        self.values.get(idx).copied().unwrap_or(0.0)
    }

    /// Create from raw data.
    pub(crate) fn from_raw(values: Vec<f64>, num_buckets: usize, num_flops: usize) -> Self {
        Self { values, num_buckets, num_flops }
    }
}

/// Maps `(node_idx, bucket)` → flat buffer offset for ONE tree.
///
/// Each decision node reserves `num_buckets × num_actions` slots.
/// The bucket count varies by the street of the node.
pub(crate) struct PostflopLayout {
    entries: Vec<NodeEntry>,
    /// Total buffer size for this tree.
    pub total_size: usize,
}

#[derive(Clone, Copy)]
pub(crate) struct NodeEntry {
    pub(crate) offset: usize,
    pub(crate) num_actions: usize,
    pub(crate) street: Street,
}

impl PostflopLayout {
    /// Build the layout from a single postflop tree and bucket counts.
    pub(crate) fn build(
        tree: &PostflopTree,
        node_streets: &[Street],
        num_flop_buckets: usize,
        num_turn_buckets: usize,
        num_river_buckets: usize,
    ) -> Self {
        let mut offset = 0;

        let entries: Vec<NodeEntry> = tree
            .nodes
            .iter()
            .enumerate()
            .map(|(i, node)| {
                let num_actions = match node {
                    PostflopNode::Decision { children, .. } => children.len(),
                    _ => 0,
                };
                let street = node_streets[i];
                let num_buckets = buckets_for_street(
                    street,
                    num_flop_buckets,
                    num_turn_buckets,
                    num_river_buckets,
                );
                let entry = NodeEntry {
                    offset,
                    num_actions,
                    street,
                };
                offset += num_buckets * num_actions;
                entry
            })
            .collect();

        Self {
            entries,
            total_size: offset,
        }
    }

    /// Returns `(base_offset, num_actions)` for a given `(node_idx, bucket)`.
    ///
    /// The caller indexes into the flat buffer as `base + action_idx`.
    #[inline]
    #[must_use]
    pub fn slot(&self, node_idx: u32, bucket: u16) -> (usize, usize) {
        let entry = &self.entries[node_idx as usize];
        let base = entry.offset + (bucket as usize) * entry.num_actions;
        (base, entry.num_actions)
    }

    /// Street for a given node.
    #[inline]
    #[must_use]
    pub fn street(&self, node_idx: u32) -> Street {
        self.entries[node_idx as usize].street
    }

    /// Number of nodes in the layout.
    pub(crate) fn num_nodes(&self) -> usize {
        self.entries.len()
    }

    /// Get entry offset and `num_actions` for a node.
    pub(crate) fn entry(&self, node_idx: usize) -> (usize, usize) {
        let e = &self.entries[node_idx];
        (e.offset, e.num_actions)
    }
}

/// Number of hand buckets for a given street.
pub(crate) fn buckets_for_street(
    street: Street,
    flop: usize,
    turn: usize,
    river: usize,
) -> usize {
    match street {
        Street::Preflop | Street::Flop => flop,
        Street::Turn => turn,
        Street::River => river,
    }
}

/// Annotate each node in a postflop tree with its street.
///
/// Walks the tree from root, tracking the current street.
/// Chance nodes transition to their `next_street` for their children.
pub(crate) fn annotate_streets(tree: &PostflopTree) -> Vec<Street> {
    let mut streets = vec![Street::Flop; tree.nodes.len()];
    annotate_recursive(&tree.nodes, 0, Street::Flop, &mut streets);
    streets
}

pub(crate) fn annotate_recursive(
    nodes: &[PostflopNode],
    idx: u32,
    current_street: Street,
    out: &mut [Street],
) {
    out[idx as usize] = current_street;
    match &nodes[idx as usize] {
        PostflopNode::Decision { children, .. } => {
            for &child in children {
                annotate_recursive(nodes, child, current_street, out);
            }
        }
        PostflopNode::Chance {
            street, children, ..
        } => {
            for &child in children {
                annotate_recursive(nodes, child, *street, out);
            }
        }
        PostflopNode::Terminal { .. } => {}
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Construction
// ──────────────────────────────────────────────────────────────────────────────

/// Error during postflop abstraction construction.
#[derive(Debug, thiserror::Error)]
pub enum PostflopAbstractionError {
    #[error("postflop tree: {0}")]
    Tree(#[from] super::postflop_tree::PostflopTreeError),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
}

/// Stage within a single flop's streaming pipeline.
#[derive(Debug, Clone)]
pub enum FlopStage {
    /// Computing buckets, equity tables, transition matrices.
    /// `step` counts completed sub-steps (0..6): flop buckets, flop equity,
    /// turn buckets, turn equity, river buckets, river equity.
    Bucketing { step: u8, total_steps: u8 },
    /// CFR solve in progress.
    Solving {
        iteration: usize,
        max_iterations: usize,
        delta: f64,
    },
    /// Flop complete — signal to clear the bar.
    Done,
}

/// Progress report during postflop abstraction construction.
#[derive(Debug, Clone)]
pub enum BuildPhase {
    /// Per-flop streaming progress.
    FlopProgress {
        flop_name: String,
        stage: FlopStage,
    },
    /// Extracting EV histograms from converged strategy.
    ExtractingEv(usize, usize),
    /// Re-clustering with EV features.
    Rebucketing(u16, u16),
    /// Computing value table from converged strategy.
    ComputingValues,
    /// Aggregate progress for MCCFR: how many flops have completed solve + extraction.
    MccfrFlopsCompleted { completed: usize, total: usize },
}

impl std::fmt::Display for BuildPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlopProgress { flop_name, stage } => match stage {
                FlopStage::Bucketing { step, total_steps } => write!(f, "Flop '{flop_name}' Bucketing ({step}/{total_steps})"),
                FlopStage::Solving { iteration, max_iterations, delta } =>
                    write!(f, "Flop '{flop_name}' CFR \u{03b4}={delta:.4} ({iteration}/{max_iterations})"),
                FlopStage::Done => write!(f, "Flop '{flop_name}' Done"),
            },
            Self::ExtractingEv(done, total) => write!(f, "EV histograms ({done}/{total})"),
            Self::Rebucketing(round, total) => write!(f, "Rebucketing ({round}/{total})"),
            Self::ComputingValues => write!(f, "Computing values"),
            Self::MccfrFlopsCompleted { completed, total } => write!(f, "MCCFR Solving ({completed}/{total} flops)"),
        }
    }
}


impl PostflopAbstraction {
    /// Build all precomputed postflop data from configuration.
    ///
    /// This is expensive (minutes) — called once before training begins.
    /// Calls `on_progress(phase)` at the start of each build phase.
    ///
    /// When `rebucket_rounds > 1`, runs an outer loop:
    /// 1. Solve all flops with current bucket assignments
    /// 2. Extract EV histograms from the converged strategy
    /// 3. Re-cluster flop buckets using EV features
    /// 4. Recompute pairwise equity for the new flop assignments
    /// 5. Repeat from (1) until final round, then compute final values
    ///
    /// # Errors
    ///
    /// Returns an error if hand bucketing or tree building fails.
    pub fn build(
        config: &PostflopModelConfig,
        _equity_table: Option<&EquityTable>,
        _cache_base: Option<&std::path::Path>,
        on_progress: impl Fn(BuildPhase) + Sync,
    ) -> Result<Self, PostflopAbstractionError> {
        let flops = if let Some(ref names) = config.fixed_flops {
            crate::preflop::ehs::parse_flops(names)
                .map_err(PostflopAbstractionError::InvalidConfig)?
        } else {
            crate::preflop::ehs::sample_canonical_flops(config.max_flop_boards)
        };

        // Build tree + layout (global, shared across all flops/rounds)
        let tree = PostflopTree::build_with_spr(config, config.primary_spr())?;
        let node_streets = annotate_streets(&tree);
        let nfb = config.num_hand_buckets_flop as usize;
        let ntb = config.num_hand_buckets_turn as usize;
        let nrb = config.num_hand_buckets_river as usize;
        let layout = match config.solve_type {
            PostflopSolveType::Bucketed => PostflopLayout::build(&tree, &node_streets, nfb, ntb, nrb),
            PostflopSolveType::Mccfr => PostflopLayout::build(&tree, &node_streets, nfb, nfb, nfb),
        };

        let (buckets, values) = match config.solve_type {
            PostflopSolveType::Bucketed => {
                build_bucketed(
                    config, &tree, &layout, &node_streets, &flops, &on_progress,
                )
            }
            PostflopSolveType::Mccfr => {
                build_mccfr(config, &tree, &layout, &node_streets, &flops, &on_progress)
            }
        };

        on_progress(BuildPhase::ComputingValues);

        let hand_avg_values = compute_hand_avg_values(&buckets, &values);

        Ok(Self {
            buckets,
            street_equity: None,
            transitions: None,
            tree,
            values,
            hand_avg_values,
            spr: config.primary_spr(),
            flops,
        })
    }

    /// Build `PostflopAbstraction` using pre-cached values, skipping the solve phase.
    ///
    /// Rebuilds trees (instant) and uses the provided `PostflopValues` directly.
    ///
    /// # Errors
    ///
    /// Returns an error if tree building fails.
    pub fn build_from_cached(
        config: &PostflopModelConfig,
        buckets: StreetBuckets,
        street_equity: Option<StreetEquity>,
        transitions: Option<TransitionMatrices>,
        values: PostflopValues,
        hand_avg_values: Vec<f64>,
        flops: Vec<[Card; 3]>,
    ) -> Result<Self, PostflopAbstractionError> {
        let tree = PostflopTree::build_with_spr(config, config.primary_spr())?;
        Ok(Self {
            buckets,
            street_equity,
            transitions,
            tree,
            values,
            hand_avg_values,
            spr: config.primary_spr(),
            flops,
        })
    }

    /// Look up precomputed hand-averaged postflop EV.
    ///
    /// Returns the average EV fraction across all flops for the given
    /// `(hero_pos, hero_hand, opp_hand)` triple. O(1).
    #[inline]
    #[must_use]
    pub fn avg_ev(&self, hero_pos: u8, hero_hand: usize, opp_hand: usize) -> f64 {
        let n = self.num_avg_hands();
        let idx = (hero_pos as usize) * n * n + hero_hand * n + opp_hand;
        self.hand_avg_values.get(idx).copied().unwrap_or(0.0)
    }

    /// Number of hands in the hand-averaged value table (169 in production).
    #[inline]
    fn num_avg_hands(&self) -> usize {
        // Table is 2 * n * n, so n = isqrt(len / 2)
        let half = self.hand_avg_values.len() / 2;
        if half == 0 { return 0; }
        let n = (half as f64).sqrt() as usize;
        debug_assert_eq!(n * n, half, "hand_avg_values has unexpected size");
        n
    }
}

/// Precompute hand-averaged postflop EV for all `(hero_pos, hero_hand, opp_hand)` triples.
///
/// Averages `values.get_by_flop(flop_idx, pos, hb, ob)` across all flops for each
/// canonical hand pair. Returns a flat `Vec<f64>` of size `2 × N × N` where
/// N = number of hands in the bucket table (169 in production).
/// Parallelized across hero hands with rayon.
#[allow(clippy::cast_precision_loss)]
pub fn compute_hand_avg_values(buckets: &StreetBuckets, values: &PostflopValues) -> Vec<f64> {
    use rayon::prelude::*;

    let num_flops = buckets.num_flop_boards();
    let num_hands = if num_flops > 0 { buckets.flop[0].len() } else { 0 };
    let inv_flops = if num_flops > 0 { 1.0 / num_flops as f64 } else { 0.0 };

    // Allocate output: [pos0: N×N, pos1: N×N]
    let mut out = vec![0.0f64; 2 * num_hands * num_hands];

    // Parallel over hero_hand — each writes to a disjoint slice per position.
    let chunks: Vec<(usize, Vec<f64>)> = (0..num_hands)
        .into_par_iter()
        .map(|hero_hand| {
            let mut local = vec![0.0f64; 2 * num_hands];
            for flop_idx in 0..num_flops {
                let hb = buckets.flop_bucket_for_hand(hero_hand, flop_idx);
                for opp_hand in 0..num_hands {
                    let ob = buckets.flop_bucket_for_hand(opp_hand, flop_idx);
                    local[opp_hand] += values.get_by_flop(flop_idx, 0, hb, ob);
                    local[num_hands + opp_hand] += values.get_by_flop(flop_idx, 1, hb, ob);
                }
            }
            (hero_hand, local)
        })
        .collect();

    let n = num_hands;
    for (hero_hand, local) in chunks {
        for opp_hand in 0..n {
            out[hero_hand * n + opp_hand] = local[opp_hand] * inv_flops;
            out[n * n + hero_hand * n + opp_hand] = local[n + opp_hand] * inv_flops;
        }
    }

    out
}

// ──────────────────────────────────────────────────────────────────────────────
// Postflop pre-solve: run CFR to convergence, then extract value table
// ──────────────────────────────────────────────────────────────────────────────

/// Result of a single flop CFR solve.
#[allow(dead_code)] // final_delta and iterations_used read in tests
pub(crate) struct FlopSolveResult {
    pub strategy_sum: Vec<f64>,
    pub final_delta: f64,
    pub iterations_used: usize,
}

/// Element-wise `dst[i] += src[i]`.
pub(crate) fn add_buffers(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// Compute max strategy change between two regret buffers across all decision nodes.
///
/// For each decision node in the tree, iterates through every bucket's action slice,
/// applies regret matching to both the old and new regret buffers, and returns the
/// weighted average strategy delta.
///
/// Each (node, bucket) position is weighted by its total absolute regret mass,
/// so frequently-reached positions with large regrets dominate the metric while
/// rarely-reached positions with near-zero regrets are ignored.
pub(crate) fn weighted_avg_strategy_delta(
    old_regrets: &[f64],
    new_regrets: &[f64],
    layout: &PostflopLayout,
    tree: &PostflopTree,
) -> f64 {
    let mut weighted_sum = 0.0f64;
    let mut total_weight = 0.0f64;

    for (node_idx, node) in tree.nodes.iter().enumerate() {
        if let PostflopNode::Decision { children, .. } = node {
            let num_actions = children.len();
            if num_actions == 0 {
                continue;
            }

            let (node_offset, _) = layout.entry(node_idx);
            let node_end = if node_idx + 1 < layout.num_nodes() {
                layout.entry(node_idx + 1).0
            } else {
                old_regrets.len()
            };

            // Process each bucket's action slice.
            let mut pos = node_offset;
            let mut old_strat = vec![0.0f64; num_actions];
            let mut new_strat = vec![0.0f64; num_actions];
            while pos + num_actions <= node_end {
                // Weight = sum of absolute regrets at this position (using new regrets).
                let weight: f64 = new_regrets[pos..pos + num_actions]
                    .iter()
                    .map(|r| r.abs())
                    .sum();

                if weight > 0.0 {
                    regret_matching_into(old_regrets, pos, &mut old_strat);
                    regret_matching_into(new_regrets, pos, &mut new_strat);
                    let mut pos_delta = 0.0f64;
                    for i in 0..num_actions {
                        pos_delta = pos_delta.max((old_strat[i] - new_strat[i]).abs());
                    }
                    weighted_sum += pos_delta * weight;
                    total_weight += weight;
                }
                pos += num_actions;
            }
        }
    }

    if total_weight > 0.0 {
        weighted_sum / total_weight
    } else {
        0.0
    }
}

/// Normalize strategy sum into a probability distribution.
pub(crate) fn normalize_strategy_sum(strategy_sum: &[f64], start: usize, num_actions: usize) -> Vec<f64> {
    let total: f64 = (0..num_actions)
        .map(|i| strategy_sum.get(start + i).copied().unwrap_or(0.0).max(0.0))
        .sum();
    if total > 0.0 {
        (0..num_actions)
            .map(|i| strategy_sum.get(start + i).copied().unwrap_or(0.0).max(0.0) / total)
            .collect()
    } else {
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / num_actions as f64;
        vec![uniform; num_actions]
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shared helpers for CFR traversal and value evaluation
// ──────────────────────────────────────────────────────────────────────────────

pub(crate) const MAX_POSTFLOP_ACTIONS: usize = 8;

pub(crate) fn postflop_terminal_value(
    terminal_type: PostflopTerminalType,
    pot_fraction: f64,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    bucket_equity: &BucketEquity,
) -> f64 {
    match terminal_type {
        PostflopTerminalType::Fold { folder } => {
            if folder == hero_pos {
                -pot_fraction / 2.0
            } else {
                pot_fraction / 2.0
            }
        }
        PostflopTerminalType::Showdown => {
            let eq = f64::from(bucket_equity.get(hero_bucket as usize, opp_bucket as usize));
            eq * pot_fraction - pot_fraction / 2.0
        }
    }
}

/// Regret matching: normalize positive regrets into a strategy.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn regret_matching_into(regret_buf: &[f64], start: usize, out: &mut [f64]) {
    let num_actions = out.len();
    let mut positive_sum = 0.0f64;

    for (i, s) in out.iter_mut().enumerate() {
        let val = if start + i < regret_buf.len() {
            regret_buf[start + i]
        } else {
            0.0
        };
        if val > 0.0 {
            *s = val;
            positive_sum += val;
        } else {
            *s = 0.0;
        }
    }

    if positive_sum > 0.0 {
        for s in out.iter_mut() {
            *s /= positive_sum;
        }
    } else {
        let uniform = 1.0 / num_actions as f64;
        out.fill(uniform);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn buckets_for_street_returns_correct_counts() {
        assert_eq!(buckets_for_street(Street::Flop, 100, 200, 300), 100);
        assert_eq!(buckets_for_street(Street::Turn, 100, 200, 300), 200);
        assert_eq!(buckets_for_street(Street::River, 100, 200, 300), 300);
        assert_eq!(buckets_for_street(Street::Preflop, 100, 200, 300), 100);
    }

    #[timed_test]
    fn annotate_streets_marks_root_as_flop() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 5.0).unwrap();
        let streets = annotate_streets(&tree);
        assert_eq!(streets[0], Street::Flop);
    }

    #[timed_test]
    fn annotate_streets_chance_children_get_next_street() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            ..PostflopModelConfig::fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 5.0).unwrap();
        let streets = annotate_streets(&tree);

        for (i, node) in tree.nodes.iter().enumerate() {
            if let PostflopNode::Chance {
                street, children, ..
            } = node
            {
                for &child in children {
                    assert_eq!(
                        streets[child as usize], *street,
                        "chance child at {child} should be on street {street:?}"
                    );
                }
                let expected_parent = match street {
                    Street::Turn => Street::Flop,
                    Street::River => Street::Turn,
                    _ => continue,
                };
                assert_eq!(
                    streets[i], expected_parent,
                    "chance node at {i} should be on {expected_parent:?}"
                );
            }
        }
    }

    #[timed_test]
    fn postflop_terminal_fold_hero_loses() {
        let eq = BucketEquity {
            equity: vec![],
            num_buckets: 0,
        };
        let val = postflop_terminal_value(
            PostflopTerminalType::Fold { folder: 0 },
            2.0, 0, 0, 0, &eq,
        );
        assert!((val - (-1.0)).abs() < 1e-9, "hero fold: loses pot/2");
    }

    #[timed_test]
    fn postflop_terminal_fold_opponent_wins() {
        let eq = BucketEquity {
            equity: vec![],
            num_buckets: 0,
        };
        let val = postflop_terminal_value(
            PostflopTerminalType::Fold { folder: 1 },
            2.0, 0, 0, 0, &eq,
        );
        assert!((val - 1.0).abs() < 1e-9, "opp fold: hero wins pot/2");
    }

    #[timed_test]
    fn postflop_terminal_showdown_uses_bucket_equity() {
        let eq = BucketEquity {
            equity: vec![vec![0.7]],
            num_buckets: 1,
        };
        let val = postflop_terminal_value(
            PostflopTerminalType::Showdown,
            2.0, 0, 0, 0, &eq,
        );
        assert!((val - 0.4).abs() < 1e-6, "showdown: eq*pot - pot/2, got {val}");
    }

    #[timed_test]
    fn regret_matching_uniform_when_no_positive() {
        let buf = vec![-1.0, -2.0, -3.0];
        let mut out = [0.0; 3];
        regret_matching_into(&buf, 0, &mut out);
        for &v in &out {
            assert!((v - 1.0 / 3.0).abs() < 1e-9);
        }
    }

    #[timed_test]
    fn regret_matching_proportional_to_positive() {
        let buf = vec![3.0, 1.0, 0.0];
        let mut out = [0.0; 3];
        regret_matching_into(&buf, 0, &mut out);
        assert!((out[0] - 0.75).abs() < 1e-9);
        assert!((out[1] - 0.25).abs() < 1e-9);
        assert!((out[2] - 0.0).abs() < 1e-9);
    }

    #[timed_test]
    fn layout_slot_returns_valid_offset() {
        let config = PostflopModelConfig {
            bet_sizes: vec![1.0],
            max_raises_per_street: 0,
            ..PostflopModelConfig::fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 5.0).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 10, 10, 10);

        assert!(layout.total_size > 0, "layout should have nonzero size");
        let (base, num_actions) = layout.slot(0, 0);
        assert!(num_actions > 0);
        assert!(base + num_actions <= layout.total_size);
    }

    #[timed_test]
    fn postflop_values_get_by_flop_index() {
        let num_flops = 2;
        let num_buckets = 2;
        let total = num_flops * 2 * num_buckets * num_buckets;
        let mut values = vec![0.0; total];
        let idx = 1 * 2 * 4 + 0 * 4 + 1 * 2 + 0;
        values[idx] = 0.42;
        let pv = PostflopValues { values, num_buckets, num_flops };
        assert!((pv.get_by_flop(1, 0, 1, 0) - 0.42).abs() < 1e-9);
    }

    #[timed_test]
    fn weighted_avg_strategy_delta_identical_buffers_is_zero() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 10, 10, 10);
        let buf = vec![1.0f64; layout.total_size];
        let delta = weighted_avg_strategy_delta(&buf, &buf, &layout, &tree);
        assert!(
            delta.abs() < 1e-12,
            "identical buffers should have zero delta, got {delta}"
        );
    }

    #[timed_test]
    fn weighted_avg_strategy_delta_detects_change() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 10, 10, 10);
        let old = vec![1.0f64; layout.total_size];
        let mut new = old.clone();
        // Flip first slot to create a strategy change.
        if layout.total_size >= 2 {
            new[0] = 100.0;
            new[1] = 0.0;
        }
        let delta = weighted_avg_strategy_delta(&old, &new, &layout, &tree);
        assert!(delta > 0.0, "different buffers should have nonzero delta");
    }

    #[timed_test]
    fn layout_entry_accessor_matches_slot() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 5.0).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 10, 10, 10);

        assert!(layout.num_nodes() > 0);
        // First decision node's entry should match slot(0, 0).
        let (entry_offset, entry_actions) = layout.entry(0);
        let (slot_base, slot_actions) = layout.slot(0, 0);
        assert_eq!(entry_offset, slot_base);
        assert_eq!(entry_actions, slot_actions);
    }

    #[timed_test]
    fn build_phase_display_flop_progress() {
        let phase = BuildPhase::FlopProgress {
            flop_name: "AhKd7s".to_string(),
            stage: FlopStage::Solving {
                iteration: 45,
                max_iterations: 200,
                delta: 0.0032,
            },
        };
        let s = format!("{phase}");
        assert!(s.contains("AhKd7s"), "should show flop name: {s}");
        assert!(s.contains("45/200"), "should show iteration: {s}");
        assert!(s.contains("0.0032"), "should show delta: {s}");

        let bucketing = BuildPhase::FlopProgress {
            flop_name: "AhKd7s".to_string(),
            stage: FlopStage::Bucketing { step: 0, total_steps: 6 },
        };
        assert!(format!("{bucketing}").contains("Bucketing"));

        let done = BuildPhase::FlopProgress {
            flop_name: "AhKd7s".to_string(),
            stage: FlopStage::Done,
        };
        assert!(format!("{done}").contains("Done"));
    }

    #[timed_test]
    fn build_phase_display_extracting_ev() {
        let phase = BuildPhase::ExtractingEv(50, 169);
        assert_eq!(format!("{phase}"), "EV histograms (50/169)");
    }

    #[timed_test]
    fn build_phase_display_rebucketing() {
        let phase = BuildPhase::Rebucketing(2, 3);
        assert_eq!(format!("{phase}"), "Rebucketing (2/3)");
    }

    #[test]
    #[ignore = "slow: full postflop abstraction pipeline with rebucketing"]
    fn build_with_rebucket_rounds_1_succeeds() {
        let mut config = PostflopModelConfig::fast();
        config.rebucket_rounds = 1;
        config.postflop_solve_iterations = 10;
        config.max_flop_boards = 3;
        let result = PostflopAbstraction::build(&config, None, None, |_| {});
        assert!(result.is_ok());
        let abs = result.unwrap();
        assert!(!abs.values.is_empty());
    }

    #[test]
    #[ignore = "slow: full postflop abstraction pipeline with 2-round rebucketing"]
    fn build_with_rebucket_rounds_2_succeeds() {
        let mut config = PostflopModelConfig::fast();
        config.rebucket_rounds = 2;
        config.postflop_solve_iterations = 10;
        config.max_flop_boards = 3;
        let result = PostflopAbstraction::build(&config, None, None, |_| {});
        assert!(result.is_ok());
        let abs = result.unwrap();
        assert!(!abs.values.is_empty());
    }

    #[test]
    #[ignore = "slow: builds full pipeline twice to verify rebucketing changes assignments"]
    fn rebucketing_round_2_changes_flop_assignments() {
        let mut config = PostflopModelConfig::fast();
        config.rebucket_rounds = 1;
        config.postflop_solve_iterations = 20;
        config.max_flop_boards = 3;
        let r1 = PostflopAbstraction::build(&config, None, None, |_| {}).unwrap();

        config.rebucket_rounds = 2;
        let r2 = PostflopAbstraction::build(&config, None, None, |_| {}).unwrap();

        // Bucket assignments should differ after EV rebucketing
        let mut any_different = false;
        for flop_idx in 0..r1.buckets.flop.len() {
            if r1.buckets.flop[flop_idx] != r2.buckets.flop[flop_idx] {
                any_different = true;
                break;
            }
        }
        assert!(any_different, "rebucketing should produce different flop assignments");

        // Turn and river should be unchanged (only flop is rebucketed)
        assert_eq!(r1.buckets.turn, r2.buckets.turn, "turn buckets should be unchanged");
        assert_eq!(r1.buckets.river, r2.buckets.river, "river buckets should be unchanged");
    }

    #[test]
    #[ignore = "slow: full MCCFR postflop pipeline via config dispatch"]
    fn build_mccfr_via_config_produces_values() {
        let config = PostflopModelConfig {
            solve_type: PostflopSolveType::Mccfr,
            num_hand_buckets_flop: 5,
            num_hand_buckets_turn: 5,
            num_hand_buckets_river: 5,
            fixed_flops: Some(vec!["AhKsQd".into()]),
            postflop_solve_iterations: 20,
            mccfr_sample_pct: 0.1,
            value_extraction_samples: 100,
            ..PostflopModelConfig::fast()
        };
        let result = PostflopAbstraction::build(&config, None, None, |_| {}).unwrap();
        assert!(!result.values.is_empty());
        // Check approximate zero-sum
        let n = config.num_hand_buckets_flop as usize;
        for hb in 0..n as u16 {
            for ob in 0..n as u16 {
                let v0 = result.values.get_by_flop(0, 0, hb, ob);
                let v1 = result.values.get_by_flop(0, 1, ob, hb);
                assert!((v0 + v1).abs() < 0.5, "should be approximately zero-sum: v0={v0}, v1={v1}");
            }
        }
    }
}
