//! Postflop abstraction: precomputed data + CFR traversal for the abstracted postflop model.
//!
//! Combines board abstraction, canonical 169-hand indexing, and postflop trees
//! into a single structure that the preflop solver queries at showdown terminals.

use serde::{Deserialize, Serialize};

use super::postflop_hands::{parse_flops, sample_canonical_flops, NUM_CANONICAL_HANDS};
use super::postflop_exhaustive::build_exhaustive;
use super::postflop_mccfr::build_mccfr;
use super::postflop_model::{PostflopModelConfig, PostflopSolveType};
use super::postflop_tree::{PostflopNode, PostflopTree};
use crate::abstraction::Street;
use crate::poker::Card;

/// All precomputed postflop data needed by the preflop solver.
pub struct PostflopAbstraction {
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

/// Precomputed EV table: `values[flop_idx][hero_pos][hero_hand][opp_hand]` → EV fraction.
///
/// Stored as a flat `Vec<f64>` indexed by
/// `flop_idx * 2 * n * n + hero_pos * n * n + hero_hand * n + opp_hand`.
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
    #[allow(dead_code)] // Used by exhaustive backend (not yet implemented)
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
    #[allow(dead_code)] // Used by exhaustive backend (not yet implemented)
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
    /// CFR solve in progress.
    Solving {
        iteration: usize,
        max_iterations: usize,
        delta: f64,
        /// Label for the convergence metric (e.g. "δ" for strategy delta, "expl" for exploitability).
        metric_label: String,
    },
    /// Extracting EV estimates from converged strategy.
    EstimatingEv { sample: usize, total_samples: usize },
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
    /// Computing value table from converged strategy.
    ComputingValues,
    /// Aggregate progress for MCCFR: how many flops have completed solve + extraction.
    MccfrFlopsCompleted { completed: usize, total: usize },
}

impl std::fmt::Display for BuildPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FlopProgress { flop_name, stage } => match stage {
                FlopStage::Solving { iteration, max_iterations, delta, metric_label } =>
                    write!(f, "Flop '{flop_name}' CFR {metric_label}={delta:.4} ({iteration}/{max_iterations})"),
                FlopStage::EstimatingEv { sample, total_samples } =>
                    write!(f, "Flop '{flop_name}' EV Extraction ({sample}/{total_samples})"),
                FlopStage::Done => write!(f, "Flop '{flop_name}' Done"),
            },
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
    /// # Errors
    ///
    /// Returns an error if tree building fails.
    pub fn build(
        config: &PostflopModelConfig,
        _equity_table: Option<&super::equity::EquityTable>,
        on_progress: impl Fn(BuildPhase) + Sync,
    ) -> Result<Self, PostflopAbstractionError> {
        let flops = if let Some(ref names) = config.fixed_flops {
            parse_flops(names).map_err(PostflopAbstractionError::InvalidConfig)?
        } else {
            sample_canonical_flops(config.max_flop_boards)
        };
        let tree = PostflopTree::build_with_spr(config, config.primary_spr())?;
        let node_streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &node_streets, NUM_CANONICAL_HANDS, NUM_CANONICAL_HANDS, NUM_CANONICAL_HANDS);

        let values = match config.solve_type {
            PostflopSolveType::Mccfr => build_mccfr(config, &tree, &layout, &node_streets, &flops, &on_progress),
            PostflopSolveType::Exhaustive => build_exhaustive(config, &tree, &layout, &node_streets, &flops, &on_progress),
        };
        on_progress(BuildPhase::ComputingValues);
        let flop_weights = crate::flops::lookup_flop_weights(&flops);
        let hand_avg_values = compute_hand_avg_values(&values, &flop_weights);
        Ok(Self { tree, values, hand_avg_values, spr: config.primary_spr(), flops })
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
        values: PostflopValues,
        hand_avg_values: Vec<f64>,
        flops: Vec<[Card; 3]>,
    ) -> Result<Self, PostflopAbstractionError> {
        let tree = PostflopTree::build_with_spr(config, config.primary_spr())?;
        Ok(Self { tree, values, hand_avg_values, spr: config.primary_spr(), flops })
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
/// Each flop's contribution is weighted by its combinatorial multiplicity
/// (`flop_weights[flop_idx]`), so that e.g. rainbow flops (weight 24) count 6×
/// more than monotone flops (weight 4), matching their real-world frequency.
///
/// Returns a flat `Vec<f64>` of size `2 * N * N` where N = 169 (canonical hands).
/// Parallelized across hero hands with rayon.
#[allow(clippy::cast_precision_loss)]
pub fn compute_hand_avg_values(values: &PostflopValues, flop_weights: &[u16]) -> Vec<f64> {
    use rayon::prelude::*;
    let num_flops = values.num_flops;
    let num_hands = NUM_CANONICAL_HANDS;
    if num_flops == 0 { return vec![0.0; 2 * num_hands * num_hands]; }
    let mut out = vec![0.0f64; 2 * num_hands * num_hands];
    // Each cell accumulates (weighted_sum, total_weight) — NaN entries from
    // unsampled/impossible hand pairs are skipped so they don't dilute the average.
    let chunks: Vec<(usize, Vec<f64>, Vec<f64>)> = (0..num_hands)
        .into_par_iter()
        .map(|hero_hand| {
            let mut sums = vec![0.0f64; 2 * num_hands];
            let mut weights = vec![0.0f64; 2 * num_hands];
            for flop_idx in 0..num_flops {
                let w = f64::from(*flop_weights.get(flop_idx).unwrap_or(&1));
                for opp_hand in 0..num_hands {
                    let v0 = values.get_by_flop(flop_idx, 0, hero_hand as u16, opp_hand as u16);
                    if !v0.is_nan() {
                        sums[opp_hand] += v0 * w;
                        weights[opp_hand] += w;
                    }
                    let v1 = values.get_by_flop(flop_idx, 1, hero_hand as u16, opp_hand as u16);
                    if !v1.is_nan() {
                        sums[num_hands + opp_hand] += v1 * w;
                        weights[num_hands + opp_hand] += w;
                    }
                }
            }
            (hero_hand, sums, weights)
        })
        .collect();
    let n = num_hands;
    for (hero_hand, sums, weights) in chunks {
        for opp_hand in 0..n {
            out[hero_hand * n + opp_hand] = if weights[opp_hand] > 0.0 {
                sums[opp_hand] / weights[opp_hand]
            } else {
                0.0
            };
            out[n * n + hero_hand * n + opp_hand] = if weights[n + opp_hand] > 0.0 {
                sums[n + opp_hand] / weights[n + opp_hand]
            } else {
                0.0
            };
        }
    }
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// Postflop pre-solve: run CFR to convergence, then extract value table
// ──────────────────────────────────────────────────────────────────────────────

/// Result of a single flop CFR solve.
#[allow(dead_code)] // delta and iterations_used read in tests
pub(crate) struct FlopSolveResult {
    pub strategy_sum: Vec<f64>,
    pub delta: f64,
    pub iterations_used: usize,
}

/// Element-wise `dst[i] += src[i]`.
pub(crate) fn add_buffers(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// Average positive regret per entry per iteration for a flat regret buffer.
///
/// Sums only positive values in `regret_sum`, divides by total entry count
/// and `iterations`. Returns 0.0 if the buffer is empty or iterations is 0.
///
/// NOTE: Superseded by `weighted_avg_strategy_delta` for MCCFR convergence
/// detection. Kept for diagnostic use.
#[cfg(test)]
pub(crate) fn avg_positive_regret_flat(regret_sum: &[f64], iterations: u64) -> f64 {
    if iterations == 0 || regret_sum.is_empty() {
        return 0.0;
    }
    let total: f64 = regret_sum.iter().filter(|&&r| r > 0.0).sum();
    total / regret_sum.len() as f64 / iterations as f64
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
        let layout = PostflopLayout::build(&tree, &streets, 169, 169, 169);

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
    fn avg_positive_regret_flat_basic() {
        // 3 positive, 1 negative, 1 zero → sum positives = 6.0, count = 5, iters = 2
        let buf = vec![1.0, 2.0, 3.0, -10.0, 0.0];
        let result = avg_positive_regret_flat(&buf, 2);
        let expected = 6.0 / 5.0 / 2.0;
        assert!((result - expected).abs() < 1e-10, "expected {expected}, got {result}");
    }

    #[timed_test]
    fn avg_positive_regret_flat_empty() {
        assert!(avg_positive_regret_flat(&[], 10).abs() < 1e-10);
        assert!(avg_positive_regret_flat(&[5.0], 0).abs() < 1e-10);
    }

    #[timed_test]
    fn weighted_avg_strategy_delta_identical_buffers_is_zero() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 169, 169, 169);
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
        let layout = PostflopLayout::build(&tree, &streets, 169, 169, 169);
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
    fn build_phase_display_flop_progress() {
        let phase = BuildPhase::FlopProgress {
            flop_name: "AhKd7s".to_string(),
            stage: FlopStage::Solving {
                iteration: 45,
                max_iterations: 200,
                delta: 0.0032,
                metric_label: "\u{03b4}".to_string(),
            },
        };
        let s = format!("{phase}");
        assert!(s.contains("AhKd7s"), "should show flop name: {s}");
        assert!(s.contains("45/200"), "should show iteration: {s}");
        assert!(s.contains("δ="), "should show delta label: {s}");
        assert!(s.contains("0.0032"), "should show delta value: {s}");

        let done = BuildPhase::FlopProgress {
            flop_name: "AhKd7s".to_string(),
            stage: FlopStage::Done,
        };
        assert!(format!("{done}").contains("Done"));
    }

    #[timed_test]
    fn compute_hand_avg_values_uses_flop_weights() {
        // Two flops, 2 "hands" (tiny for speed).
        // Flop 0: all EVs = 1.0, weight = 24 (rainbow)
        // Flop 1: all EVs = 0.0, weight = 4 (monotone)
        // Unweighted average would be 0.5.
        // Weighted average should be 24/(24+4) * 1.0 + 4/(24+4) * 0.0 ≈ 0.857.
        let num_flops = 2;
        let num_buckets = 2;
        let total = num_flops * 2 * num_buckets * num_buckets;
        let mut raw = vec![0.0f64; total];
        // Flop 0: all entries = 1.0
        for i in 0..(2 * num_buckets * num_buckets) {
            raw[i] = 1.0;
        }
        // Flop 1: all entries = 0.0 (already zero)
        let values = PostflopValues::from_raw(raw, num_buckets, num_flops);
        let weights: Vec<u16> = vec![24, 4];

        let avg = compute_hand_avg_values(&values, &weights);
        let expected = 24.0 / 28.0; // ≈ 0.857
        // Check first entry (pos=0, hero=0, opp=0)
        assert!(
            (avg[0] - expected).abs() < 1e-9,
            "weighted avg should be {expected}, got {}",
            avg[0]
        );
        // Also verify pos=1 slice
        let pos1_idx = num_buckets * num_buckets;
        assert!(
            (avg[pos1_idx] - expected).abs() < 1e-9,
            "pos=1 weighted avg should be {expected}, got {}",
            avg[pos1_idx]
        );
    }

    #[test]
    #[ignore = "slow: full MCCFR postflop pipeline via config dispatch"]
    fn build_mccfr_via_config_produces_values() {
        let config = PostflopModelConfig {
            solve_type: PostflopSolveType::Mccfr,
            fixed_flops: Some(vec!["AhKsQd".into()]),
            postflop_solve_iterations: 20,
            mccfr_sample_pct: 0.1,
            value_extraction_samples: 100,
            ..PostflopModelConfig::fast()
        };
        let result = PostflopAbstraction::build(&config, None, |_| {}).unwrap();
        assert!(!result.values.is_empty());
        for h in 0..5u16 {
            for o in 0..5u16 {
                let v0 = result.values.get_by_flop(0, 0, h, o);
                let v1 = result.values.get_by_flop(0, 1, o, h);
                assert!((v0 + v1).abs() < 0.5, "approximately zero-sum: v0={v0}, v1={v1}");
            }
        }
    }
}
