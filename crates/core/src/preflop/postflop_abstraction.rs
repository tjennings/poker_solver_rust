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

use rand::rngs::SmallRng;
use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::equity::EquityTable;
use super::hand_buckets::{self, BucketEquity, StreetBuckets, StreetEquity};
use super::postflop_model::PostflopModelConfig;
use super::postflop_tree::{PostflopNode, PostflopTerminalType, PostflopTree};
use crate::abstraction::Street;
use crate::poker::Card;

/// All precomputed postflop data needed by the preflop solver.
pub struct PostflopAbstraction {
    pub buckets: StreetBuckets,
    pub street_equity: StreetEquity,
    /// Single shared tree template at `config.primary_spr()`.
    pub tree: PostflopTree,
    /// Precomputed EV table from solved postflop game.
    pub values: PostflopValues,
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
    values: Vec<f64>,
    num_buckets: usize,
    num_flops: usize,
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

    /// Create from raw data (for testing).
    #[cfg(test)]
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
struct NodeEntry {
    offset: usize,
    num_actions: usize,
    street: Street,
}

impl PostflopLayout {
    /// Build the layout from a single postflop tree and bucket counts.
    fn build(
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
fn buckets_for_street(
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
fn annotate_streets(tree: &PostflopTree) -> Vec<Street> {
    let mut streets = vec![Street::Flop; tree.nodes.len()];
    annotate_recursive(&tree.nodes, 0, Street::Flop, &mut streets);
    streets
}

fn annotate_recursive(
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

/// Progress report during postflop abstraction construction.
#[derive(Debug, Clone)]
pub enum BuildPhase {
    /// Computing EHS features and clustering hands into buckets.
    /// Contains `(hands_done, total_hands)`.
    HandBuckets(usize, usize),
    /// Computing bucket-vs-bucket equity table.
    /// Contains `(steps_done, total_steps)`.
    EquityTable(usize, usize),
    /// Building postflop game trees.
    Trees,
    /// Computing flat buffer layout.
    Layout,
    /// Per-flop CFR solve with convergence tracking.
    SolvingPostflop {
        round: u16,
        total_rounds: u16,
        flop_name: String,
        iteration: usize,
        max_iterations: usize,
        delta: f64,
    },
    /// Extracting EV histograms from converged strategy.
    ExtractingEv(usize, usize),
    /// Re-clustering with EV features.
    Rebucketing(u16, u16),
    /// Computing value table from converged strategy.
    ComputingValues,
}

impl std::fmt::Display for BuildPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HandBuckets(done, total) => write!(f, "Hand buckets ({done}/{total})"),
            Self::EquityTable(done, total) => write!(f, "Equity table ({done}/{total})"),
            Self::Trees => write!(f, "Postflop trees"),
            Self::Layout => write!(f, "Buffer layout"),
            Self::SolvingPostflop { round, total_rounds, flop_name, iteration, max_iterations, delta } => {
                write!(f, "[{round}/{total_rounds}] Flop '{flop_name}' iter {iteration}/{max_iterations} \u{03b4}={delta:.4}")
            }
            Self::ExtractingEv(done, total) => write!(f, "EV histograms ({done}/{total})"),
            Self::Rebucketing(round, total) => write!(f, "Rebucketing ({round}/{total})"),
            Self::ComputingValues => write!(f, "Computing values"),
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
        // Phase 1: Build initial EHS abstraction
        let (mut buckets, mut street_equity, flops) = load_or_build_abstraction(
            config,
            &on_progress,
        )?;

        // Build tree (once, shared across all rounds)
        on_progress(BuildPhase::Trees);
        let tree = PostflopTree::build_with_spr(config, config.primary_spr())?;

        on_progress(BuildPhase::Layout);
        let node_streets = annotate_streets(&tree);

        let num_flop_b = buckets.num_flop_buckets as usize;
        let num_turn_b = buckets.num_turn_buckets as usize;
        let num_river_b = buckets.num_river_buckets as usize;

        let layout = PostflopLayout::build(
            &tree,
            &node_streets,
            num_flop_b,
            num_turn_b,
            num_river_b,
        );

        let total_iters = config.postflop_solve_iterations as usize;
        let samples = if config.postflop_solve_samples > 0 {
            config.postflop_solve_samples as usize
        } else {
            num_flop_b
        };

        let flop_names: Vec<String> = flops.iter()
            .map(|f| format!("{}{}{}", f[0], f[1], f[2]))
            .collect();

        let total_rounds = config.rebucket_rounds;
        let num_hands = hand_buckets::NUM_HANDS;
        let num_flops = buckets.flop.len();
        let hands: Vec<_> = crate::hands::all_hands().collect();
        let mut last_solve_results = Vec::new();

        for round in 1..=total_rounds {
            // Solve all flops with current bucket assignments
            last_solve_results = solve_postflop_per_flop(
                &tree,
                &layout,
                &street_equity,
                num_flop_b,
                total_iters,
                samples,
                config.rebucket_delta_threshold,
                &flop_names,
                round,
                total_rounds,
                &on_progress,
            );

            // If not the last round, extract EV and rebucket
            if round < total_rounds {
                // Compute intermediate values from converged strategy
                let strat_refs: Vec<Vec<f64>> = last_solve_results.iter()
                    .map(|r| r.strategy_sum.clone())
                    .collect();
                let values = compute_postflop_values(
                    &tree, &layout, &street_equity, &strat_refs, num_flop_b,
                );

                // Extract EV histograms
                on_progress(BuildPhase::ExtractingEv(0, num_hands));
                let ev_histograms = hand_buckets::build_ev_histograms(
                    &buckets, &values, num_hands, num_flop_b,
                );
                on_progress(BuildPhase::ExtractingEv(num_hands, num_hands));

                // Recluster flop buckets using EV features
                on_progress(BuildPhase::Rebucketing(round, total_rounds));
                buckets.flop = hand_buckets::recluster_flop_buckets(
                    &ev_histograms, buckets.num_flop_buckets, num_flops, num_hands,
                );

                // Recompute flop pairwise equity with new assignments
                on_progress(BuildPhase::EquityTable(0, num_flops));
                let new_flop_equity: Vec<BucketEquity> = (0..num_flops)
                    .map(|flop_idx| {
                        let eq = hand_buckets::compute_pairwise_bucket_equity(
                            &hands,
                            &[flops[flop_idx].as_ref()],
                            &buckets.flop[flop_idx],
                            buckets.num_flop_buckets as usize,
                            1,
                            config.equity_rollout_fraction,
                        );
                        on_progress(BuildPhase::EquityTable(flop_idx + 1, num_flops));
                        eq
                    })
                    .collect();
                street_equity.flop = new_flop_equity;
                // Turn and river equity unchanged
            }
        }

        // Final values from last round
        on_progress(BuildPhase::ComputingValues);
        let per_flop_strategy_sums: Vec<Vec<f64>> = last_solve_results
            .into_iter()
            .map(|r| r.strategy_sum)
            .collect();
        let values = compute_postflop_values(
            &tree,
            &layout,
            &street_equity,
            &per_flop_strategy_sums,
            num_flop_b,
        );

        Ok(Self {
            buckets,
            street_equity,
            tree,
            values,
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
        street_equity: StreetEquity,
        values: PostflopValues,
        flops: Vec<[Card; 3]>,
    ) -> Result<Self, PostflopAbstractionError> {
        let tree = PostflopTree::build_with_spr(config, config.primary_spr())?;
        Ok(Self {
            buckets,
            street_equity,
            tree,
            values,
            spr: config.primary_spr(),
            flops,
        })
    }
}

/// Build abstraction data: independent per-street buckets + equity tables.
///
/// Computes real bucket-pair equity from the histogram CDFs and river equities
/// produced during bucketing. For each street:
/// - Flop/turn: extract average equity from each CDF via `cdf_to_avg_equity`,
///   then group by bucket to get centroid equity.
/// - River: use raw scalar equities directly.
/// - Pair equity: `equity(a, b) = centroid_a / (centroid_a + centroid_b)`.
#[allow(clippy::unnecessary_wraps)]
fn load_or_build_abstraction(
    config: &PostflopModelConfig,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Result<(StreetBuckets, StreetEquity, Vec<[Card; 3]>), PostflopAbstractionError> {
    // Cache is disabled during the StreetBuckets migration (format changed).
    // Task 9 will re-enable caching with the new format.

    on_progress(BuildPhase::HandBuckets(0, hand_buckets::NUM_HANDS));

    let hands: Vec<_> = crate::hands::all_hands().collect();
    let flops = if let Some(ref names) = config.fixed_flops {
        crate::preflop::ehs::parse_flops(names)
            .map_err(PostflopAbstractionError::InvalidConfig)?
    } else {
        crate::preflop::ehs::sample_canonical_flops(config.max_flop_boards)
    };

    let result = hand_buckets::build_street_buckets_independent(
        &hands,
        &flops,
        config.num_hand_buckets_flop,
        config.num_hand_buckets_turn,
        config.num_hand_buckets_river,
        &|progress| {
            use hand_buckets::BuildProgress;
            match progress {
                BuildProgress::FlopFeatures(done, total)
                | BuildProgress::TurnFeatures(done, total)
                | BuildProgress::RiverFeatures(done, total) => {
                    on_progress(BuildPhase::HandBuckets(done, total));
                }
                BuildProgress::FlopClustering
                | BuildProgress::TurnClustering
                | BuildProgress::RiverClustering => {}
            }
        },
    );

    let num_flops = flops.len();
    on_progress(BuildPhase::EquityTable(0, num_flops + 2));

    let street_equity = result.compute_pairwise_street_equity(
        &hands,
        &flops,
        &result.turn_boards,
        &result.river_boards,
        config.equity_rollout_fraction,
        |done, total| on_progress(BuildPhase::EquityTable(done, total)),
    );

    Ok((result.buckets, street_equity, flops))
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

/// Equity references for a single flop solve.
struct SolveEquity<'a> {
    flop: &'a BucketEquity,
    turn: &'a BucketEquity,
    river: &'a BucketEquity,
}

/// Solve all flops in parallel, each with the shared tree template.
///
/// Returns per-flop solve results (strategy sums + convergence info).
#[allow(clippy::cast_possible_truncation, clippy::too_many_arguments)]
fn solve_postflop_per_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    street_equity: &StreetEquity,
    num_flop_buckets: usize,
    num_iterations: usize,
    samples_per_iter: usize,
    delta_threshold: f64,
    flop_names: &[String],
    round: u16,
    total_rounds: u16,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Vec<FlopSolveResult> {
    let num_flops = street_equity.flop.len();
    let use_exhaustive = num_flop_buckets * num_flop_buckets <= samples_per_iter;
    let actual_pairs = if use_exhaustive {
        num_flop_buckets * num_flop_buckets
    } else {
        samples_per_iter
    };

    let buf_size = layout.total_size;
    #[allow(clippy::cast_precision_loss)]
    let mb = buf_size as f64 * 8.0 / 1_000_000.0;
    let mode = if use_exhaustive { "exhaustive" } else { "sampled" };
    tracing::debug!(
        spr = format_args!("{:.1}", tree.spr),
        nodes = tree.node_count(),
        buf_size,
        mb = format_args!("{mb:.1}"),
        mode,
        actual_pairs,
        num_flops,
        delta_threshold,
        "Per-flop solve allocated"
    );

    (0..num_flops)
        .into_par_iter()
        .map(|flop_idx| {
            let solve_eq = SolveEquity {
                flop: &street_equity.flop[flop_idx],
                turn: &street_equity.turn,
                river: &street_equity.river,
            };
            let flop_name = flop_names.get(flop_idx).map_or("?", |s| s.as_str());
            solve_one_flop(
                tree,
                layout,
                &solve_eq,
                num_flop_buckets,
                buf_size,
                num_iterations,
                samples_per_iter,
                delta_threshold,
                flop_idx,
                flop_name,
                round,
                total_rounds,
                on_progress,
            )
        })
        .collect()
}

/// Run CFR for a single flop with the shared tree template.
///
/// Runs up to `num_iterations` CFR iterations, stopping early when the max
/// strategy change between consecutive iterations drops below `delta_threshold`.
/// Early stopping requires at least 2 completed iterations.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn solve_one_flop(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    num_flop_buckets: usize,
    buf_size: usize,
    num_iterations: usize,
    samples_per_iter: usize,
    delta_threshold: f64,
    flop_idx: usize,
    flop_name: &str,
    round: u16,
    total_rounds: u16,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> FlopSolveResult {
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let use_exhaustive = num_flop_buckets * num_flop_buckets <= samples_per_iter;
    let mut final_delta = 0.0;
    let mut iterations_used = 0;

    for iter in 0..num_iterations {
        let iteration = iter as u64 + 1;

        // Snapshot regrets before this iteration for delta computation.
        let prev_regrets = regret_sum.clone();

        let (dr, ds) = if use_exhaustive {
            exhaustive_cfr_iteration(
                tree, layout, solve_eq, &regret_sum,
                num_flop_buckets, buf_size, iteration,
            )
        } else {
            sampled_cfr_iteration(
                tree, layout, solve_eq, &regret_sum,
                num_flop_buckets, buf_size, samples_per_iter, iteration, iter,
                flop_idx,
            )
        };

        add_buffers(&mut regret_sum, &dr);
        add_buffers(&mut strategy_sum, &ds);
        iterations_used = iter + 1;

        // Compute strategy delta (meaningful after iteration 1).
        if iter >= 1 {
            final_delta = max_strategy_delta(&prev_regrets, &regret_sum, layout, tree);
        }

        on_progress(BuildPhase::SolvingPostflop {
            round,
            total_rounds,
            flop_name: flop_name.to_string(),
            iteration: iter + 1,
            max_iterations: num_iterations,
            delta: final_delta,
        });

        // Early stopping after at least 2 iterations.
        if iter >= 1 && final_delta < delta_threshold {
            break;
        }
    }

    FlopSolveResult { strategy_sum, final_delta, iterations_used }
}

#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn exhaustive_cfr_iteration(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    regret_sum: &[f64],
    num_flop_buckets: usize,
    buf_size: usize,
    iteration: u64,
) -> (Vec<f64>, Vec<f64>) {
    (0..num_flop_buckets)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), hb| {
                let hb = hb as u16;
                for ob in 0..num_flop_buckets as u16 {
                    for hero_pos in 0..2u8 {
                        solve_cfr_traverse(
                            tree, layout, solve_eq, regret_sum,
                            &mut dr, &mut ds,
                            0, hb, ob, hero_pos, 1.0, 1.0, iteration,
                        );
                    }
                }
                (dr, ds)
            },
        )
        .reduce(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr1, mut ds1), (dr2, ds2)| {
                add_buffers(&mut dr1, &dr2);
                add_buffers(&mut ds1, &ds2);
                (dr1, ds1)
            },
        )
}

#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn sampled_cfr_iteration(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    regret_sum: &[f64],
    num_flop_buckets: usize,
    buf_size: usize,
    samples_per_iter: usize,
    iteration: u64,
    iter_idx: usize,
    flop_idx: usize,
) -> (Vec<f64>, Vec<f64>) {
    let seed = iter_idx as u64 * 1_000_003 + flop_idx as u64;
    let mut rng = SmallRng::seed_from_u64(seed);
    let pairs: Vec<(u16, u16)> = (0..samples_per_iter)
        .map(|_| {
            let hb = rng.random_range(0..num_flop_buckets as u16);
            let ob = rng.random_range(0..num_flop_buckets as u16);
            (hb, ob)
        })
        .collect();

    let num_threads = rayon::current_num_threads().max(1);
    let chunk_size = (pairs.len() / num_threads).max(1);

    pairs
        .par_chunks(chunk_size)
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), chunk| {
                for &(hb, ob) in chunk {
                    for hero_pos in 0..2u8 {
                        solve_cfr_traverse(
                            tree, layout, solve_eq, regret_sum,
                            &mut dr, &mut ds,
                            0, hb, ob, hero_pos, 1.0, 1.0, iteration,
                        );
                    }
                }
                (dr, ds)
            },
        )
        .reduce(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr1, mut ds1), (dr2, ds2)| {
                add_buffers(&mut dr1, &dr2);
                add_buffers(&mut ds1, &ds2);
                (dr1, ds1)
            },
        )
}

/// Single CFR traversal for the postflop solve phase. Returns hero EV fraction.
///
/// With imperfect recall, bucket IDs pass through unchanged at Chance nodes
/// (street transitions). The player does not track bucket transitions — the
/// strategy at a turn/river node is the same regardless of which flop bucket
/// led there.
#[allow(clippy::too_many_arguments)]
fn solve_cfr_traverse(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    iteration: u64,
) -> f64 {
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal { terminal_type, pot_fraction } => {
            let current_street = layout.street(node_idx);
            let eq_table = solve_equity_for_street(solve_eq, current_street);
            postflop_terminal_value(*terminal_type, *pot_fraction, hero_bucket, opp_bucket, hero_pos, eq_table)
        }
        PostflopNode::Chance { children, weights, .. } => {
            children.iter().zip(weights.iter())
                .map(|(&child, &w)| {
                    w * solve_cfr_traverse(
                        tree, layout, solve_eq, snapshot, dr, ds,
                        child, hero_bucket, opp_bucket, hero_pos,
                        reach_hero, reach_opp, iteration,
                    )
                })
                .sum()
        }
        PostflopNode::Decision { position, children, .. } => {
            let is_hero = *position == hero_pos;
            let bucket = if is_hero { hero_bucket } else { opp_bucket };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let mut strategy = [0.0f64; MAX_POSTFLOP_ACTIONS];
            regret_matching_into(snapshot, start, &mut strategy[..num_actions]);

            if is_hero {
                solve_traverse_hero(
                    tree, layout, solve_eq, snapshot, dr, ds,
                    start, hero_bucket, opp_bucket, hero_pos,
                    reach_hero, reach_opp, children, &strategy[..num_actions], iteration,
                )
            } else {
                solve_traverse_opponent(
                    tree, layout, solve_eq, snapshot, dr, ds,
                    hero_bucket, opp_bucket, hero_pos,
                    reach_hero, reach_opp, children, &strategy[..num_actions], iteration,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_traverse_hero(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    slot_start: usize,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    children: &[u32],
    strategy: &[f64],
    iteration: u64,
) -> f64 {
    let num_actions = children.len();
    let mut action_values = [0.0f64; MAX_POSTFLOP_ACTIONS];

    for (i, &child) in children.iter().enumerate() {
        action_values[i] = solve_cfr_traverse(
            tree, layout, solve_eq, snapshot, dr, ds,
            child, hero_bucket, opp_bucket, hero_pos,
            reach_hero * strategy[i], reach_opp, iteration,
        );
    }

    let node_value: f64 = strategy.iter()
        .zip(&action_values[..num_actions])
        .map(|(s, v)| s * v)
        .sum();

    #[allow(clippy::cast_precision_loss)]
    let weight = iteration as f64;
    for (i, val) in action_values[..num_actions].iter().enumerate() {
        dr[slot_start + i] += weight * reach_opp * (val - node_value);
    }
    for (i, &s) in strategy.iter().enumerate() {
        ds[slot_start + i] += weight * reach_hero * s;
    }

    node_value
}

#[allow(clippy::too_many_arguments)]
fn solve_traverse_opponent(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    children: &[u32],
    strategy: &[f64],
    iteration: u64,
) -> f64 {
    children.iter().enumerate()
        .map(|(i, &child)| {
            strategy[i] * solve_cfr_traverse(
                tree, layout, solve_eq, snapshot, dr, ds,
                child, hero_bucket, opp_bucket, hero_pos,
                reach_hero, reach_opp * strategy[i], iteration,
            )
        })
        .sum()
}

/// Element-wise `dst[i] += src[i]`.
fn add_buffers(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s;
    }
}

/// Compute max strategy change between two regret buffers across all decision nodes.
///
/// For each decision node in the tree, iterates through every bucket's action slice,
/// applies regret matching to both the old and new regret buffers, and returns the
/// maximum absolute difference in any action probability.
fn max_strategy_delta(
    old_regrets: &[f64],
    new_regrets: &[f64],
    layout: &PostflopLayout,
    tree: &PostflopTree,
) -> f64 {
    let mut max_delta = 0.0f64;

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
                regret_matching_into(old_regrets, pos, &mut old_strat);
                regret_matching_into(new_regrets, pos, &mut new_strat);
                for i in 0..num_actions {
                    let diff = (old_strat[i] - new_strat[i]).abs();
                    max_delta = max_delta.max(diff);
                }
                pos += num_actions;
            }
        }
    }
    max_delta
}

// ──────────────────────────────────────────────────────────────────────────────
// Value table computation from converged strategy
// ──────────────────────────────────────────────────────────────────────────────

#[allow(clippy::cast_possible_truncation)]
fn compute_postflop_values(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    street_equity: &StreetEquity,
    per_flop_strategy_sums: &[Vec<f64>],
    num_flop_buckets: usize,
) -> PostflopValues {
    let num_flops = per_flop_strategy_sums.len();
    let total = num_flops * 2 * num_flop_buckets * num_flop_buckets;
    let mut values = vec![0.0f64; total];

    for flop_idx in 0..num_flops {
        let strategy_sum = &per_flop_strategy_sums[flop_idx];
        let solve_eq = SolveEquity {
            flop: &street_equity.flop[flop_idx],
            turn: &street_equity.turn,
            river: &street_equity.river,
        };
        for hero_pos in 0..2u8 {
            for hero_bucket in 0..num_flop_buckets as u16 {
                for opp_bucket in 0..num_flop_buckets as u16 {
                    let ev = eval_with_avg_strategy(
                        tree, layout, &solve_eq, strategy_sum,
                        0, hero_bucket, opp_bucket, hero_pos,
                    );
                    let idx = flop_idx * 2 * num_flop_buckets * num_flop_buckets
                        + (hero_pos as usize) * num_flop_buckets * num_flop_buckets
                        + (hero_bucket as usize) * num_flop_buckets
                        + opp_bucket as usize;
                    values[idx] = ev;
                }
            }
        }
    }

    PostflopValues { values, num_buckets: num_flop_buckets, num_flops }
}

/// Walk tree using averaged (converged) strategy, returning hero EV fraction.
#[allow(clippy::too_many_arguments)]
fn eval_with_avg_strategy(
    tree: &PostflopTree,
    layout: &PostflopLayout,
    solve_eq: &SolveEquity<'_>,
    strategy_sum: &[f64],
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
) -> f64 {
    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal { terminal_type, pot_fraction } => {
            let current_street = layout.street(node_idx);
            let eq_table = solve_equity_for_street(solve_eq, current_street);
            postflop_terminal_value(*terminal_type, *pot_fraction, hero_bucket, opp_bucket, hero_pos, eq_table)
        }
        PostflopNode::Chance { children, weights, .. } => {
            children.iter().zip(weights.iter())
                .map(|(&child, &w)| {
                    w * eval_with_avg_strategy(
                        tree, layout, solve_eq, strategy_sum,
                        child, hero_bucket, opp_bucket, hero_pos,
                    )
                })
                .sum()
        }
        PostflopNode::Decision { position, children, .. } => {
            let bucket = if *position == hero_pos { hero_bucket } else { opp_bucket };
            let (start, _) = layout.slot(node_idx, bucket);
            let num_actions = children.len();

            let strategy = normalize_strategy_sum(strategy_sum, start, num_actions);

            children.iter().enumerate()
                .map(|(i, &child)| {
                    strategy[i] * eval_with_avg_strategy(
                        tree, layout, solve_eq, strategy_sum,
                        child, hero_bucket, opp_bucket, hero_pos,
                    )
                })
                .sum()
        }
    }
}

/// Normalize strategy sum into a probability distribution.
fn normalize_strategy_sum(strategy_sum: &[f64], start: usize, num_actions: usize) -> Vec<f64> {
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

const MAX_POSTFLOP_ACTIONS: usize = 8;

/// Select the equity table for the given street.
fn solve_equity_for_street<'a>(solve_eq: &'a SolveEquity<'a>, street: Street) -> &'a BucketEquity {
    match street {
        Street::Preflop | Street::Flop => solve_eq.flop,
        Street::Turn => solve_eq.turn,
        Street::River => solve_eq.river,
    }
}

fn postflop_terminal_value(
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
fn regret_matching_into(regret_buf: &[f64], start: usize, out: &mut [f64]) {
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
    fn max_strategy_delta_identical_buffers_is_zero() {
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 10, 10, 10);
        let buf = vec![1.0f64; layout.total_size];
        let delta = max_strategy_delta(&buf, &buf, &layout, &tree);
        assert!(
            delta.abs() < 1e-12,
            "identical buffers should have zero delta, got {delta}"
        );
    }

    #[timed_test]
    fn max_strategy_delta_detects_change() {
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
        let delta = max_strategy_delta(&old, &new, &layout, &tree);
        assert!(delta > 0.0, "different buffers should have nonzero delta");
    }

    #[timed_test]
    fn solve_one_flop_returns_result_struct() {
        // Verify that solve_one_flop returns FlopSolveResult with correct fields.
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        // Create minimal equity tables (5 buckets).
        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let solve_eq = SolveEquity {
            flop: &eq,
            turn: &eq,
            river: &eq,
        };
        let result = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 4, 25, 0.0001,
            0, "AhKd7s", 1, 1, &|_| {},
        );
        assert_eq!(result.strategy_sum.len(), buf_size);
        assert!(result.iterations_used >= 2);
        assert!(result.iterations_used <= 4);
        assert!(result.final_delta.is_finite());
    }

    #[timed_test]
    fn solve_one_flop_early_stop_with_zero_threshold() {
        // With threshold=0.0, should run all iterations (never stops early
        // because delta is never exactly 0 after real CFR iterations).
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let solve_eq = SolveEquity {
            flop: &eq,
            turn: &eq,
            river: &eq,
        };
        let result = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 3, 25, 0.0,
            0, "AhKd7s", 1, 1, &|_| {},
        );
        assert_eq!(result.iterations_used, 3, "zero threshold should run all iterations");
    }

    #[timed_test]
    fn solve_one_flop_early_stop_with_large_threshold() {
        // With a very large threshold, should stop at iteration 2 (minimum).
        let config = PostflopModelConfig::fast();
        let tree = PostflopTree::build_with_spr(&config, 3.5).unwrap();
        let streets = annotate_streets(&tree);
        let layout = PostflopLayout::build(&tree, &streets, 5, 5, 5);
        let buf_size = layout.total_size;

        let eq = BucketEquity {
            equity: vec![vec![0.5; 5]; 5],
            num_buckets: 5,
        };
        let solve_eq = SolveEquity {
            flop: &eq,
            turn: &eq,
            river: &eq,
        };
        let result = solve_one_flop(
            &tree, &layout, &solve_eq,
            5, buf_size, 100, 25, f64::INFINITY,
            0, "AhKd7s", 1, 1, &|_| {},
        );
        assert_eq!(result.iterations_used, 2, "huge threshold should stop at minimum 2 iterations");
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
    fn build_phase_display_solving_postflop() {
        let phase = BuildPhase::SolvingPostflop {
            round: 1,
            total_rounds: 2,
            flop_name: "AhKd7s".to_string(),
            iteration: 45,
            max_iterations: 200,
            delta: 0.0032,
        };
        let s = format!("{phase}");
        assert!(s.contains("1/2"), "should show round: {s}");
        assert!(s.contains("AhKd7s"), "should show flop name: {s}");
        assert!(s.contains("45/200"), "should show iteration: {s}");
        assert!(s.contains("0.0032"), "should show delta: {s}");
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
}
