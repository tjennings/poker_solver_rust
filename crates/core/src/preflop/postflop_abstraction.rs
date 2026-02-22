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

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use super::abstraction_cache;
use super::board_abstraction::{BoardAbstraction, BoardAbstractionConfig};
use super::equity::EquityTable;
use super::hand_buckets::{self, BucketEquity, HandBucketMapping};
use super::postflop_model::PostflopModelConfig;
use super::postflop_tree::{PostflopNode, PostflopTerminalType, PostflopTree};
use crate::abstraction::Street;

/// All precomputed postflop data needed by the preflop solver.
pub struct PostflopAbstraction {
    pub board: BoardAbstraction,
    pub buckets: HandBucketMapping,
    pub bucket_equity: BucketEquity,
    pub trees: Vec<PostflopTree>,
    /// Precomputed EV table from solved postflop game.
    pub values: PostflopValues,
    /// Canonical SPR values for runtime SPR mapping.
    pub canonical_sprs: Vec<f64>,
}

/// Precomputed EV table: `values[spr_idx][hero_pos][hero_bucket][opp_bucket]` → EV fraction.
///
/// Stored as a flat `Vec<f64>` indexed by
/// `spr_idx * 2 * n * n + hero_pos * n * n + hero_bucket * n + opp_bucket`.
pub struct PostflopValues {
    values: Vec<f64>,
    num_buckets: usize,
    num_sprs: usize,
}

impl PostflopValues {
    /// Number of SPR slots in the value table.
    #[must_use]
    pub fn num_sprs(&self) -> usize {
        self.num_sprs
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

    /// Look up postflop EV by SPR index.
    #[inline]
    #[must_use]
    pub fn get_by_spr(&self, spr_idx: usize, hero_pos: u8, hero_bucket: u16, opp_bucket: u16) -> f64 {
        let n = self.num_buckets;
        let idx = spr_idx * 2 * n * n + (hero_pos as usize) * n * n + (hero_bucket as usize) * n + opp_bucket as usize;
        self.values.get(idx).copied().unwrap_or(0.0)
    }

}

/// Maps `(node_idx, bucket)` → flat buffer offset for ONE tree.
///
/// Each decision node reserves `num_buckets × num_actions` slots.
/// The bucket count varies by the street of the node.
pub struct PostflopLayout {
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
    #[error("board abstraction: {0}")]
    Board(#[from] super::board_abstraction::BoardAbstractionError),
    #[error("hand buckets: {0}")]
    Buckets(#[from] hand_buckets::BucketError),
    #[error("postflop tree: {0}")]
    Tree(#[from] super::postflop_tree::PostflopTreeError),
    #[error("abstraction cache: {0}")]
    Cache(#[from] abstraction_cache::CacheError),
    #[error("canonical_sprs must be non-empty")]
    EmptyCanonicalSprs,
}

/// Progress report during postflop abstraction construction.
#[derive(Debug, Clone)]
pub enum BuildPhase {
    /// Clustering canonical flops into texture buckets.
    BoardAbstraction,
    /// Computing EHS features and clustering hands into buckets.
    /// Contains `(hands_done, total_hands)`.
    HandBuckets(usize, usize),
    /// Computing bucket-vs-bucket equity table.
    EquityTable,
    /// Building postflop game trees for each SPR.
    Trees,
    /// Computing flat buffer layout.
    Layout,
    /// Solving postflop game to convergence. Contains `(iteration, total)`.
    SolvingPostflop(usize, usize),
    /// Computing value table from converged strategy.
    ComputingValues,
}

impl std::fmt::Display for BuildPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BoardAbstraction => write!(f, "Board abstraction"),
            Self::HandBuckets(done, total) => write!(f, "Hand buckets ({done}/{total})"),
            Self::EquityTable => write!(f, "Equity table"),
            Self::Trees => write!(f, "Postflop trees"),
            Self::Layout => write!(f, "Buffer layout"),
            Self::SolvingPostflop(iter, total) => write!(f, "Solving postflop ({iter}/{total})"),
            Self::ComputingValues => write!(f, "Computing value table"),
        }
    }
}

/// Total number of canonical preflop hands (for fine-grained progress).
pub const NUM_CANONICAL_HANDS: usize = hand_buckets::NUM_HANDS;

impl PostflopAbstraction {
    /// Build all precomputed postflop data from configuration.
    ///
    /// This is expensive (minutes) — called once before training begins.
    /// Calls `on_progress(phase)` at the start of each build phase.
    ///
    /// When `equity_table` is provided, bucket equity is derived from the
    /// precomputed 169×169 pairwise equities (much more accurate). Without it,
    /// falls back to the EHS centroid approximation.
    ///
    /// When `cache_base` is provided, the expensive abstraction phases (board,
    /// hand buckets, equity) are cached to disk. On subsequent runs with the
    /// same abstraction config, these are loaded from cache instead of recomputed.
    ///
    /// # Errors
    ///
    /// Returns an error if board abstraction, hand bucketing, or tree building fails.
    pub fn build(
        config: &PostflopModelConfig,
        equity_table: Option<&EquityTable>,
        cache_base: Option<&std::path::Path>,
        on_progress: impl Fn(BuildPhase) + Sync,
    ) -> Result<Self, PostflopAbstractionError> {
        if config.canonical_sprs.is_empty() {
            return Err(PostflopAbstractionError::EmptyCanonicalSprs);
        }
        let (board, buckets, bucket_equity) = load_or_build_abstraction(
            config,
            equity_table,
            cache_base,
            &on_progress,
        )?;

        on_progress(BuildPhase::Trees);
        let trees = build_all_spr_trees(config)?;

        on_progress(BuildPhase::Layout);
        let node_streets: Vec<Vec<Street>> = trees
            .iter()
            .map(annotate_streets)
            .collect();

        let num_b = buckets.num_flop_buckets as usize;

        let spr_layouts = build_per_spr_layouts(
            &trees,
            &node_streets,
            num_b,
            buckets.num_turn_buckets as usize,
            buckets.num_river_buckets as usize,
        );

        let total_iters = config.postflop_solve_iterations as usize;
        let samples = if config.postflop_solve_samples > 0 {
            config.postflop_solve_samples as usize
        } else {
            num_b
        };
        let num_sprs = trees.len();
        let total_steps = total_iters * num_sprs;
        on_progress(BuildPhase::SolvingPostflop(0, total_steps));

        let spr_strategy_sums = solve_postflop_per_spr(
            &trees,
            &spr_layouts,
            &bucket_equity,
            num_b,
            total_iters,
            samples,
            |step, total| on_progress(BuildPhase::SolvingPostflop(step, total)),
        );

        on_progress(BuildPhase::ComputingValues);
        let values = compute_postflop_values(
            &trees,
            &spr_layouts,
            &bucket_equity,
            &spr_strategy_sums,
            num_b,
        );

        Ok(Self {
            board,
            buckets,
            bucket_equity,
            trees,
            values,
            canonical_sprs: config.canonical_sprs.clone(),
        })
    }
}

/// Try loading abstraction data from cache; build and save on miss.
fn load_or_build_abstraction(
    config: &PostflopModelConfig,
    equity_table: Option<&EquityTable>,
    cache_base: Option<&std::path::Path>,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Result<(BoardAbstraction, HandBucketMapping, BucketEquity), PostflopAbstractionError> {
    let key = abstraction_cache::cache_key(config, equity_table.is_some());

    if let Some(base) = cache_base
        && let Some(cached) = abstraction_cache::load(base, &key)
    {
        let dir = abstraction_cache::cache_dir(base, &key);
        eprintln!("Abstraction cache hit: {}", dir.display());
        return Ok(cached);
    }

    on_progress(BuildPhase::BoardAbstraction);
    let board = build_board_abstraction(config)?;

    on_progress(BuildPhase::HandBuckets(0, hand_buckets::NUM_HANDS));
    let (buckets, bucket_equity) =
        build_hand_buckets_and_equity(config, &board, equity_table, on_progress)?;

    on_progress(BuildPhase::EquityTable);

    if let Some(base) = cache_base {
        abstraction_cache::save(base, &key, &board, &buckets, &bucket_equity)?;
        let dir = abstraction_cache::cache_dir(base, &key);
        eprintln!("Abstraction cache saved: {}", dir.display());
    }

    Ok((board, buckets, bucket_equity))
}

fn build_board_abstraction(
    config: &PostflopModelConfig,
) -> Result<BoardAbstraction, super::board_abstraction::BoardAbstractionError> {
    let ba_config = BoardAbstractionConfig {
        num_flop_textures: config.num_flop_textures,
        num_turn_transitions: config.num_turn_transitions,
        num_river_transitions: config.num_river_transitions,
        kmeans_max_iter: 50,
    };
    BoardAbstraction::build(&ba_config)
}

/// Build hand buckets and bucket equity table from EHS features.
///
/// When `equity_table` is provided, derives bucket equity from the 169×169
/// pairwise equities (accurate). Otherwise falls back to EHS centroid formula.
fn build_hand_buckets_and_equity(
    config: &PostflopModelConfig,
    board: &BoardAbstraction,
    equity_table: Option<&EquityTable>,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> Result<(HandBucketMapping, BucketEquity), hand_buckets::BucketError> {
    use crate::hands::all_hands;

    let flop_samples: Vec<Vec<[crate::poker::Card; 3]>> = board
        .prototype_flops
        .iter()
        .map(|f| vec![*f])
        .collect();

    let num_buckets = config.num_hand_buckets_flop;
    let num_textures = flop_samples.len();
    hand_buckets::validate_buckets(num_buckets, num_textures as u16)?;

    let hands: Vec<_> = all_hands().collect();
    let total = hand_buckets::NUM_HANDS;
    let features = hand_buckets::compute_all_flop_features(
        &hands,
        &flop_samples,
        &|done| on_progress(BuildPhase::HandBuckets(done, total)),
    );

    let flop_buckets = hand_buckets::cluster_global(&features, num_buckets, num_textures);

    hand_buckets::log_bucket_diagnostics(
        &hands, &features, &flop_buckets, num_buckets, &board.prototype_flops,
    );

    let bucket_equity = if let Some(eq_table) = equity_table {
        // Use real pairwise equities — much more accurate.
        hand_buckets::build_bucket_equity_from_equity_table(
            &flop_buckets,
            eq_table,
            num_buckets as usize,
        )
    } else {
        // Fallback: derive from EHS centroids.
        let centroids =
            hand_buckets::bucket_ehs_centroids(&features, &flop_buckets, num_buckets as usize);
        hand_buckets::build_bucket_equity_from_centroids(&centroids)
    };

    let mapping = HandBucketMapping {
        num_flop_buckets: num_buckets,
        num_turn_buckets: num_buckets,
        num_river_buckets: num_buckets,
        turn_buckets: flop_buckets.clone(),
        river_buckets: flop_buckets.clone(),
        flop_buckets,
    };

    Ok((mapping, bucket_equity))
}

fn build_all_spr_trees(
    config: &PostflopModelConfig,
) -> Result<Vec<PostflopTree>, super::postflop_tree::PostflopTreeError> {
    config.canonical_sprs
        .iter()
        .map(|&spr| PostflopTree::build_with_spr(config, spr))
        .collect()
}

// ──────────────────────────────────────────────────────────────────────────────
// Postflop pre-solve: run CFR to convergence, then extract value table
// ──────────────────────────────────────────────────────────────────────────────

/// Build a separate `PostflopLayout` for each SPR so each solve uses a small
/// isolated buffer instead of one monolithic buffer for all SPRs.
fn build_per_spr_layouts(
    trees: &[PostflopTree],
    node_streets: &[Vec<Street>],
    num_flop_buckets: usize,
    num_turn_buckets: usize,
    num_river_buckets: usize,
) -> Vec<PostflopLayout> {
    trees.iter().enumerate().map(|(i, tree)| {
        PostflopLayout::build(
            tree,
            &node_streets[i],
            num_flop_buckets,
            num_turn_buckets,
            num_river_buckets,
        )
    }).collect()
}

/// Solve all SPRs in parallel, each with its own small buffer.
///
/// Returns per-SPR strategy sums.
#[allow(clippy::cast_possible_truncation)]
fn solve_postflop_per_spr(
    trees: &[PostflopTree],
    spr_layouts: &[PostflopLayout],
    bucket_equity: &BucketEquity,
    num_buckets: usize,
    num_iterations: usize,
    samples_per_iter: usize,
    on_progress: impl Fn(usize, usize) + Sync,
) -> Vec<Vec<f64>> {
    let num_sprs = trees.len();
    let total_steps = num_iterations * num_sprs;
    let use_exhaustive = num_buckets * num_buckets <= samples_per_iter;
    let actual_pairs = if use_exhaustive {
        num_buckets * num_buckets
    } else {
        samples_per_iter
    };

    for (spr_idx, tree) in trees.iter().enumerate() {
        let buf_size = spr_layouts[spr_idx].total_size;
        #[allow(clippy::cast_precision_loss)]
        let mb = buf_size as f64 * 8.0 / 1_000_000.0;
        let mode = if use_exhaustive { "exhaustive" } else { "sampled" };
        eprintln!(
            "  SPR {:.1}: {} nodes, {buf_size} slots ({mb:.1} MB), {mode} {actual_pairs} pairs/iter",
            tree.spr,
            tree.node_count(),
        );
    }

    let progress = AtomicUsize::new(0);

    (0..num_sprs)
        .into_par_iter()
        .map(|spr_idx| {
            let layout = &spr_layouts[spr_idx];
            solve_one_spr(
                trees,
                layout,
                bucket_equity,
                spr_idx,
                num_buckets,
                layout.total_size,
                num_iterations,
                samples_per_iter,
                &progress,
                total_steps,
                &on_progress,
            )
        })
        .collect()
}

/// Run CFR for a single SPR with parallel inner loop.
///
/// Uses exhaustive bucket-pair enumeration when N² ≤ `samples_per_iter`
/// (eliminates sampling variance), otherwise falls back to random sampling.
/// Reads `regret_sum` directly each iteration (no per-iteration clone needed
/// since it is immutable within the inner loop).
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn solve_one_spr(
    trees: &[PostflopTree],
    layout: &PostflopLayout,
    bucket_equity: &BucketEquity,
    spr_idx: usize,
    num_buckets: usize,
    buf_size: usize,
    num_iterations: usize,
    samples_per_iter: usize,
    progress: &AtomicUsize,
    total_steps: usize,
    on_progress: &(impl Fn(usize, usize) + Sync),
) -> Vec<f64> {
    let mut regret_sum = vec![0.0f64; buf_size];
    let mut strategy_sum = vec![0.0f64; buf_size];
    let use_exhaustive = num_buckets * num_buckets <= samples_per_iter;

    for iter in 0..num_iterations {
        let step = progress.fetch_add(1, Ordering::Relaxed) + 1;
        on_progress(step, total_steps);

        let iteration = iter as u64 + 1;

        // Parallel fold: each rayon task accumulates its own (dr, ds) deltas,
        // reading regret_sum immutably (no clone needed).
        let (dr, ds) = if use_exhaustive {
            exhaustive_cfr_iteration(
                trees, layout, bucket_equity, &regret_sum,
                spr_idx, num_buckets, buf_size, iteration,
            )
        } else {
            sampled_cfr_iteration(
                trees, layout, bucket_equity, &regret_sum,
                spr_idx, num_buckets, buf_size, samples_per_iter, iteration, iter,
            )
        };

        add_buffers(&mut regret_sum, &dr);
        add_buffers(&mut strategy_sum, &ds);
    }

    strategy_sum
}

/// One CFR iteration over all N² bucket pairs, parallelised across hero buckets.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn exhaustive_cfr_iteration(
    trees: &[PostflopTree],
    layout: &PostflopLayout,
    bucket_equity: &BucketEquity,
    regret_sum: &[f64],
    spr_idx: usize,
    num_buckets: usize,
    buf_size: usize,
    iteration: u64,
) -> (Vec<f64>, Vec<f64>) {
    (0..num_buckets)
        .into_par_iter()
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), hb| {
                let hb = hb as u16;
                for ob in 0..num_buckets as u16 {
                    for hero_pos in 0..2u8 {
                        solve_cfr_traverse(
                            trees, layout, bucket_equity, regret_sum,
                            &mut dr, &mut ds,
                            spr_idx, 0, hb, ob, hero_pos, 1.0, 1.0, iteration,
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

/// One CFR iteration with random bucket-pair sampling, parallelised across chunks.
#[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
fn sampled_cfr_iteration(
    trees: &[PostflopTree],
    layout: &PostflopLayout,
    bucket_equity: &BucketEquity,
    regret_sum: &[f64],
    spr_idx: usize,
    num_buckets: usize,
    buf_size: usize,
    samples_per_iter: usize,
    iteration: u64,
    iter_idx: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Generate pairs sequentially (RNG is sequential), then process in parallel.
    let seed = iter_idx as u64 * 1_000_003 + spr_idx as u64;
    let mut rng = SmallRng::seed_from_u64(seed);
    let pairs: Vec<(u16, u16)> = (0..samples_per_iter)
        .map(|_| {
            let hb = rng.random_range(0..num_buckets as u16);
            let ob = rng.random_range(0..num_buckets as u16);
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
                            trees, layout, bucket_equity, regret_sum,
                            &mut dr, &mut ds,
                            spr_idx, 0, hb, ob, hero_pos, 1.0, 1.0, iteration,
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
#[allow(clippy::too_many_arguments)]
fn solve_cfr_traverse(
    trees: &[PostflopTree],
    layout: &PostflopLayout,
    bucket_equity: &BucketEquity,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    spr_idx: usize,
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    iteration: u64,
) -> f64 {
    let tree = &trees[spr_idx];

    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal { terminal_type, pot_fraction } => {
            postflop_terminal_value(*terminal_type, *pot_fraction, hero_bucket, opp_bucket, hero_pos, bucket_equity)
        }
        PostflopNode::Chance { children, weights, .. } => {
            children.iter().zip(weights.iter())
                .map(|(&child, &w)| {
                    w * solve_cfr_traverse(
                        trees, layout, bucket_equity, snapshot, dr, ds,
                        spr_idx, child, hero_bucket, opp_bucket, hero_pos,
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
                    trees, layout, bucket_equity, snapshot, dr, ds,
                    start, spr_idx, hero_bucket, opp_bucket, hero_pos,
                    reach_hero, reach_opp, children, &strategy[..num_actions], iteration,
                )
            } else {
                solve_traverse_opponent(
                    trees, layout, bucket_equity, snapshot, dr, ds,
                    spr_idx, hero_bucket, opp_bucket, hero_pos,
                    reach_hero, reach_opp, children, &strategy[..num_actions], iteration,
                )
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn solve_traverse_hero(
    trees: &[PostflopTree],
    layout: &PostflopLayout,
    bucket_equity: &BucketEquity,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    slot_start: usize,
    spr_idx: usize,
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
            trees, layout, bucket_equity, snapshot, dr, ds,
            spr_idx, child, hero_bucket, opp_bucket, hero_pos,
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
    trees: &[PostflopTree],
    layout: &PostflopLayout,
    bucket_equity: &BucketEquity,
    snapshot: &[f64],
    dr: &mut [f64],
    ds: &mut [f64],
    spr_idx: usize,
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
                trees, layout, bucket_equity, snapshot, dr, ds,
                spr_idx, child, hero_bucket, opp_bucket, hero_pos,
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

// ──────────────────────────────────────────────────────────────────────────────
// Value table computation from converged strategy
// ──────────────────────────────────────────────────────────────────────────────

/// Compute EV table from per-SPR converged strategy sums.
#[allow(clippy::cast_possible_truncation)]
fn compute_postflop_values(
    trees: &[PostflopTree],
    spr_layouts: &[PostflopLayout],
    bucket_equity: &BucketEquity,
    spr_strategy_sums: &[Vec<f64>],
    num_buckets: usize,
) -> PostflopValues {
    let num_sprs = trees.len();
    let total = num_sprs * 2 * num_buckets * num_buckets;
    let mut values = vec![0.0f64; total];

    for spr_idx in 0..num_sprs {
        let layout = &spr_layouts[spr_idx];
        let strategy_sum = &spr_strategy_sums[spr_idx];
        for hero_pos in 0..2u8 {
            for hero_bucket in 0..num_buckets as u16 {
                for opp_bucket in 0..num_buckets as u16 {
                    let ev = eval_with_avg_strategy(
                        trees, layout, bucket_equity, strategy_sum,
                        spr_idx, 0, hero_bucket, opp_bucket, hero_pos,
                    );
                    let idx = spr_idx * 2 * num_buckets * num_buckets
                        + (hero_pos as usize) * num_buckets * num_buckets
                        + (hero_bucket as usize) * num_buckets
                        + opp_bucket as usize;
                    values[idx] = ev;
                }
            }
        }
    }

    PostflopValues { values, num_buckets, num_sprs }
}

/// Walk tree using averaged (converged) strategy, returning hero EV fraction.
#[allow(clippy::too_many_arguments)]
fn eval_with_avg_strategy(
    trees: &[PostflopTree],
    layout: &PostflopLayout,
    bucket_equity: &BucketEquity,
    strategy_sum: &[f64],
    spr_idx: usize,
    node_idx: u32,
    hero_bucket: u16,
    opp_bucket: u16,
    hero_pos: u8,
) -> f64 {
    let tree = &trees[spr_idx];

    match &tree.nodes[node_idx as usize] {
        PostflopNode::Terminal { terminal_type, pot_fraction } => {
            postflop_terminal_value(*terminal_type, *pot_fraction, hero_bucket, opp_bucket, hero_pos, bucket_equity)
        }
        PostflopNode::Chance { children, weights, .. } => {
            children.iter().zip(weights.iter())
                .map(|(&child, &w)| {
                    w * eval_with_avg_strategy(
                        trees, layout, bucket_equity, strategy_sum,
                        spr_idx, child, hero_bucket, opp_bucket, hero_pos,
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
                        trees, layout, bucket_equity, strategy_sum,
                        spr_idx, child, hero_bucket, opp_bucket, hero_pos,
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
                // Hero folded: loses half the pot_fraction
                -pot_fraction / 2.0
            } else {
                // Opponent folded: hero wins half the pot_fraction
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
            num_turn_transitions: 1,
            num_river_transitions: 1,
            bet_sizes: vec![1.0],
            raises_per_street: 0,
            ..PostflopModelConfig::fast()
        };
        let tree = PostflopTree::build_with_spr(&config, 5.0).unwrap();
        let streets = annotate_streets(&tree);

        // Find a chance node and verify its children are on the next street.
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
                // The chance node itself should be on the prior street
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
            2.0,
            0,
            0,
            0,
            &eq,
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
            2.0,
            0,
            0,
            0,
            &eq,
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
            2.0,
            0,
            0,
            0,
            &eq,
        );
        // eq=0.7, pot_frac=2.0: 0.7*2.0 - 1.0 = 0.4
        // f32→f64 conversion introduces ~1e-8 error
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
            num_turn_transitions: 1,
            num_river_transitions: 1,
            bet_sizes: vec![1.0],
            raises_per_street: 0,
            ..PostflopModelConfig::fast()
        };
        let trees = build_all_spr_trees(&config).unwrap();
        let node_streets: Vec<Vec<Street>> = trees
            .iter()
            .map(annotate_streets)
            .collect();
        let layouts = build_per_spr_layouts(&trees, &node_streets, 10, 10, 10);

        // Check each layout has nonzero size
        for layout in &layouts {
            assert!(layout.total_size > 0, "layout should have nonzero size");
        }

        // Slot for root node, bucket 0 in first tree should be valid
        let (base, num_actions) = layouts[0].slot(0, 0);
        assert!(num_actions > 0);
        assert!(base + num_actions <= layouts[0].total_size);
    }

    #[timed_test]
    fn build_with_canonical_sprs_produces_different_trees() {
        let config = PostflopModelConfig {
            canonical_sprs: vec![1.0, 10.0],
            ..PostflopModelConfig::fast()
        };
        let trees = build_all_spr_trees(&config).unwrap();
        assert_eq!(trees.len(), 2);
        assert_ne!(trees[0].node_count(), trees[1].node_count(),
            "SPR 1.0 and 10.0 should produce different tree sizes");
    }

    #[timed_test]
    fn postflop_values_get_by_spr_index() {
        let num_sprs = 2;
        let num_buckets = 2;
        let total = num_sprs * 2 * num_buckets * num_buckets;
        let mut values = vec![0.0; total];
        let idx = 1 * 2 * 4 + 0 * 4 + 1 * 2 + 0;
        values[idx] = 0.42;
        let pv = PostflopValues { values, num_buckets, num_sprs };
        assert!((pv.get_by_spr(1, 0, 1, 0) - 0.42).abs() < 1e-9);
    }

    #[timed_test]
    fn build_all_spr_trees_creates_one_per_canonical_spr() {
        let config = PostflopModelConfig {
            canonical_sprs: vec![0.5, 1.0, 5.0],
            ..PostflopModelConfig::fast()
        };
        let trees = build_all_spr_trees(&config).unwrap();
        assert_eq!(trees.len(), 3);
    }
}
