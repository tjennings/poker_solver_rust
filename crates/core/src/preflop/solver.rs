//! CFR solver for preflop game trees with alternating updates.
//!
//! Supports three variants via [`CfrVariant`]:
//! - **Vanilla**: standard CFR with uniform iteration weighting and no discounting.
//! - **DCFR**: discounted CFR with α/β/γ discounting and linear (LCFR) iteration weighting.
//! - **CFR+**: regrets floored to zero each iteration, linear strategy weighting.
//!
//! Both variants perform full-game CFR over 169 canonical hand matchups with
//! alternating player traversals. Each iteration is parallelized via rayon:
//! a frozen regret snapshot is shared across threads, with flat buffer deltas
//! merged via parallel reduce.
//!
//! Storage uses flat `Vec<f64>` buffers indexed by `(node, hand, action)`
//! for cache-friendly access and fast clone/merge operations.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::{CfrVariant, PreflopConfig};
use super::equity::EquityTable;
use super::postflop_abstraction::PostflopAbstraction;
use super::postflop_tree::PotType;
use super::tree::{PreflopAction, PreflopNode, PreflopTree, TerminalType};

const NUM_HANDS: usize = 169;
const MAX_ACTIONS: usize = 8;

/// Extracted average strategy from Linear CFR training.
///
/// Key = packed `(node_index, hand_index)`. Value = action probabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreflopStrategy {
    strategies: FxHashMap<u64, Vec<f64>>,
}

impl PreflopStrategy {
    /// Pack `(node_idx, hand_idx)` into a single `u64` key.
    fn key(node_idx: u32, hand_idx: u16) -> u64 {
        (u64::from(node_idx) << 16) | u64::from(hand_idx)
    }

    /// Action probabilities for a hand at the root node (index 0).
    #[must_use]
    pub fn get_root_probs(&self, hand_idx: usize) -> Vec<f64> {
        self.get_probs(0, hand_idx)
    }

    /// Action probabilities for a hand at any node.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_probs(&self, node_idx: u32, hand_idx: usize) -> Vec<f64> {
        // Safe: hand_idx is always < 169, well within u16 range
        self.strategies
            .get(&Self::key(node_idx, hand_idx as u16))
            .cloned()
            .unwrap_or_default()
    }

    /// Consume self and return the inner strategy map.
    #[must_use]
    pub fn into_inner(self) -> FxHashMap<u64, Vec<f64>> {
        self.strategies
    }

    /// Number of info sets with stored strategies.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    /// Whether the strategy map is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }
}

/// Per-node investment amounts for each player (HU only: [p0, p1]).
type NodeInvestments = Vec<[u32; 2]>;

/// Pre-computed layout mapping `(node, hand)` pairs to flat buffer offsets.
///
/// Each decision node reserves `NUM_HANDS * num_actions` contiguous slots.
/// Terminal nodes have zero-width regions.
struct NodeLayout {
    /// `(offset, num_actions)` per tree node. Terminals have `num_actions = 0`.
    entries: Vec<(usize, usize)>,
    /// Total buffer size in `f64` elements.
    total_size: usize,
}

impl NodeLayout {
    fn from_tree(tree: &PreflopTree) -> Self {
        let mut entries = Vec::with_capacity(tree.nodes.len());
        let mut offset = 0;
        for node in &tree.nodes {
            let num_actions = match node {
                PreflopNode::Decision { children, .. } => children.len(),
                PreflopNode::Terminal { .. } => 0,
            };
            entries.push((offset, num_actions));
            offset += NUM_HANDS * num_actions;
        }
        Self {
            entries,
            total_size: offset,
        }
    }

    /// Start index and action count for a `(node, hand)` pair.
    #[inline]
    fn slot(&self, node_idx: u32, hand_idx: u16) -> (usize, usize) {
        let (base, n) = self.entries[node_idx as usize];
        (base + (hand_idx as usize) * n, n)
    }
}

/// Immutable context shared across all traversals in one iteration.
struct Ctx<'a> {
    tree: &'a PreflopTree,
    investments: &'a NodeInvestments,
    equity: &'a EquityTable,
    layout: &'a NodeLayout,
    snapshot: &'a [f64],
    /// ε-greedy exploration factor (0.0 = pure regret matching).
    exploration: f64,
    /// Current iteration number (1-based). Used for LCFR linear weighting
    /// of strategy contributions: `ds += iteration * reach * strategy`.
    iteration: u64,
    /// CFR variant controlling iteration weighting.
    cfr_variant: CfrVariant,
    /// Per-player starting stacks (SB units).
    stacks: [u32; 2],
    /// Optional postflop abstraction for modeling postflop play at showdown terminals.
    postflop: Option<&'a PostflopState>,
}

/// Postflop state: pre-solved value table + raise counts for pot-type classification.
pub(crate) struct PostflopState {
    pub(crate) abstraction: PostflopAbstraction,
    /// Per-node raise counts for the preflop tree (to determine `PotType`).
    pub(crate) raise_counts: Vec<u8>,
}

/// Preflop CFR solver with alternating updates.
///
/// Traverses the preflop tree for every 169x169 canonical hand matchup
/// using alternating player traversals. Supports vanilla CFR (no discounting),
/// DCFR (α/β/γ discounting with linear iteration weighting), and CFR+
/// (regrets floored to zero, linear strategy weighting).
pub struct PreflopSolver {
    tree: PreflopTree,
    equity: EquityTable,
    investments: NodeInvestments,
    layout: NodeLayout,
    /// Pre-computed valid `(hand_i, hand_j)` pairs with non-zero weight.
    pairs: Vec<(u16, u16)>,
    /// Cumulative regrets, flat buffer indexed by `layout.slot()`.
    regret_sum: Vec<f64>,
    /// Cumulative weighted strategy, flat buffer indexed by `layout.slot()`.
    strategy_sum: Vec<f64>,
    iteration: u64,
    /// Instantaneous regret from the most recent iteration's traversal.
    /// Used for convergence metrics instead of cumulative regret sums,
    /// which are unsuitable under DCFR's asymmetric discounting.
    last_instantaneous_regret: Vec<f64>,
    /// CFR variant (Vanilla, DCFR, or CFR+).
    cfr_variant: CfrVariant,
    /// DCFR positive regret discount exponent.
    dcfr_alpha: f64,
    /// DCFR negative regret discount exponent.
    dcfr_beta: f64,
    /// DCFR strategy sum discount exponent.
    dcfr_gamma: f64,
    /// Number of initial iterations without DCFR discounting.
    dcfr_warmup: u64,
    /// ε-greedy exploration factor.
    exploration: f64,
    /// Per-player starting stacks (SB units).
    stacks: [u32; 2],
    /// Optional postflop model state.
    postflop: Option<PostflopState>,
    /// Reusable snapshot buffer for frozen regrets (avoids per-iteration allocation).
    snapshot_buf: Vec<f64>,
}

impl PreflopSolver {
    /// Create a solver with a uniform equity table (all matchups = 0.5).
    #[must_use]
    pub fn new(config: &PreflopConfig) -> Self {
        Self::new_with_equity(config, EquityTable::new_uniform())
    }

    /// Create a solver with a custom precomputed equity table.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn new_with_equity(config: &PreflopConfig, equity: EquityTable) -> Self {
        let tree = PreflopTree::build(config);
        let investments = precompute_investments(&tree, config);
        let layout = NodeLayout::from_tree(&tree);
        // Safe: h1, h2 always < 169, well within u16
        let pairs: Vec<(u16, u16)> = (0..NUM_HANDS)
            .flat_map(|h1| (0..NUM_HANDS).map(move |h2| (h1, h2)))
            .filter(|&(h1, h2)| equity.weight(h1, h2) > 0.0)
            .map(|(h1, h2)| (h1 as u16, h2 as u16))
            .collect();
        let buf_size = layout.total_size;
        Self {
            tree,
            equity,
            investments,
            layout,
            pairs,
            regret_sum: vec![0.0; buf_size],
            strategy_sum: vec![0.0; buf_size],
            iteration: 0,
            last_instantaneous_regret: Vec::new(),
            cfr_variant: config.cfr_variant,
            dcfr_alpha: config.dcfr_alpha,
            dcfr_beta: config.dcfr_beta,
            dcfr_gamma: config.dcfr_gamma,
            dcfr_warmup: config.dcfr_warmup,
            exploration: config.exploration,
            stacks: [
                config.stacks.first().copied().unwrap_or(0),
                config.stacks.get(1).copied().unwrap_or(0),
            ],
            postflop: None,
            snapshot_buf: vec![0.0; buf_size],
        }
    }

    /// Attach a precomputed postflop abstraction to improve showdown values.
    ///
    /// When attached, showdown terminals in the preflop tree will use postflop
    /// CFR traversal instead of raw equity lookup.
    pub fn attach_postflop(&mut self, abstraction: PostflopAbstraction, _config: &PreflopConfig) {
        let raise_counts = precompute_raise_counts(&self.tree);
        self.postflop = Some(PostflopState {
            abstraction,
            raise_counts,
        });
    }

    /// Run `iterations` of DCFR training with alternating updates.
    pub fn train(&mut self, iterations: u64) {
        for _ in 0..iterations {
            self.iteration += 1;
            self.train_one_iteration();
        }
    }

    /// Current iteration count.
    #[must_use]
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// The game tree used by this solver.
    #[must_use]
    pub fn tree(&self) -> &PreflopTree {
        &self.tree
    }

    /// Extract the average strategy (normalized `strategy_sum`).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn strategy(&self) -> PreflopStrategy {
        let mut strategies = FxHashMap::default();
        for (node_idx, node) in self.tree.nodes.iter().enumerate() {
            let num_actions = match node {
                PreflopNode::Decision { children, .. } => children.len(),
                PreflopNode::Terminal { .. } => continue,
            };
            // Safe: hand_idx < 169, node_idx fits u32 for any reasonable tree
            for hand_idx in 0..NUM_HANDS {
                let (start, _) = self.layout.slot(node_idx as u32, hand_idx as u16);
                let sums = &self.strategy_sum[start..start + num_actions];
                let key = PreflopStrategy::key(node_idx as u32, hand_idx as u16);
                strategies.insert(key, normalize(sums));
            }
        }
        PreflopStrategy { strategies }
    }

    /// Raw regret sums for a hand at a specific node (for diagnostics).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn regret_at(&self, node_idx: u32, hand_idx: usize) -> Vec<f64> {
        let num_actions = match &self.tree.nodes[node_idx as usize] {
            PreflopNode::Decision { children, .. } => children.len(),
            PreflopNode::Terminal { .. } => return vec![],
        };
        let (start, _) = self.layout.slot(node_idx, hand_idx as u16);
        self.regret_sum[start..start + num_actions].to_vec()
    }

    /// Raw strategy sums for a hand at a specific node (for diagnostics).
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn strategy_sum_at(&self, node_idx: u32, hand_idx: usize) -> Vec<f64> {
        let num_actions = match &self.tree.nodes[node_idx as usize] {
            PreflopNode::Decision { children, .. } => children.len(),
            PreflopNode::Terminal { .. } => return vec![],
        };
        let (start, _) = self.layout.slot(node_idx, hand_idx as u16);
        self.strategy_sum[start..start + num_actions].to_vec()
    }

    /// Average positive regret per slot.
    ///
    /// - **DCFR**: Uses the most recent iteration's instantaneous regret to avoid
    ///   inflation from asymmetric positive/negative discounting. Divides out the
    ///   LCFR iteration weight.
    /// - **Vanilla / CFR+**: Uses cumulative `regret_sum` divided by iteration count,
    ///   since there is no discounting bias. (CFR+ regrets are always ≥ 0.)
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_positive_regret(&self) -> f64 {
        if self.iteration == 0 {
            return 0.0;
        }

        match self.cfr_variant {
            CfrVariant::Vanilla | CfrVariant::CfrPlus => {
                if self.regret_sum.is_empty() {
                    return 0.0;
                }
                let mut total = 0.0;
                for &r in &self.regret_sum {
                    if r > 0.0 {
                        total += r;
                    }
                }
                total / self.regret_sum.len() as f64 / self.iteration as f64
            }
            CfrVariant::Dcfr | CfrVariant::Linear => {
                if self.last_instantaneous_regret.is_empty() {
                    return 0.0;
                }
                let mut total = 0.0;
                for &r in &self.last_instantaneous_regret {
                    if r > 0.0 {
                        total += r;
                    }
                }
                // The traversal weights regret deltas by `iteration` (LCFR linear
                // weighting), so divide it back out to get the unweighted regret.
                total / self.last_instantaneous_regret.len() as f64 / self.iteration as f64
            }
        }
    }

    /// Compute exploitability of the current average strategy.
    ///
    /// Returns exploitability per hand in SB chip units (SB=1, BB=2).
    /// Multiply by 500 to convert to mBB/hand.
    /// A value of 0 means the strategy is a Nash equilibrium.
    #[must_use]
    pub fn exploitability(&self) -> f64 {
        super::exploitability::compute_exploitability(
            &self.tree,
            &self.strategy(),
            &self.equity,
            &self.investments,
            self.postflop.as_ref(),
            self.stacks,
        )
    }

    /// Compute DCFR strategy discount: `(t / (t + 1))^γ`.
    #[allow(clippy::cast_precision_loss)]
    fn strategy_discount(&self) -> f64 {
        let t = self.iteration as f64;
        (t / (t + 1.0)).powf(self.dcfr_gamma)
    }

    /// Apply DCFR cumulative regret discounting to the hero's nodes only.
    ///
    /// With alternating updates, only the traversing player's regrets are
    /// updated each iteration. Discounting must match: only discount the
    /// hero's regrets so each player's regrets are discounted once per
    /// their update, matching standard simultaneous-update DCFR behavior.
    ///
    /// Positive regrets multiplied by `t^α / (t^α + 1)`.
    /// Negative regrets multiplied by `t^β / (t^β + 1)`.
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn discount_regrets(&mut self, hero_pos: u8) {
        let t = (self.iteration + 1) as f64;
        let pos_factor = t.powf(self.dcfr_alpha) / (t.powf(self.dcfr_alpha) + 1.0);
        let neg_factor = t.powf(self.dcfr_beta) / (t.powf(self.dcfr_beta) + 1.0);
        for (node_idx, node) in self.tree.nodes.iter().enumerate() {
            let (position, num_actions) = match node {
                PreflopNode::Decision {
                    position, children, ..
                } => (*position, children.len()),
                PreflopNode::Terminal { .. } => continue,
            };
            if position != hero_pos {
                continue;
            }
            let (base, _) = self.layout.slot(node_idx as u32, 0);
            let end = base + NUM_HANDS * num_actions;
            for r in &mut self.regret_sum[base..end] {
                if *r > 0.0 {
                    *r *= pos_factor;
                } else if *r < 0.0 {
                    *r *= neg_factor;
                }
            }
        }
    }

    /// Multiplicatively discount existing strategy sums for the hero's nodes.
    ///
    /// Standard DCFR: `S^{T+1} = sd * S^T + new_contribution`.
    /// This ensures early (poor) strategies get exponentially washed out.
    #[allow(clippy::cast_possible_truncation)]
    fn discount_strategy_sums(&mut self, hero_pos: u8, sd: f64) {
        for (node_idx, node) in self.tree.nodes.iter().enumerate() {
            let (position, num_actions) = match node {
                PreflopNode::Decision {
                    position, children, ..
                } => (*position, children.len()),
                PreflopNode::Terminal { .. } => continue,
            };
            if position != hero_pos {
                continue;
            }
            let (base, _) = self.layout.slot(node_idx as u32, 0);
            let end = base + NUM_HANDS * num_actions;
            for s in &mut self.strategy_sum[base..end] {
                *s *= sd;
            }
        }
    }

    /// Single training iteration: snapshot regrets, parallel traverse, merge, discount.
    ///
    /// Uses **simultaneous updates**: both players traverse every iteration
    /// using the same frozen regret snapshot.
    #[allow(clippy::cast_precision_loss)]
    fn train_one_iteration(&mut self) {
        self.snapshot_buf.clone_from(&self.regret_sum);

        let ctx = Ctx {
            tree: &self.tree,
            investments: &self.investments,
            equity: &self.equity,
            layout: &self.layout,
            snapshot: &self.snapshot_buf,
            exploration: self.exploration,
            iteration: self.iteration,
            cfr_variant: self.cfr_variant,
            stacks: self.stacks,
            postflop: self.postflop.as_ref(),
        };

        let (mr, ms) = parallel_traverse(&ctx, &self.pairs);

        self.apply_discounting();
        add_into(&mut self.regret_sum, &mr);
        add_into(&mut self.strategy_sum, &ms);
        if self.cfr_variant == CfrVariant::CfrPlus {
            self.floor_regrets();
        }
        self.last_instantaneous_regret = mr;
    }

    /// Apply DCFR discounting to both players' cumulative values.
    /// Skipped entirely for Vanilla CFR and CFR+.
    fn apply_discounting(&mut self) {
        if !matches!(self.cfr_variant, CfrVariant::Dcfr | CfrVariant::Linear) {
            return;
        }
        if self.iteration > self.dcfr_warmup {
            let sd = self.strategy_discount();
            for pos in 0..2u8 {
                self.discount_regrets(pos);
                self.discount_strategy_sums(pos, sd);
            }
        }
    }

    /// Floor all cumulative regrets to zero (CFR+ update rule).
    fn floor_regrets(&mut self) {
        for r in &mut self.regret_sum {
            if *r < 0.0 {
                *r = 0.0;
            }
        }
    }
}

/// Parallel fold+reduce over all hand pairs, returning merged deltas.
fn parallel_traverse(
    ctx: &Ctx<'_>,
    pairs: &[(u16, u16)],
) -> (Vec<f64>, Vec<f64>) {
    let buf_size = ctx.layout.total_size;

    pairs
        .par_iter()
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), &(h1, h2)| {
                let w = ctx.equity.weight(h1 as usize, h2 as usize);
                for hero_pos in 0..2u8 {
                    let (hh, oh) = if hero_pos == 0 { (h1, h2) } else { (h2, h1) };
                    cfr_traverse(ctx, &mut dr, &mut ds, 0, hh, oh, hero_pos, 1.0, w);
                }
                (dr, ds)
            },
        )
        .reduce(
            || (vec![0.0; buf_size], vec![0.0; buf_size]),
            |(mut ar, mut a_s), (br, bs)| {
                add_into(&mut ar, &br);
                add_into(&mut a_s, &bs);
                (ar, a_s)
            },
        )
}

/// Element-wise `dst[i] += src[i]`.
#[inline]
fn add_into(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}

/// Recursive CFR traversal. Returns expected value for the hero player.
///
/// Reads strategy from `ctx.snapshot` (frozen for the iteration),
/// writes regret and strategy deltas to flat `dr` / `ds` buffers.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn cfr_traverse(
    ctx: &Ctx<'_>,
    dr: &mut [f64],
    ds: &mut [f64],
    node_idx: u32,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
) -> f64 {
    let inv = ctx.investments[node_idx as usize];
    let hero_inv = f64::from(inv[hero_pos as usize]);

    match &ctx.tree.nodes[node_idx as usize] {
        PreflopNode::Terminal { terminal_type, pot } => match terminal_type {
            TerminalType::Showdown => {
                if let Some(pf_state) = ctx.postflop {
                    let eq = ctx.equity.equity(hero_hand as usize, opp_hand as usize);
                    postflop_showdown_value(
                        pf_state, node_idx, *pot, hero_inv,
                        hero_hand, opp_hand, hero_pos,
                        eq, ctx.stacks,
                    )
                } else {
                    terminal_value(
                        *terminal_type, *pot, hero_inv,
                        hero_hand, opp_hand, hero_pos, ctx.equity,
                    )
                }
            }
            TerminalType::Fold { .. } => terminal_value(
                *terminal_type, *pot, hero_inv,
                hero_hand, opp_hand, hero_pos, ctx.equity,
            ),
        },
        PreflopNode::Decision {
            position, children, ..
        } => {
            let num_actions = children.len();
            let is_hero = *position == hero_pos;
            let hand_for_key = if is_hero { hero_hand } else { opp_hand };
            let (start, _) = ctx.layout.slot(node_idx, hand_for_key);

            let mut intended = [0.0f64; MAX_ACTIONS];
            regret_matching_into(ctx.snapshot, start, &mut intended[..num_actions]);

            if is_hero {
                // Epsilon-greedy exploration: only applied to the traversing
                // player's strategy. The opponent plays pure regret-matched.
                let mut traversal = intended;
                if ctx.exploration > 0.0 {
                    let eps = ctx.exploration;
                    #[allow(clippy::cast_precision_loss)]
                    let n_inv = 1.0 / num_actions as f64;
                    for s in &mut traversal[..num_actions] {
                        *s = (1.0 - eps).mul_add(*s, eps * n_inv);
                    }
                }
                traverse_hero(
                    ctx, dr, ds, start, hero_hand, opp_hand, hero_pos,
                    reach_hero, reach_opp, children,
                    &intended[..num_actions], &traversal[..num_actions],
                )
            } else {
                traverse_opponent(
                    ctx, dr, ds, hero_hand, opp_hand, hero_pos,
                    reach_hero, reach_opp, children, &intended[..num_actions],
                )
            }
        }
    }
}

/// Hero's decision: compute regrets and update strategy/regret deltas.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn traverse_hero(
    ctx: &Ctx<'_>,
    dr: &mut [f64],
    ds: &mut [f64],
    slot_start: usize,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    children: &[u32],
    intended: &[f64],
    traversal: &[f64],
) -> f64 {
    let num_actions = children.len();
    let mut action_values = [0.0f64; MAX_ACTIONS];
    for (i, &child_idx) in children.iter().enumerate() {
        action_values[i] = cfr_traverse(
            ctx, dr, ds, child_idx, hero_hand, opp_hand, hero_pos,
            reach_hero * traversal[i], reach_opp,
        );
    }

    let node_value: f64 = traversal
        .iter()
        .zip(&action_values[..num_actions])
        .map(|(s, v)| s * v)
        .sum();

    // Regret weighting: DCFR uses linear (LCFR), Vanilla and CFR+ use uniform.
    // Strategy weighting: DCFR and CFR+ use linear, Vanilla uses uniform.
    #[allow(clippy::cast_precision_loss)]
    let (regret_weight, strategy_weight) = match ctx.cfr_variant {
        CfrVariant::Vanilla => (1.0, 1.0),
        CfrVariant::Dcfr | CfrVariant::Linear => {
            let w = ctx.iteration as f64;
            (w, w)
        }
        CfrVariant::CfrPlus => (1.0, ctx.iteration as f64),
    };

    for (i, val) in action_values[..num_actions].iter().enumerate() {
        dr[slot_start + i] += regret_weight * reach_opp * (val - node_value);
    }
    for (i, &s) in intended.iter().enumerate() {
        ds[slot_start + i] += strategy_weight * reach_hero * s;
    }

    node_value
}

/// Opponent's decision: traverse using opponent's strategy.
#[allow(clippy::too_many_arguments, clippy::similar_names)]
fn traverse_opponent(
    ctx: &Ctx<'_>,
    dr: &mut [f64],
    ds: &mut [f64],
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    children: &[u32],
    strategy: &[f64],
) -> f64 {
    let mut node_value = 0.0f64;
    for (i, &child_idx) in children.iter().enumerate() {
        let child_value = cfr_traverse(
            ctx, dr, ds, child_idx, hero_hand, opp_hand, hero_pos,
            reach_hero, reach_opp * strategy[i],
        );
        node_value += strategy[i] * child_value;
    }
    node_value
}

/// Compute hero's utility at a terminal node.
pub(crate) fn terminal_value(
    terminal_type: TerminalType,
    pot: u32,
    hero_inv: f64,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    equity: &EquityTable,
) -> f64 {
    match terminal_type {
        TerminalType::Fold { folder } => {
            if folder == hero_pos {
                -hero_inv
            } else {
                f64::from(pot) - hero_inv
            }
        }
        TerminalType::Showdown => {
            let eq = equity.equity(hero_hand as usize, opp_hand as usize);
            eq * f64::from(pot) - hero_inv
        }
    }
}

/// Regret matching: write positive-regret-normalized strategy into `out`.
#[allow(clippy::cast_precision_loss)]
fn regret_matching_into(regret_buf: &[f64], start: usize, out: &mut [f64]) {
    let num_actions = out.len();
    let mut positive_sum = 0.0f64;

    for (i, s) in out.iter_mut().enumerate() {
        let val = regret_buf[start + i];
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
        // Safe: num_actions is small (< MAX_ACTIONS)
        let uniform = 1.0 / num_actions as f64;
        out.fill(uniform);
    }
}

/// Normalize a cumulative strategy sum into a probability distribution.
#[allow(clippy::cast_precision_loss)]
fn normalize(sums: &[f64]) -> Vec<f64> {
    let total: f64 = sums.iter().sum();
    if total > 0.0 {
        sums.iter().map(|&s| s / total).collect()
    } else if sums.is_empty() {
        Vec::new()
    } else {
        // Safe: sums.len() is small (< 10)
        vec![1.0 / sums.len() as f64; sums.len()]
    }
}

/// Select the index of the closest SPR model to `actual_spr`.
///
/// `sprs` must be non-empty. Returns the index into `sprs` with the
/// smallest absolute distance to `actual_spr`.
fn select_closest_spr(sprs: &[f64], actual_spr: f64) -> usize {
    sprs.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - actual_spr).abs().total_cmp(&(*b - actual_spr).abs())
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Look up pre-solved postflop EV from the value table with SPR scaling.
///
/// The postflop model was solved at a fixed SPR (e.g. 3.5). When the actual
/// remaining stacks at a preflop terminal are shallower than the model assumed,
/// we interpolate between pure equity value and the model value based on the
/// ratio of actual SPR to model SPR. This prevents physically impossible
/// values when the model's SPR exceeds the actual stack depth.
///
/// Returns EV in the same units as `terminal_value` (chips relative to start).
#[allow(clippy::too_many_arguments, clippy::similar_names)]
pub(crate) fn postflop_showdown_value(
    pf_state: &PostflopState,
    preflop_node_idx: u32,
    pot: u32,
    hero_inv: f64,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    equity: f64,
    stacks: [u32; 2],
) -> f64 {
    let pot_f = f64::from(pot);

    // Pure equity value (fallback when no postflop play is possible).
    let eq_value = equity * pot_f - hero_inv;

    // The postflop model was trained for raised pots (PotType::Raised at a
    // fixed SPR).  For limped-pot showdowns (raise_count == 0) applying the
    // model inflates values by ~5× relative to fold terminals (e.g. AA showdown
    // ≈ 10.87 chips vs fold ≈ 2 chips), creating a degenerate limp-trap
    // equilibrium where SB never raises.  Fall back to raw equity here.
    let raise_count = pf_state
        .raise_counts
        .get(preflop_node_idx as usize)
        .copied()
        .unwrap_or(0);
    if raise_count == 0 {
        return eq_value;
    }

    // O(1) lookup into precomputed hand-averaged EV table.
    let pf_ev_frac = pf_state.abstraction.avg_ev(
        hero_pos, hero_hand as usize, opp_hand as usize,
    );

    // Model value (what the postflop model predicts at its trained SPR).
    let model_value = pf_ev_frac * pot_f + (pot_f / 2.0 - hero_inv);

    // Compute actual SPR at this terminal.
    let opp_inv = pot_f - hero_inv;
    let hero_remaining = f64::from(stacks[hero_pos as usize]) - hero_inv;
    let opp_remaining = f64::from(stacks[1 - hero_pos as usize]) - opp_inv;
    let effective_remaining = hero_remaining.min(opp_remaining).max(0.0);
    let actual_spr = if pot > 0 { effective_remaining / pot_f } else { 0.0 };

    let model_spr = pf_state.abstraction.spr;
    if model_spr <= 0.0 || actual_spr >= model_spr {
        return model_value;
    }

    // Interpolate: at actual_spr=0 use pure equity, at model_spr use full model.
    let ratio = actual_spr / model_spr;
    eq_value + (model_value - eq_value) * ratio
}

/// Precompute the raise count at every node in the preflop tree.
///
/// Walks from root, counting `Raise`/`AllIn` actions along each path.
fn precompute_raise_counts(tree: &PreflopTree) -> Vec<u8> {
    let mut counts = vec![0u8; tree.nodes.len()];
    fill_raise_counts(tree, 0, 0, &mut counts);
    counts
}

fn fill_raise_counts(tree: &PreflopTree, node_idx: u32, count: u8, out: &mut [u8]) {
    out[node_idx as usize] = count;
    if let PreflopNode::Decision {
        children,
        action_labels,
        ..
    } = &tree.nodes[node_idx as usize]
    {
        for (&child, action) in children.iter().zip(action_labels) {
            let child_count = match action {
                PreflopAction::Raise(_) | PreflopAction::AllIn => count + 1,
                _ => count,
            };
            fill_raise_counts(tree, child, child_count, out);
        }
    }
}

/// Precompute per-player investments at every node in the tree.
///
/// Walks the tree from the root, tracking how much each player has committed.
/// Returns a `Vec<[u32; 2]>` indexed by node index.
fn precompute_investments(tree: &PreflopTree, config: &PreflopConfig) -> NodeInvestments {
    let mut investments = vec![[0u32; 2]; tree.nodes.len()];
    let mut root_inv = [0u32; 2];
    for &(pos, amount) in &config.blinds {
        if pos < 2 {
            root_inv[pos] += amount.min(config.stacks[pos]);
        }
    }
    for &(pos, amount) in &config.antes {
        if pos < 2 {
            root_inv[pos] += amount.min(config.stacks[pos]);
        }
    }
    investments[0] = root_inv;
    fill_investments(tree, 0, root_inv, &mut investments);
    investments
}

/// Recursively fill investment data for all children of a decision node.
fn fill_investments(tree: &PreflopTree, node_idx: u32, inv: [u32; 2], out: &mut NodeInvestments) {
    let node = &tree.nodes[node_idx as usize];
    let (position, children, action_labels) = match node {
        PreflopNode::Decision {
            position,
            children,
            action_labels,
        } => (*position, children.as_slice(), action_labels.as_slice()),
        PreflopNode::Terminal { .. } => return,
    };

    for (&child_idx, action) in children.iter().zip(action_labels) {
        let child_inv = compute_child_investment(tree, inv, position, action, child_idx);
        out[child_idx as usize] = child_inv;
        fill_investments(tree, child_idx, child_inv, out);
    }
}

/// Compute the per-player investments at a child node given the parent's
/// investments, the acting position, and the action taken.
fn compute_child_investment(
    tree: &PreflopTree,
    inv: [u32; 2],
    position: u8,
    action: &PreflopAction,
    child_idx: u32,
) -> [u32; 2] {
    let p = position as usize;
    let mut child_inv = inv;

    match action {
        PreflopAction::Fold => {
            // No investment change
        }
        PreflopAction::Call => {
            // Caller matches the highest investment
            let target = inv[0].max(inv[1]);
            child_inv[p] = target;
        }
        PreflopAction::Raise(_) | PreflopAction::AllIn => {
            // Derive from the child terminal's pot (sum of both investments).
            // The non-acting player's investment hasn't changed, so:
            // `acting_player_inv = child_pot - other_player_inv`
            let other = 1 - p;
            let child_pot = first_terminal_pot(tree, child_idx);
            child_inv[p] = child_pot.saturating_sub(inv[other]);
        }
    }

    child_inv
}

/// Find the pot of the first reachable terminal via a fold path.
///
/// Starting from `node_idx`, follows fold edges (or call edges if no fold)
/// to reach a terminal and read its pot. This tells us the total chips
/// invested at that point in the tree, which equals the sum of all
/// player investments after the parent action was taken.
fn first_terminal_pot(tree: &PreflopTree, node_idx: u32) -> u32 {
    match &tree.nodes[node_idx as usize] {
        PreflopNode::Terminal { pot, .. } => *pot,
        PreflopNode::Decision {
            children,
            action_labels,
            ..
        } => {
            // A fold terminal gives us the pot with no further investment changes
            for (i, action) in action_labels.iter().enumerate() {
                if matches!(action, PreflopAction::Fold)
                    && let PreflopNode::Terminal { pot, .. } = &tree.nodes[children[i] as usize]
                {
                    return *pot;
                }
            }
            // Fall back to following a call path
            for (i, action) in action_labels.iter().enumerate() {
                if matches!(action, PreflopAction::Call) {
                    return first_terminal_pot(tree, children[i]);
                }
            }
            0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::{PreflopConfig, RaiseSize};
    use test_macros::timed_test;

    /// A minimal config with a tiny tree for fast unit tests.
    fn tiny_config() -> PreflopConfig {
        let mut config = PreflopConfig::heads_up(3);
        config.raise_sizes = vec![vec![RaiseSize::Bb(3.0)]];
        config.raise_cap = 1;
        config
    }

    #[timed_test]
    fn solver_creates_from_config() {
        let config = tiny_config();
        let solver = PreflopSolver::new(&config);
        assert_eq!(solver.iteration(), 0);
    }

    #[timed_test]
    fn solver_runs_one_iteration() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);
        solver.train(1);
        assert_eq!(solver.iteration(), 1);
    }

    #[timed_test]
    fn solver_strategy_is_valid_distribution() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);
        solver.train(2);
        let strategy = solver.strategy();
        for hand_idx in 0..169 {
            let probs = strategy.get_root_probs(hand_idx);
            if probs.is_empty() {
                continue;
            }
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "hand {hand_idx}: sum = {sum}");
        }
    }

    #[timed_test]
    fn strategy_has_entries_for_all_hands() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);
        solver.train(1);
        let strategy = solver.strategy();
        for hand_idx in 0..169 {
            let probs = strategy.get_root_probs(hand_idx);
            assert!(
                !probs.is_empty(),
                "hand {hand_idx} should have strategy at root"
            );
        }
    }

    #[timed_test]
    fn solver_stores_dcfr_params_from_config() {
        let config = tiny_config();
        let solver = PreflopSolver::new(&config);
        assert!((solver.dcfr_alpha - 1.5).abs() < f64::EPSILON);
        assert!((solver.dcfr_beta - 0.5).abs() < f64::EPSILON);
        assert!((solver.dcfr_gamma - 2.0).abs() < f64::EPSILON);
    }

    #[timed_test]
    fn discount_regrets_scales_positive_and_negative() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);
        // Find the root node's layout slot (position=0)
        let (base, _) = solver.layout.slot(0, 0);
        if solver.regret_sum.len() > base + 2 {
            solver.regret_sum[base] = 10.0; // positive
            solver.regret_sum[base + 1] = -5.0; // negative
            solver.regret_sum[base + 2] = 0.0; // zero
            solver.iteration = 5;
            // Discount hero_pos=0 (root node position is 0)
            solver.discount_regrets(0);

            let t = 6.0_f64; // iteration + 1
            let pf = t.powf(1.5) / (t.powf(1.5) + 1.0);
            let nf = t.powf(0.5) / (t.powf(0.5) + 1.0);
            assert!((solver.regret_sum[base] - 10.0 * pf).abs() < 1e-10);
            assert!((solver.regret_sum[base + 1] - (-5.0 * nf)).abs() < 1e-10);
            assert!((solver.regret_sum[base + 2]).abs() < 1e-10);
        }
    }

    #[timed_test]
    fn strategy_discount_formula() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);
        solver.iteration = 10;
        let sd = solver.strategy_discount();
        let expected = (10.0_f64 / 11.0).powf(2.0);
        assert!(
            (sd - expected).abs() < 1e-10,
            "sd={sd}, expected={expected}"
        );
    }

    #[timed_test]
    fn alternating_updates_traverse_different_players() {
        let config = tiny_config();
        let mut solver = PreflopSolver::new(&config);

        // Run iteration 1 (hero_pos = 1 since iteration=1, 1%2=1)
        solver.train(1);
        let regrets_after_1 = solver.regret_sum.clone();

        // Run iteration 2 (hero_pos = 0 since iteration=2, 2%2=0)
        solver.train(1);
        let regrets_after_2 = solver.regret_sum.clone();

        // Both iterations should produce non-trivial changes
        let changed_1: usize = regrets_after_1.iter().filter(|&&r| r.abs() > 1e-15).count();
        let changed_2: usize = regrets_after_2
            .iter()
            .zip(&regrets_after_1)
            .filter(|&(a, b)| (a - b).abs() > 1e-15)
            .count();
        assert!(changed_1 > 0, "iteration 1 should change some regrets");
        assert!(changed_2 > 0, "iteration 2 should change some regrets");
    }

    /// Vanilla CFR: no discounting is applied.
    #[timed_test]
    fn vanilla_no_discounting() {
        use super::super::config::CfrVariant;
        let mut config = tiny_config();
        config.cfr_variant = CfrVariant::Vanilla;
        let mut solver = PreflopSolver::new(&config);

        // Seed some regret values
        let (base, _) = solver.layout.slot(0, 0);
        if solver.regret_sum.len() > base + 2 {
            solver.regret_sum[base] = 10.0;
            solver.regret_sum[base + 1] = -5.0;
            solver.iteration = 5;

            // Run one full training iteration — discounting should be skipped
            let before_pos = solver.regret_sum[base];
            let before_neg = solver.regret_sum[base + 1];
            solver.apply_discounting();

            // Regrets should be unchanged (no multiplicative discounting)
            assert!(
                (solver.regret_sum[base] - before_pos).abs() < 1e-10,
                "positive regret should not be discounted in vanilla mode"
            );
            assert!(
                (solver.regret_sum[base + 1] - before_neg).abs() < 1e-10,
                "negative regret should not be discounted in vanilla mode"
            );
        }
    }

    /// Vanilla CFR produces valid strategies after training.
    #[timed_test]
    fn vanilla_produces_valid_strategy() {
        use super::super::config::CfrVariant;
        let mut config = tiny_config();
        config.cfr_variant = CfrVariant::Vanilla;
        let mut solver = PreflopSolver::new(&config);
        solver.train(10);
        let strategy = solver.strategy();
        for hand_idx in 0..169 {
            let probs = strategy.get_root_probs(hand_idx);
            if probs.is_empty() {
                continue;
            }
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "hand {hand_idx}: sum = {sum}");
        }
    }

    /// Vanilla CFR uses uniform weighting (weight=1), not LCFR linear weighting.
    #[timed_test]
    fn vanilla_uniform_weighting() {
        use super::super::config::CfrVariant;
        let mut config = tiny_config();
        config.cfr_variant = CfrVariant::Vanilla;
        let mut solver = PreflopSolver::new(&config);
        solver.train(5);

        // With uniform weighting, strategy_sum accumulates 1×reach×strategy per iteration.
        // With LCFR, it would be T×reach×strategy. After 5 iterations of vanilla,
        // strategy sums should be moderate, not scaled by iteration number.
        let (base, n) = solver.layout.slot(0, 0);
        let total: f64 = solver.strategy_sum[base..base + n].iter().sum();
        assert!(total > 0.0, "strategy sums should be non-zero after training");

        // Compare with DCFR: same config but DCFR should have larger sums
        // due to linear weighting (weight grows with iteration)
        let mut dcfr_config = tiny_config();
        dcfr_config.cfr_variant = CfrVariant::Dcfr;
        let mut dcfr_solver = PreflopSolver::new(&dcfr_config);
        dcfr_solver.train(5);
        let (dbase, dn) = dcfr_solver.layout.slot(0, 0);
        let dcfr_total: f64 = dcfr_solver.strategy_sum[dbase..dbase + dn].iter().sum();

        // DCFR with linear weighting should accumulate more than vanilla
        assert!(
            dcfr_total > total,
            "DCFR strategy sums ({dcfr_total}) should exceed vanilla ({total}) due to LCFR weighting"
        );
    }

    #[timed_test]
    fn investments_at_root_are_blinds() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        let inv = precompute_investments(&tree, &config);
        // SB posted 1, BB posted 2
        assert_eq!(inv[0], [1, 2]);
    }

    #[timed_test]
    fn investments_at_fold_terminal() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        let inv = precompute_investments(&tree, &config);
        // SB folds: investments unchanged from root
        if let PreflopNode::Decision {
            children,
            action_labels,
            ..
        } = &tree.nodes[0]
        {
            let fold_idx = action_labels
                .iter()
                .position(|a| matches!(a, PreflopAction::Fold));
            if let Some(fi) = fold_idx {
                let child = children[fi] as usize;
                assert_eq!(inv[child], [1, 2], "fold should not change investments");
            }
        }
    }

    #[timed_test]
    fn investments_after_limp() {
        let config = PreflopConfig::heads_up(100);
        let tree = PreflopTree::build(&config);
        let inv = precompute_investments(&tree, &config);
        // SB calls (limps): SB inv goes from 1 to 2
        if let PreflopNode::Decision {
            children,
            action_labels,
            ..
        } = &tree.nodes[0]
        {
            let call_idx = action_labels
                .iter()
                .position(|a| matches!(a, PreflopAction::Call));
            if let Some(ci) = call_idx {
                let child = children[ci] as usize;
                assert_eq!(inv[child], [2, 2], "after SB limps, both invested 2");
            }
        }
    }

    /// CFR+: negative regrets are floored to zero after each iteration.
    #[timed_test]
    fn cfrplus_floors_negative_regrets() {
        use super::super::config::CfrVariant;
        let mut config = tiny_config();
        config.cfr_variant = CfrVariant::CfrPlus;
        let mut solver = PreflopSolver::new(&config);

        // Seed negative regrets
        let (base, _) = solver.layout.slot(0, 0);
        if solver.regret_sum.len() > base + 2 {
            solver.regret_sum[base] = 10.0;
            solver.regret_sum[base + 1] = -5.0;
            solver.regret_sum[base + 2] = -100.0;
        }

        // Train one iteration — flooring happens at end
        solver.train(1);

        // All regrets must be >= 0
        for (i, &r) in solver.regret_sum.iter().enumerate() {
            assert!(r >= 0.0, "regret[{i}] = {r} should be >= 0 under CFR+");
        }
    }

    /// CFR+: produces valid probability distributions after training.
    #[timed_test]
    fn cfrplus_produces_valid_strategy() {
        use super::super::config::CfrVariant;
        let mut config = tiny_config();
        config.cfr_variant = CfrVariant::CfrPlus;
        let mut solver = PreflopSolver::new(&config);
        solver.train(10);
        let strategy = solver.strategy();
        for hand_idx in 0..169 {
            let probs = strategy.get_root_probs(hand_idx);
            if probs.is_empty() {
                continue;
            }
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 0.01, "hand {hand_idx}: sum = {sum}");
        }
    }

    /// CFR+: DCFR discounting is not applied.
    #[timed_test]
    fn cfrplus_no_dcfr_discounting() {
        use super::super::config::CfrVariant;
        let mut config = tiny_config();
        config.cfr_variant = CfrVariant::CfrPlus;
        let mut solver = PreflopSolver::new(&config);

        let (base, _) = solver.layout.slot(0, 0);
        if solver.regret_sum.len() > base + 1 {
            solver.regret_sum[base] = 10.0;
            solver.regret_sum[base + 1] = 3.0;
            solver.iteration = 5;

            let before_0 = solver.regret_sum[base];
            let before_1 = solver.regret_sum[base + 1];
            solver.apply_discounting();

            assert!(
                (solver.regret_sum[base] - before_0).abs() < 1e-10,
                "CFR+ should not apply DCFR discounting to regrets"
            );
            assert!(
                (solver.regret_sum[base + 1] - before_1).abs() < 1e-10,
                "CFR+ should not apply DCFR discounting to regrets"
            );
        }
    }

    /// Verify that `postflop_showdown_value` uses the correct position mapping:
    /// hero_pos=0 (SB/IP) should read pos-0 values, hero_pos=1 (BB/OOP) should
    /// read pos-1 values.
    #[timed_test]
    fn postflop_showdown_value_position_mapping() {
        use crate::preflop::postflop_abstraction::{PostflopAbstraction, PostflopValues};
        use crate::preflop::postflop_tree::{PostflopNode, PostflopTree, PotType};

        // Build a tiny abstraction with n=2 hands and asymmetric IP vs OOP EVs.
        // hand_avg_values layout: [pos0(IP) n*n entries] [pos1(OOP) n*n entries]
        let n = 2;
        let mut hand_avg_values = vec![0.0; 2 * n * n];
        // IP (pos 0): hand 0 vs hand 1 = +0.3 pot fraction
        hand_avg_values[0 * n * n + 0 * n + 1] = 0.3;
        // OOP (pos 1): hand 1 vs hand 0 = -0.1 pot fraction
        hand_avg_values[1 * n * n + 1 * n + 0] = -0.1;

        let abstraction = PostflopAbstraction {
            tree: PostflopTree {
                nodes: vec![PostflopNode::Terminal {
                    terminal_type: crate::preflop::postflop_tree::PostflopTerminalType::Showdown,
                    pot_fraction: 1.0,
                }],
                pot_type: PotType::Raised,
                spr: 5.0,
            },
            values: PostflopValues {
                values: vec![],
                num_buckets: n,
                num_flops: 0,
            },
            hand_avg_values,
            spr: 5.0,
            flops: vec![],
        };

        let pf_state = PostflopState {
            abstraction,
            raise_counts: vec![1], // raise_count=1 → PotType::Raised
        };

        let pot = 10;
        let hero_inv = 5.0;
        // Stacks deep enough that actual_spr >= model_spr (no interpolation).
        let stacks = [60, 60];
        let equity = 0.5; // unused when actual_spr >= model_spr

        // SB (hero_pos=0) looking up hand 0 vs opp hand 1 → should get IP value 0.3
        let ev_sb = postflop_showdown_value(&pf_state, 0, pot, hero_inv, 0, 1, 0, equity, stacks);
        // Expected: 0.3 * 10 + (10/2 - 5) = 3.0
        assert!(
            (ev_sb - 3.0).abs() < 1e-10,
            "SB should read IP (pos 0) values: got {ev_sb}, expected 3.0"
        );

        // BB (hero_pos=1) looking up hand 1 vs opp hand 0 → should get OOP value -0.1
        let ev_bb = postflop_showdown_value(&pf_state, 0, pot, hero_inv, 1, 0, 1, equity, stacks);
        // Expected: -0.1 * 10 + (10/2 - 5) = -1.0
        assert!(
            (ev_bb - (-1.0)).abs() < 1e-10,
            "BB should read OOP (pos 1) values: got {ev_bb}, expected -1.0"
        );
    }

    /// Limped-pot showdowns (raise_count=0) must fall back to raw equity,
    /// not the postflop model. This prevents the limp-trap degeneracy where
    /// inflated postflop model values make SB never raise.
    #[timed_test]
    fn postflop_showdown_value_limped_pot_uses_equity() {
        use crate::preflop::postflop_abstraction::{PostflopAbstraction, PostflopValues};
        use crate::preflop::postflop_tree::{PostflopNode, PostflopTree, PotType};

        let n = 2;
        let mut hand_avg_values = vec![0.0; 2 * n * n];
        // Set a large postflop model value to verify it's NOT used.
        hand_avg_values[0 * n * n + 0 * n + 1] = 2.7; // IP: hand 0 vs hand 1

        let abstraction = PostflopAbstraction {
            tree: PostflopTree {
                nodes: vec![PostflopNode::Terminal {
                    terminal_type: crate::preflop::postflop_tree::PostflopTerminalType::Showdown,
                    pot_fraction: 1.0,
                }],
                pot_type: PotType::Raised,
                spr: 5.0,
            },
            values: PostflopValues { values: vec![], num_buckets: n, num_flops: 0 },
            hand_avg_values,
            spr: 5.0,
            flops: vec![],
        };

        // Node 0 has raise_count=0 (limped pot); Node 1 has raise_count=1 (raised pot).
        let pf_state = PostflopState {
            abstraction,
            raise_counts: vec![0, 1],
        };

        let pot = 4;
        let hero_inv = 2.0;
        let stacks = [200, 200];
        let equity = 0.85;

        // Limped pot (node 0, raise_count=0): should use raw equity.
        let ev_limped = postflop_showdown_value(&pf_state, 0, pot, hero_inv, 0, 1, 0, equity, stacks);
        let expected_eq = equity * f64::from(pot) - hero_inv; // 0.85*4 - 2 = 1.4
        assert!(
            (ev_limped - expected_eq).abs() < 1e-10,
            "Limped pot should use raw equity: got {ev_limped}, expected {expected_eq}"
        );

        // Raised pot (node 1, raise_count=1): should use postflop model.
        let ev_raised = postflop_showdown_value(&pf_state, 1, pot, hero_inv, 0, 1, 0, equity, stacks);
        // Model value: 2.7 * 4 + (4/2 - 2) = 10.8
        assert!(
            (ev_raised - 10.8).abs() < 1e-10,
            "Raised pot should use postflop model: got {ev_raised}, expected 10.8"
        );

        // Key assertion: raised pot value >> limped pot value, so raising is incentivized.
        assert!(
            ev_raised > ev_limped * 3.0,
            "Raised pot value ({ev_raised}) should be much larger than limped ({ev_limped})"
        );
    }

    #[timed_test]
    fn select_closest_spr_picks_nearest() {
        let sprs = [2.0, 6.0, 20.0];
        assert_eq!(super::select_closest_spr(&sprs, 1.0), 0);
        assert_eq!(super::select_closest_spr(&sprs, 3.0), 0);
        assert_eq!(super::select_closest_spr(&sprs, 4.5), 1);
        assert_eq!(super::select_closest_spr(&sprs, 5.0), 1);
        assert_eq!(super::select_closest_spr(&sprs, 12.0), 1);
        assert_eq!(super::select_closest_spr(&sprs, 13.5), 2);
        assert_eq!(super::select_closest_spr(&sprs, 50.0), 2);
    }

    #[timed_test]
    fn select_closest_spr_single_element() {
        assert_eq!(super::select_closest_spr(&[3.5], 0.0), 0);
        assert_eq!(super::select_closest_spr(&[3.5], 100.0), 0);
    }
}
