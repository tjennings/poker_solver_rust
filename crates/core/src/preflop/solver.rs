//! Linear CFR solver for preflop game trees.
//!
//! Performs full-game CFR over 169 canonical hand matchups, using linear
//! weighting (regrets and strategy sums scaled by iteration number `t`).
//! Each iteration is parallelized via rayon: a frozen regret snapshot is
//! shared across threads, with flat buffer deltas merged via parallel reduce.
//!
//! Storage uses flat `Vec<f64>` buffers indexed by `(node, hand, action)`
//! for cache-friendly access and fast clone/merge operations.

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::PreflopConfig;
use super::equity::EquityTable;
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
        Self { entries, total_size: offset }
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
}

/// Preflop Linear CFR solver.
///
/// Traverses the preflop tree for every 169x169 canonical hand matchup,
/// accumulating regrets and strategy sums with linear weighting.
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
        }
    }

    /// Run `iterations` of Linear CFR training.
    pub fn train(&mut self, iterations: u64) {
        for _ in 0..iterations {
            self.iteration += 1;
            let t = self.iteration;
            self.train_one_iteration(t);
        }
    }

    /// Current iteration count.
    #[must_use]
    pub fn iteration(&self) -> u64 {
        self.iteration
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

    /// Single training iteration: snapshot regrets, parallel traverse, merge.
    fn train_one_iteration(&mut self, t: u64) {
        // Clone is a fast memcpy of a flat buffer (no HashMap allocation)
        let snapshot = self.regret_sum.clone();
        let buf_size = self.layout.total_size;

        let ctx = Ctx {
            tree: &self.tree,
            investments: &self.investments,
            equity: &self.equity,
            layout: &self.layout,
            snapshot: &snapshot,
        };

        // Parallel fold+reduce: each rayon task accumulates into its own
        // flat buffer, then buffers are merged in a parallel reduction tree.
        let pairs = &self.pairs;
        let (merged_regret, merged_strategy) = pairs
            .par_iter()
            .fold(
                || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
                |(mut dr, mut ds), &(h1, h2)| {
                    let w = ctx.equity.weight(h1 as usize, h2 as usize);
                    cfr_traverse(&ctx, &mut dr, &mut ds, 0, h1, h2, 0, 1.0, w, t);
                    cfr_traverse(&ctx, &mut dr, &mut ds, 0, h2, h1, 1, 1.0, w, t);
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
            );

        add_into(&mut self.regret_sum, &merged_regret);
        add_into(&mut self.strategy_sum, &merged_strategy);
    }
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
#[allow(clippy::too_many_arguments)]
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
    t: u64,
) -> f64 {
    let inv = ctx.investments[node_idx as usize];
    let hero_inv = f64::from(inv[hero_pos as usize]);

    match &ctx.tree.nodes[node_idx as usize] {
        PreflopNode::Terminal { terminal_type, pot } => {
            terminal_value(*terminal_type, *pot, hero_inv, hero_hand, opp_hand, hero_pos, ctx.equity)
        }
        PreflopNode::Decision { position, children, .. } => {
            let num_actions = children.len();
            let is_hero = *position == hero_pos;
            let hand_for_key = if is_hero { hero_hand } else { opp_hand };
            let (start, _) = ctx.layout.slot(node_idx, hand_for_key);
            let mut strategy = [0.0f64; MAX_ACTIONS];
            regret_matching_into(ctx.snapshot, start, &mut strategy[..num_actions]);

            if is_hero {
                traverse_hero(
                    ctx, dr, ds,
                    start, hero_hand, opp_hand, hero_pos, reach_hero, reach_opp,
                    t, children, &strategy[..num_actions],
                )
            } else {
                traverse_opponent(
                    ctx, dr, ds,
                    hero_hand, opp_hand, hero_pos, reach_hero, reach_opp,
                    t, children, &strategy[..num_actions],
                )
            }
        }
    }
}

/// Hero's decision: compute regrets and update strategy/regret deltas.
#[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
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
    t: u64,
    children: &[u32],
    strategy: &[f64],
) -> f64 {
    let num_actions = children.len();
    let mut action_values = [0.0f64; MAX_ACTIONS];
    for (i, &child_idx) in children.iter().enumerate() {
        action_values[i] = cfr_traverse(
            ctx, dr, ds,
            child_idx, hero_hand, opp_hand, hero_pos,
            reach_hero * strategy[i], reach_opp, t,
        );
    }

    let node_value: f64 = strategy.iter()
        .zip(&action_values[..num_actions])
        .map(|(s, v)| s * v)
        .sum();
    let t_f64 = t as f64;

    // Accumulate regret deltas (weighted by t for linear CFR)
    for (i, val) in action_values[..num_actions].iter().enumerate() {
        dr[slot_start + i] += t_f64 * reach_opp * (val - node_value);
    }

    // Accumulate strategy deltas (weighted by t * reach_hero)
    for (i, &s) in strategy.iter().enumerate() {
        ds[slot_start + i] += t_f64 * reach_hero * s;
    }

    node_value
}

/// Opponent's decision: traverse using opponent's strategy.
#[allow(clippy::too_many_arguments)]
fn traverse_opponent(
    ctx: &Ctx<'_>,
    dr: &mut [f64],
    ds: &mut [f64],
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    reach_hero: f64,
    reach_opp: f64,
    t: u64,
    children: &[u32],
    strategy: &[f64],
) -> f64 {
    let mut node_value = 0.0f64;
    for (i, &child_idx) in children.iter().enumerate() {
        let child_value = cfr_traverse(
            ctx, dr, ds,
            child_idx, hero_hand, opp_hand, hero_pos,
            reach_hero, reach_opp * strategy[i], t,
        );
        node_value += strategy[i] * child_value;
    }
    node_value
}

/// Compute hero's utility at a terminal node.
fn terminal_value(
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
fn fill_investments(
    tree: &PreflopTree,
    node_idx: u32,
    inv: [u32; 2],
    out: &mut NodeInvestments,
) {
    let node = &tree.nodes[node_idx as usize];
    let (position, children, action_labels) = match node {
        PreflopNode::Decision { position, children, action_labels } => {
            (*position, children.as_slice(), action_labels.as_slice())
        }
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
        PreflopNode::Decision { children, action_labels, .. } => {
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
    use crate::preflop::PreflopConfig;
    use test_macros::timed_test;

    /// A minimal config with a tiny tree for fast unit tests.
    fn tiny_config() -> PreflopConfig {
        let mut config = PreflopConfig::heads_up(3);
        config.raise_sizes = vec![vec![3.0]];
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
            assert!(
                (sum - 1.0).abs() < 0.01,
                "hand {hand_idx}: sum = {sum}"
            );
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
        if let PreflopNode::Decision { children, action_labels, .. } = &tree.nodes[0] {
            let fold_idx = action_labels.iter().position(|a| matches!(a, PreflopAction::Fold));
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
        if let PreflopNode::Decision { children, action_labels, .. } = &tree.nodes[0] {
            let call_idx = action_labels.iter().position(|a| matches!(a, PreflopAction::Call));
            if let Some(ci) = call_idx {
                let child = children[ci] as usize;
                assert_eq!(inv[child], [2, 2], "after SB limps, both invested 2");
            }
        }
    }
}
