//! Linear CFR solver for preflop game trees.
//!
//! Performs full-game CFR over 169 canonical hand matchups, using linear
//! weighting (regrets and strategy sums scaled by iteration number `t`).

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::PreflopConfig;
use super::equity::EquityTable;
use super::tree::{PreflopAction, PreflopNode, PreflopTree, TerminalType};

const NUM_HANDS: usize = 169;

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

/// Preflop Linear CFR solver.
///
/// Traverses the preflop tree for every 169x169 canonical hand matchup,
/// accumulating regrets and strategy sums with linear weighting.
pub struct PreflopSolver {
    tree: PreflopTree,
    equity: EquityTable,
    /// Per-node per-player investments, precomputed from the tree.
    investments: NodeInvestments,
    /// Cumulative regrets per action, keyed by `(node_idx, hand_idx)`.
    regret_sum: FxHashMap<u64, Vec<f64>>,
    /// Cumulative weighted strategy per action, keyed by `(node_idx, hand_idx)`.
    strategy_sum: FxHashMap<u64, Vec<f64>>,
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
    pub fn new_with_equity(config: &PreflopConfig, equity: EquityTable) -> Self {
        let tree = PreflopTree::build(config);
        let investments = precompute_investments(&tree, config);
        Self {
            tree,
            equity,
            investments,
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
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
    pub fn strategy(&self) -> PreflopStrategy {
        let mut strategies = FxHashMap::default();
        for (&key, sums) in &self.strategy_sum {
            strategies.insert(key, normalize(sums));
        }
        PreflopStrategy { strategies }
    }

    /// Single training iteration: traverse all hand matchups for both players.
    #[allow(clippy::cast_possible_truncation)]
    fn train_one_iteration(&mut self, t: u64) {
        for h1 in 0..NUM_HANDS {
            for h2 in 0..NUM_HANDS {
                if self.equity.weight(h1, h2) <= 0.0 {
                    continue;
                }
                // Safe: h1, h2 always < 169, well within u16
                self.cfr_traverse(0, h1 as u16, h2 as u16, 0, 1.0, 1.0, t);
                self.cfr_traverse(0, h2 as u16, h1 as u16, 1, 1.0, 1.0, t);
            }
        }
    }

    /// Recursive CFR traversal. Returns expected value for the hero player.
    #[allow(clippy::too_many_arguments)]
    fn cfr_traverse(
        &mut self,
        node_idx: u32,
        hero_hand: u16,
        opp_hand: u16,
        hero_pos: u8,
        reach_hero: f64,
        reach_opp: f64,
        t: u64,
    ) -> f64 {
        let inv = self.investments[node_idx as usize];
        let hero_inv = f64::from(inv[hero_pos as usize]);

        match self.tree.nodes[node_idx as usize].clone() {
            PreflopNode::Terminal { terminal_type, pot } => {
                terminal_value(terminal_type, pot, hero_inv, hero_hand, opp_hand, hero_pos, &self.equity)
            }
            PreflopNode::Decision { position, children, .. } => {
                let num_actions = children.len();
                let is_hero = position == hero_pos;
                let hand_for_key = if is_hero { hero_hand } else { opp_hand };
                let key = PreflopStrategy::key(node_idx, hand_for_key);
                let strategy = regret_matching(&self.regret_sum, key, num_actions);

                if is_hero {
                    self.traverse_hero(
                        key, hero_hand, opp_hand, hero_pos, reach_hero, reach_opp,
                        t, &children, &strategy, num_actions,
                    )
                } else {
                    self.traverse_opponent(
                        hero_hand, opp_hand, hero_pos, reach_hero, reach_opp,
                        t, &children, &strategy,
                    )
                }
            }
        }
    }

    /// Hero's decision: compute regrets and update strategy/regret sums.
    #[allow(clippy::too_many_arguments, clippy::cast_precision_loss)]
    fn traverse_hero(
        &mut self,
        key: u64,
        hero_hand: u16,
        opp_hand: u16,
        hero_pos: u8,
        reach_hero: f64,
        reach_opp: f64,
        t: u64,
        children: &[u32],
        strategy: &[f64],
        num_actions: usize,
    ) -> f64 {
        let mut action_values = vec![0.0f64; num_actions];
        for (i, &child_idx) in children.iter().enumerate() {
            action_values[i] = self.cfr_traverse(
                child_idx, hero_hand, opp_hand, hero_pos,
                reach_hero * strategy[i], reach_opp, t,
            );
        }

        let node_value: f64 = strategy.iter().zip(&action_values).map(|(s, v)| s * v).sum();
        let t_f64 = t as f64;

        // Update regrets (weighted by t for linear CFR)
        let regrets = self.regret_sum.entry(key).or_insert_with(|| vec![0.0; num_actions]);
        for (i, val) in action_values.iter().enumerate() {
            regrets[i] += t_f64 * reach_opp * (val - node_value);
        }

        // Update strategy sum (weighted by t * reach_hero)
        let strat_sum = self.strategy_sum.entry(key).or_insert_with(|| vec![0.0; num_actions]);
        for (i, &s) in strategy.iter().enumerate() {
            strat_sum[i] += t_f64 * reach_hero * s;
        }

        node_value
    }

    /// Opponent's decision: traverse using opponent's strategy.
    #[allow(clippy::too_many_arguments)]
    fn traverse_opponent(
        &mut self,
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
            let child_value = self.cfr_traverse(
                child_idx, hero_hand, opp_hand, hero_pos,
                reach_hero, reach_opp * strategy[i], t,
            );
            node_value += strategy[i] * child_value;
        }
        node_value
    }
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

/// Regret matching: normalize positive regrets into a probability distribution.
#[allow(clippy::cast_precision_loss)]
fn regret_matching(regret_sum: &FxHashMap<u64, Vec<f64>>, key: u64, num_actions: usize) -> Vec<f64> {
    let mut strategy = vec![0.0f64; num_actions];
    let mut positive_sum = 0.0f64;

    if let Some(r) = regret_sum.get(&key) {
        for (i, &val) in r.iter().enumerate() {
            if val > 0.0 {
                strategy[i] = val;
                positive_sum += val;
            }
        }
    }

    if positive_sum > 0.0 {
        for s in &mut strategy {
            *s /= positive_sum;
        }
    } else {
        // Safe: num_actions is small (< 10)
        let uniform = 1.0 / num_actions as f64;
        strategy.fill(uniform);
    }

    strategy
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
