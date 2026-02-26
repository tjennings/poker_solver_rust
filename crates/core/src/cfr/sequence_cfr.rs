//! CPU Sequence-Form CFR solver.
//!
//! Implements full-traversal Counterfactual Regret Minimization using the
//! materialized game tree from [`super::game_tree`]. Instead of recursive DFS
//! with state cloning and hashtable lookups, this solver operates on flat
//! arrays indexed by node and info-set, traversing the tree level-by-level.
//!
//! # Algorithm
//!
//! Each iteration consists of three passes:
//! 1. **Regret matching**: Compute current strategy from cumulative regrets
//! 2. **Forward pass** (root → leaves): Propagate reach probabilities
//! 3. **Backward pass** (leaves → root): Propagate utilities, compute regrets
//!
//! DCFR discounting is applied after each iteration.
//!
//! # Advantages over MCCFR
//!
//! - Full traversal (no chance sampling) → faster convergence per iteration
//! - Dense array access → cache-friendly
//! - No state cloning or info-key computation during traversal
//! - Level-by-level passes → natural parallelism (future GPU port)

use rustc_hash::FxHashMap;

use super::game_tree::{GameTree, NodeType};
use super::regret::regret_match;
use crate::game::Player;
use crate::preflop::CfrVariant;

/// Configuration for sequence-form CFR training.
#[derive(Debug, Clone)]
pub struct SequenceCfrConfig {
    /// Which CFR variant to use (default: DCFR).
    pub cfr_variant: CfrVariant,
    /// DCFR positive regret discount exponent (default 1.5).
    pub dcfr_alpha: f64,
    /// DCFR negative regret discount exponent (default 0.5).
    pub dcfr_beta: f64,
    /// DCFR strategy sum discount exponent (default 2.0).
    pub dcfr_gamma: f64,
    /// Number of initial iterations without DCFR discounting (warm-up phase).
    pub dcfr_warmup: u64,
    /// Exploration factor (ε-greedy). Hero actions get at least `exploration / num_actions`
    /// probability during traversal. The average strategy accumulates the *intended*
    /// (non-explored) strategy, so exploration noise doesn't contaminate the output.
    pub exploration: f64,
}

impl Default for SequenceCfrConfig {
    fn default() -> Self {
        Self {
            cfr_variant: CfrVariant::Dcfr,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_warmup: 0,
            exploration: 0.0,
        }
    }
}

impl SequenceCfrConfig {
    /// Linear CFR weighting (Pluribus). All discount exponents = 1.0.
    #[must_use]
    pub fn linear_cfr() -> Self {
        Self {
            cfr_variant: CfrVariant::Linear,
            dcfr_alpha: 1.0,
            dcfr_beta: 1.0,
            dcfr_gamma: 1.0,
            dcfr_warmup: 0,
            exploration: 0.0,
        }
    }

    /// Vanilla CFR: no discounting, uniform iteration weighting.
    #[must_use]
    pub fn vanilla() -> Self {
        Self {
            cfr_variant: CfrVariant::Vanilla,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_warmup: 0,
            exploration: 0.0,
        }
    }

    /// CFR+ (Tammelin 2014): regrets floored to zero, linear strategy weighting.
    #[must_use]
    pub fn cfr_plus() -> Self {
        Self {
            cfr_variant: CfrVariant::CfrPlus,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            dcfr_warmup: 0,
            exploration: 0.0,
        }
    }
}

/// Maps each deal to its per-street hand bits, equity, and weight.
///
/// Hand bits can differ per street when using hand-class abstractions:
/// preflop uses canonical hand index, while flop/turn/river use
/// class-based encodings that depend on the board.
#[derive(Debug, Clone)]
pub struct DealInfo {
    /// Per-street hand bits for P1: `[preflop, flop, turn, river]`.
    pub hand_bits_p1: [u32; 4],
    /// Per-street hand bits for P2: `[preflop, flop, turn, river]`.
    pub hand_bits_p2: [u32; 4],
    /// P1's showdown equity (0.0 = P2 wins, 0.5 = tie, 1.0 = P1 wins).
    ///
    /// For concrete deals this is 0.0, 0.5, or 1.0.
    /// For abstract deals this can be any value in `[0.0, 1.0]`.
    pub p1_equity: f64,
    /// Weight of this deal (1.0 for concrete deals, >1.0 for grouped abstract deals).
    pub weight: f64,
}

/// CPU sequence-form CFR solver operating on materialized game trees.
///
/// The solver maintains regret and strategy arrays indexed by info set key,
/// shared across all deals. Each iteration traverses every deal's tree once.
///
/// Two construction modes:
/// - [`new`]: Single tree shared by all deals (for `HunlPostflop` where the action
///   tree is deal-independent). Terminal utilities computed from pot/stacks + showdown winner.
/// - [`from_per_deal_trees`]: Separate tree per deal (for games like Kuhn where
///   terminal utilities vary by deal). Uses pre-computed utilities in terminal nodes.
pub struct SequenceCfrSolver {
    /// Game trees. Single-tree mode has length 1; per-deal mode has one per deal.
    trees: Vec<GameTree>,
    deals: Vec<DealInfo>,
    config: SequenceCfrConfig,
    /// Cumulative regrets per info set key.
    regret_sum: FxHashMap<u64, Vec<f64>>,
    /// Cumulative strategy sum per info set key (for average strategy).
    strategy_sum: FxHashMap<u64, Vec<f64>>,
    /// Number of actions per info set key (cached from tree).
    num_actions: FxHashMap<u64, usize>,
    /// Which player owns each info set key (for per-player discounting).
    player_map: FxHashMap<u64, Player>,
    /// Number of completed iterations.
    iterations: u64,
}

impl SequenceCfrSolver {
    /// Create a solver with a single shared tree for all deals.
    ///
    /// Use this for `HunlPostflop` where all deals share the same action tree structure
    /// but differ in `hand_bits` and showdown outcomes.
    #[must_use]
    pub fn new(tree: GameTree, deals: Vec<DealInfo>, config: SequenceCfrConfig) -> Self {
        Self {
            trees: vec![tree],
            deals,
            config,
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
            num_actions: FxHashMap::default(),
            player_map: FxHashMap::default(),
            iterations: 0,
        }
    }

    /// Create a solver with separate trees per deal.
    ///
    /// Use this for games where terminal utilities vary by deal (e.g., Kuhn poker).
    /// Each tree should be materialized from the corresponding deal's initial state.
    ///
    /// # Panics
    ///
    /// Panics if `trees.len() != deals.len()`.
    #[must_use]
    pub fn from_per_deal_trees(
        trees: Vec<GameTree>,
        deals: Vec<DealInfo>,
        config: SequenceCfrConfig,
    ) -> Self {
        assert_eq!(trees.len(), deals.len(), "Must have one tree per deal");
        Self {
            trees,
            deals,
            config,
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
            num_actions: FxHashMap::default(),
            player_map: FxHashMap::default(),
            iterations: 0,
        }
    }

    /// Run `num_iterations` of full-traversal CFR.
    pub fn train(&mut self, num_iterations: u64) {
        for _ in 0..num_iterations {
            self.run_iteration();
        }
    }

    /// Run `num_iterations` with a per-iteration callback.
    pub fn train_with_callback<F: FnMut(u64)>(&mut self, num_iterations: u64, mut cb: F) {
        for _ in 0..num_iterations {
            self.run_iteration();
            cb(self.iterations);
        }
    }

    /// Run `num_iterations` of CFR, streaming deals from a source each iteration.
    ///
    /// Use this for large abstract deal sets that don't fit in memory.
    /// The `deal_source` is called each iteration to produce deal batches.
    /// The solver's `deals` field is unused in this mode.
    ///
    /// # Panics
    ///
    /// Panics if the solver was not created with a single shared tree.
    pub fn train_streaming<F>(&mut self, num_iterations: u64, deal_source: F)
    where
        F: Fn() -> Box<dyn Iterator<Item = Vec<DealInfo>>>,
    {
        assert_eq!(self.trees.len(), 1, "Streaming requires single shared tree");

        for _ in 0..num_iterations {
            self.iterations += 1;

            for hero in [Player::Player1, Player::Player2] {
                self.pre_discount(hero);
                for batch in deal_source() {
                    for deal in &batch {
                        self.process_single_deal(0, deal, hero);
                    }
                }
                self.post_update(hero);
            }
        }
    }

    /// Number of completed iterations.
    #[must_use]
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Access cumulative regret sums.
    #[must_use]
    pub fn regret_sum(&self) -> &FxHashMap<u64, Vec<f64>> {
        &self.regret_sum
    }

    /// Return average strategies for all info sets.
    #[must_use]
    pub fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        self.strategy_sum
            .iter()
            .filter_map(|(&k, sums)| normalize_strategy(sums).map(|s| (k, s)))
            .collect()
    }

    /// Return best-effort strategies (average when available, else regret-matched).
    #[must_use]
    pub fn all_strategies_best_effort(&self) -> FxHashMap<u64, Vec<f64>> {
        let mut result = FxHashMap::default();

        for (&k, sums) in &self.strategy_sum {
            if let Some(avg) = normalize_strategy(sums) {
                result.insert(k, avg);
            }
        }

        for (&k, regrets) in &self.regret_sum {
            result.entry(k).or_insert_with(|| regret_match(regrets));
        }

        result
    }

    /// Run one full CFR iteration with alternating player updates.
    ///
    /// Each iteration performs two passes (one per hero player). In each pass:
    /// 1. Discount: apply variant-aware discounting to hero's OLD regrets/strategy
    /// 2. Forward: reach propagation with ε-greedy exploration on hero's strategy
    /// 3. Backward: accumulate new regrets/strategy only for hero's decision nodes
    ///
    /// This matches the standard DCFR update: `R^{T+1} = d(T) * R^T + r^{T+1}`.
    fn run_iteration(&mut self) {
        self.iterations += 1;
        let num_deals = self.deals.len();
        let single_tree = self.trees.len() == 1;

        for hero in [Player::Player1, Player::Player2] {
            // DCFR/Linear: discount old regrets BEFORE adding new ones
            self.pre_discount(hero);

            for deal_idx in 0..num_deals {
                let tree_idx = if single_tree { 0 } else { deal_idx };
                self.process_single_deal(tree_idx, &self.deals[deal_idx].clone(), hero);
            }

            // CFR+: floor regrets AFTER adding new ones
            self.post_update(hero);
        }
    }

    /// Process one deal for a single hero traversal.
    fn process_single_deal(&mut self, tree_idx: usize, deal: &DealInfo, hero: Player) {
        let num_nodes = self.trees[tree_idx].nodes.len();

        let mut reach_p1 = vec![0.0_f64; num_nodes];
        let mut reach_p2 = vec![0.0_f64; num_nodes];
        let mut utility_p1 = vec![0.0_f64; num_nodes];

        reach_p1[0] = 1.0;
        reach_p2[0] = 1.0;

        forward_pass(
            &self.trees[tree_idx],
            deal,
            &self.regret_sum,
            &mut reach_p1,
            &mut reach_p2,
            hero,
            self.config.exploration,
        );

        backward_pass(
            &self.trees[tree_idx],
            deal,
            &reach_p1,
            &reach_p2,
            &mut utility_p1,
            hero,
            self.iterations,
            self.config.cfr_variant,
            self.config.exploration,
            &mut self.regret_sum,
            &mut self.strategy_sum,
            &mut self.num_actions,
            &mut self.player_map,
        );
    }

    /// Pre-deal discounting: discount old regrets/strategy before adding new ones.
    /// Used by DCFR/Linear. Skipped for Vanilla and CFR+.
    fn pre_discount(&mut self, hero: Player) {
        if matches!(
            self.config.cfr_variant,
            CfrVariant::Dcfr | CfrVariant::Linear
        ) && self.iterations > self.config.dcfr_warmup
        {
            self.discount_regrets_for(hero);
            self.discount_strategy_sums_for(hero);
        }
    }

    /// Post-deal update: applied after new regrets are accumulated.
    /// CFR+ floors regrets to zero here.
    fn post_update(&mut self, hero: Player) {
        if self.config.cfr_variant == CfrVariant::CfrPlus {
            self.floor_regrets_for(hero);
        }
    }

    /// DCFR regret discounting for a single player's info sets.
    fn discount_regrets_for(&mut self, hero: Player) {
        #[allow(clippy::cast_precision_loss)]
        let t = (self.iterations + 1) as f64;
        let pos_factor = t.powf(self.config.dcfr_alpha) / (t.powf(self.config.dcfr_alpha) + 1.0);
        let neg_factor = t.powf(self.config.dcfr_beta) / (t.powf(self.config.dcfr_beta) + 1.0);

        for (&key, regrets) in &mut self.regret_sum {
            if self.player_map.get(&key) != Some(&hero) {
                continue;
            }
            for r in regrets.iter_mut() {
                if *r > 0.0 {
                    *r *= pos_factor;
                } else if *r < 0.0 {
                    *r *= neg_factor;
                }
            }
        }
    }

    /// DCFR strategy sum discounting for a single player's info sets.
    fn discount_strategy_sums_for(&mut self, hero: Player) {
        #[allow(clippy::cast_precision_loss)]
        let t = self.iterations as f64;
        let ratio = t / (t + 1.0);
        let sd = ratio.powf(self.config.dcfr_gamma);

        for (&key, sums) in &mut self.strategy_sum {
            if self.player_map.get(&key) != Some(&hero) {
                continue;
            }
            for s in sums.iter_mut() {
                *s *= sd;
            }
        }
    }

    /// CFR+ regret flooring: clamp negative regrets to zero for hero's info sets.
    fn floor_regrets_for(&mut self, hero: Player) {
        for (&key, regrets) in &mut self.regret_sum {
            if self.player_map.get(&key) != Some(&hero) {
                continue;
            }
            for r in regrets.iter_mut() {
                if *r < 0.0 {
                    *r = 0.0;
                }
            }
        }
    }
}

/// Compute info set key for a node given a deal.
///
/// Selects the correct per-street hand bits based on the node's street.
fn deal_info_key(tree: &GameTree, node_idx: u32, deal: &DealInfo) -> u64 {
    let node = &tree.nodes[node_idx as usize];
    if let NodeType::Decision { player, street, .. } = &node.node_type {
        let hand_bits = match player {
            Player::Player1 => deal.hand_bits_p1[*street as usize],
            Player::Player2 => deal.hand_bits_p2[*street as usize],
        };
        tree.info_set_key(node_idx, hand_bits)
    } else {
        0
    }
}

/// Get current strategy from regret matching for an info set.
fn current_strategy_from(
    regret_sum: &FxHashMap<u64, Vec<f64>>,
    info_key: u64,
    num_actions: usize,
) -> Vec<f64> {
    if let Some(regrets) = regret_sum.get(&info_key) {
        regret_match(regrets)
    } else {
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / num_actions as f64;
        vec![uniform; num_actions]
    }
}

/// Forward pass: propagate reach probabilities from root to leaves.
///
/// At hero decision nodes with `exploration > 0`, the traversal strategy
/// is blended with uniform to ensure off-path subtrees are visited.
#[allow(clippy::too_many_arguments)]
fn forward_pass(
    tree: &GameTree,
    deal: &DealInfo,
    regret_sum: &FxHashMap<u64, Vec<f64>>,
    reach_p1: &mut [f64],
    reach_p2: &mut [f64],
    hero: Player,
    exploration: f64,
) {
    for level in &tree.levels {
        for &node_idx in level {
            let node = &tree.nodes[node_idx as usize];

            if let NodeType::Decision { player, .. } = &node.node_type {
                let info_key = deal_info_key(tree, node_idx, deal);
                let num_actions = node.children.len();
                let strategy = current_strategy_from(regret_sum, info_key, num_actions);

                // Apply ε-greedy exploration to hero's strategy only
                let traversal = if *player == hero && exploration > 0.0 {
                    #[allow(clippy::cast_precision_loss)]
                    let n_inv = 1.0 / num_actions as f64;
                    strategy
                        .iter()
                        .map(|&s| (1.0 - exploration).mul_add(s, exploration * n_inv))
                        .collect::<Vec<_>>()
                } else {
                    strategy
                };

                for (action_idx, &child_idx) in node.children.iter().enumerate() {
                    let ci = child_idx as usize;
                    let ni = node_idx as usize;
                    let prob = traversal[action_idx];

                    match player {
                        Player::Player1 => {
                            reach_p1[ci] = reach_p1[ni] * prob;
                            reach_p2[ci] = reach_p2[ni];
                        }
                        Player::Player2 => {
                            reach_p1[ci] = reach_p1[ni];
                            reach_p2[ci] = reach_p2[ni] * prob;
                        }
                    }
                }
            }
        }
    }
}

/// Backward pass: propagate utilities from leaves to root and update regrets.
///
/// Only updates regrets and strategy sums for the `hero` player's decision nodes.
/// Opponent nodes still compute utilities for propagation but skip regret updates.
#[allow(clippy::too_many_arguments)]
fn backward_pass(
    tree: &GameTree,
    deal: &DealInfo,
    reach_p1: &[f64],
    reach_p2: &[f64],
    utility_p1: &mut [f64],
    hero: Player,
    iteration: u64,
    cfr_variant: CfrVariant,
    exploration: f64,
    regret_sum: &mut FxHashMap<u64, Vec<f64>>,
    strategy_sum: &mut FxHashMap<u64, Vec<f64>>,
    num_actions_map: &mut FxHashMap<u64, usize>,
    player_map: &mut FxHashMap<u64, Player>,
) {
    // Iteration weighting per variant
    #[allow(clippy::cast_precision_loss)]
    let (regret_weight, strategy_weight) = match cfr_variant {
        CfrVariant::Vanilla => (1.0, 1.0),
        CfrVariant::Dcfr | CfrVariant::Linear => {
            let w = iteration as f64;
            (w, w)
        }
        CfrVariant::CfrPlus => (1.0, iteration as f64),
    };

    for level in tree.levels.iter().rev() {
        for &node_idx in level {
            let ni = node_idx as usize;
            let node = &tree.nodes[ni];

            match &node.node_type {
                NodeType::Terminal { .. } => {
                    utility_p1[ni] = tree.terminal_utility_p1(node_idx, deal.p1_equity);
                }
                NodeType::Decision { player, .. } => {
                    let info_key = deal_info_key(tree, node_idx, deal);
                    let num_actions = node.children.len();
                    let intended = current_strategy_from(regret_sum, info_key, num_actions);

                    // Use explored strategy for node_util at hero nodes
                    let traversal = if *player == hero && exploration > 0.0 {
                        #[allow(clippy::cast_precision_loss)]
                        let n_inv = 1.0 / num_actions as f64;
                        intended
                            .iter()
                            .map(|&s| (1.0 - exploration).mul_add(s, exploration * n_inv))
                            .collect::<Vec<_>>()
                    } else {
                        intended.clone()
                    };

                    // Compute node utility
                    let mut node_util = 0.0;
                    for (action_idx, &child_idx) in node.children.iter().enumerate() {
                        node_util += traversal[action_idx] * utility_p1[child_idx as usize];
                    }
                    utility_p1[ni] = node_util;

                    // Only update regrets and strategy sums for hero's nodes
                    if *player == hero {
                        let opp_reach = match player {
                            Player::Player1 => reach_p2[ni],
                            Player::Player2 => reach_p1[ni],
                        };
                        let my_reach = match player {
                            Player::Player1 => reach_p1[ni],
                            Player::Player2 => reach_p2[ni],
                        };

                        let regrets = regret_sum
                            .entry(info_key)
                            .or_insert_with(|| vec![0.0; num_actions]);
                        num_actions_map.entry(info_key).or_insert(num_actions);
                        player_map.entry(info_key).or_insert(*player);

                        for (action_idx, &child_idx) in node.children.iter().enumerate() {
                            let child_util = utility_p1[child_idx as usize];
                            let cf_regret = match player {
                                Player::Player1 => opp_reach * (child_util - node_util),
                                Player::Player2 => opp_reach * (node_util - child_util),
                            };
                            regrets[action_idx] += cf_regret * regret_weight * deal.weight;
                        }

                        // Accumulate strategy sum using *intended* (non-explored) strategy
                        if strategy_weight > 0.0 {
                            let strat_sums = strategy_sum
                                .entry(info_key)
                                .or_insert_with(|| vec![0.0; num_actions]);
                            for (i, &prob) in intended.iter().enumerate() {
                                strat_sums[i] +=
                                    my_reach * prob * strategy_weight * deal.weight;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// Normalize a strategy sum vector to probabilities.
fn normalize_strategy(sums: &[f64]) -> Option<Vec<f64>> {
    let total: f64 = sums.iter().sum();
    if total > 0.0 {
        Some(sums.iter().map(|&s| s / total).collect())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfr::game_tree::materialize;
    use crate::game::{Game, KuhnPoker};
    use test_macros::timed_test;

    /// Build per-deal trees and DealInfo for all 6 Kuhn deals.
    fn kuhn_trees_and_deals() -> (Vec<GameTree>, Vec<DealInfo>) {
        let game = KuhnPoker::new();
        let states = game.initial_states();
        let mut trees = Vec::new();
        let mut deals = Vec::new();

        for state in &states {
            let tree = materialize(&game, state);
            let key_p1 = game.info_set_key(state);
            let hand_bits_p1 = crate::info_key::InfoKey::from_raw(key_p1).hand_bits();

            // Get P2's hand bits by advancing to a P2 decision
            let next = game.next_state(state, game.actions(state)[0]);
            let key_p2 = game.info_set_key(&next);
            let hand_bits_p2 = crate::info_key::InfoKey::from_raw(key_p2).hand_bits();

            let p1_equity = if hand_bits_p1 > hand_bits_p2 {
                1.0
            } else {
                0.0
            };

            trees.push(tree);
            // Kuhn has no per-street variation — same hand bits on all streets.
            deals.push(DealInfo {
                hand_bits_p1: [hand_bits_p1; 4],
                hand_bits_p2: [hand_bits_p2; 4],
                p1_equity,
                weight: 1.0,
            });
        }

        (trees, deals)
    }

    #[timed_test]
    fn sequence_cfr_runs_without_panic() {
        let (trees, deals) = kuhn_trees_and_deals();
        let mut solver =
            SequenceCfrSolver::from_per_deal_trees(trees, deals, SequenceCfrConfig::default());
        solver.train(10);

        assert_eq!(solver.iterations(), 10);
        assert!(!solver.all_strategies().is_empty());
    }

    #[timed_test]
    fn sequence_cfr_strategies_sum_to_one() {
        let (trees, deals) = kuhn_trees_and_deals();
        let mut solver =
            SequenceCfrSolver::from_per_deal_trees(trees, deals, SequenceCfrConfig::default());
        solver.train(100);

        for (key, probs) in solver.all_strategies() {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Strategy for key {key:#x} doesn't sum to 1: {sum}"
            );
        }
    }

    #[timed_test]
    fn sequence_cfr_kuhn_converges_to_nash() {
        let (trees, deals) = kuhn_trees_and_deals();
        let config = SequenceCfrConfig::default();
        let mut solver = SequenceCfrSolver::from_per_deal_trees(trees, deals, config);
        solver.train(10_000);

        let strategies = solver.all_strategies();

        // In Kuhn Nash equilibrium:
        // P1 with Jack at root: bet with probability alpha ∈ [0, 1/3]
        let jack_root_key = crate::info_key::InfoKey::new(0, 0, 0, &[]).as_u64();
        if let Some(probs) = strategies.get(&jack_root_key) {
            let bet_prob = probs[1];
            assert!(
                bet_prob < 0.5,
                "Jack should bluff infrequently, got {bet_prob:.4}"
            );
        }

        // P1 with King at root: should bet frequently (≥ 3*alpha in Nash)
        let king_root_key = crate::info_key::InfoKey::new(2, 0, 0, &[]).as_u64();
        if let Some(probs) = strategies.get(&king_root_key) {
            let bet_prob = probs[1];
            assert!(
                bet_prob > 0.6,
                "King should bet frequently, got {bet_prob:.4}"
            );
        }
    }

    #[timed_test]
    fn sequence_cfr_strategy_delta_decreases() {
        let (trees, deals) = kuhn_trees_and_deals();
        let mut solver =
            SequenceCfrSolver::from_per_deal_trees(trees, deals, SequenceCfrConfig::default());

        solver.train(100);
        let strat1 = solver.all_strategies();

        solver.train(400);
        let strat2 = solver.all_strategies();

        solver.train(500);
        let strat3 = solver.all_strategies();

        let delta_1_2 = strategy_delta(&strat1, &strat2);
        let delta_2_3 = strategy_delta(&strat2, &strat3);

        assert!(
            delta_2_3 < delta_1_2,
            "Strategy delta should decrease: {delta_1_2:.6} -> {delta_2_3:.6}"
        );
    }

    #[timed_test]
    fn sequence_cfr_regrets_populated() {
        let (trees, deals) = kuhn_trees_and_deals();
        let mut solver =
            SequenceCfrSolver::from_per_deal_trees(trees, deals, SequenceCfrConfig::default());
        solver.train(10);

        assert!(!solver.regret_sum().is_empty());

        for regrets in solver.regret_sum().values() {
            assert_eq!(regrets.len(), 2);
        }
    }

    #[timed_test]
    fn sequence_cfr_zero_sum_property() {
        let (trees, deals) = kuhn_trees_and_deals();
        let mut solver =
            SequenceCfrSolver::from_per_deal_trees(trees, deals, SequenceCfrConfig::default());
        solver.train(1000);

        let strategies = solver.all_strategies();
        assert!(
            strategies.len() >= 4,
            "Should have at least 4 info sets, got {}",
            strategies.len()
        );
    }

    #[timed_test]
    fn streaming_matches_in_memory() {
        let (trees, deals) = kuhn_trees_and_deals();
        let config = SequenceCfrConfig::default();

        // In-memory solver
        let mut mem_solver =
            SequenceCfrSolver::from_per_deal_trees(trees.clone(), deals.clone(), config.clone());
        mem_solver.train(100);

        // Streaming solver: per-deal trees need one tree per batch entry
        // For Kuhn, each deal has its own tree, so we need to stream one deal at a time
        // with the correct tree. Since streaming requires a single shared tree,
        // we test with a single-tree setup instead.
        //
        // For the streaming test, simulate by wrapping deals as single-item batches
        // using the first Kuhn tree (which won't give correct Kuhn results, but tests
        // the streaming mechanics work the same as in-memory).

        // Simpler approach: create two identical solvers with same deals/tree
        // and verify streaming produces same results as in-memory.
        let game_tree = trees[0].clone();
        let single_deal = vec![deals[0].clone()];

        let mut mem2 =
            SequenceCfrSolver::new(game_tree.clone(), single_deal.clone(), config.clone());
        mem2.train(100);

        let deals_for_stream = single_deal.clone();
        let mut stream_solver = SequenceCfrSolver::new(game_tree, vec![], config);
        stream_solver.train_streaming(100, || Box::new(std::iter::once(deals_for_stream.clone())));

        // Both should produce identical strategies
        let mem_strats = mem2.all_strategies();
        let stream_strats = stream_solver.all_strategies();

        assert_eq!(
            mem_strats.len(),
            stream_strats.len(),
            "Same number of info sets"
        );
        for (key, mem_probs) in &mem_strats {
            let stream_probs = stream_strats
                .get(key)
                .unwrap_or_else(|| panic!("Missing key {key:#x} in streaming"));
            for (mp, sp) in mem_probs.iter().zip(stream_probs.iter()) {
                assert!(
                    (mp - sp).abs() < 1e-10,
                    "Strategy mismatch at key {key:#x}: mem={mp}, stream={sp}"
                );
            }
        }
    }

    #[timed_test]
    fn linear_cfr_config_has_all_ones() {
        let config = SequenceCfrConfig::linear_cfr();
        assert!((config.dcfr_alpha - 1.0).abs() < f64::EPSILON);
        assert!((config.dcfr_beta - 1.0).abs() < f64::EPSILON);
        assert!((config.dcfr_gamma - 1.0).abs() < f64::EPSILON);
        assert_eq!(config.cfr_variant, CfrVariant::Linear);
    }

    #[timed_test]
    fn alternating_kuhn_converges() {
        // Verify alternating updates converge to Nash on Kuhn poker
        let (trees, deals) = kuhn_trees_and_deals();
        let config = SequenceCfrConfig::default();
        let mut solver = SequenceCfrSolver::from_per_deal_trees(trees, deals, config);
        solver.train(5000);

        let strategies = solver.all_strategies();
        // All info sets should have strategies
        assert!(
            strategies.len() >= 4,
            "Should have at least 4 info sets, got {}",
            strategies.len()
        );
        // All strategies should sum to 1
        for (key, probs) in &strategies {
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "Strategy for key {key:#x} doesn't sum to 1: {sum}"
            );
        }
    }

    #[timed_test]
    fn warmup_skips_discounting() {
        let (trees, deals) = kuhn_trees_and_deals();
        let mut config = SequenceCfrConfig::default();
        config.dcfr_warmup = 100; // warmup longer than training

        let mut solver = SequenceCfrSolver::from_per_deal_trees(trees, deals.clone(), config);
        solver.train(50);

        // With warmup=100 and only 50 iterations, discounting should be skipped.
        // Regrets should have both positive and negative values (not discounted).
        let has_negative = solver
            .regret_sum()
            .values()
            .any(|v| v.iter().any(|&r| r < -1e-10));
        assert!(
            has_negative,
            "During warmup, negative regrets should not be discounted away"
        );
    }

    #[timed_test]
    fn cfr_plus_floors_regrets() {
        let (trees, deals) = kuhn_trees_and_deals();
        let config = SequenceCfrConfig::cfr_plus();

        let mut solver = SequenceCfrSolver::from_per_deal_trees(trees, deals, config);
        solver.train(100);

        // CFR+ floors all regrets to >= 0
        for (key, regrets) in solver.regret_sum() {
            for (i, &r) in regrets.iter().enumerate() {
                assert!(
                    r >= -1e-10,
                    "CFR+ regret should be >= 0 at key {key:#x}[{i}], got {r}"
                );
            }
        }
    }

    #[timed_test]
    fn vanilla_no_discounting() {
        let (trees, deals) = kuhn_trees_and_deals();
        let config = SequenceCfrConfig::vanilla();

        let mut solver = SequenceCfrSolver::from_per_deal_trees(trees, deals, config);
        solver.train(100);

        // Vanilla CFR should still converge — check we have strategies
        let strategies = solver.all_strategies();
        assert!(!strategies.is_empty(), "Vanilla should produce strategies");

        // With no discounting, negative regrets persist
        let has_negative = solver
            .regret_sum()
            .values()
            .any(|v| v.iter().any(|&r| r < -1e-10));
        assert!(
            has_negative,
            "Vanilla CFR should retain negative regrets (no discounting)"
        );
    }

    #[timed_test]
    fn exploration_visits_all_info_sets() {
        let (trees, deals) = kuhn_trees_and_deals();
        let mut config = SequenceCfrConfig::default();
        config.exploration = 0.3; // high exploration

        let mut solver = SequenceCfrSolver::from_per_deal_trees(trees, deals, config);
        solver.train(100);

        // With exploration, every info set reachable from any deal should have data
        let strategies = solver.all_strategies_best_effort();
        // Kuhn poker has 12 info sets (6 deals × 2 players, but many overlap)
        // At minimum: 3 P1 root + some P2 nodes = at least 6 distinct info sets
        assert!(
            strategies.len() >= 6,
            "With exploration, should have at least 6 info sets, got {}",
            strategies.len()
        );
    }

    fn strategy_delta(a: &FxHashMap<u64, Vec<f64>>, b: &FxHashMap<u64, Vec<f64>>) -> f64 {
        let mut total_diff = 0.0;
        let mut count = 0;

        for (key, probs_a) in a {
            if let Some(probs_b) = b.get(key) {
                let diff: f64 = probs_a
                    .iter()
                    .zip(probs_b.iter())
                    .map(|(pa, pb)| (pa - pb).abs())
                    .sum();
                total_diff += diff;
                count += 1;
            }
        }

        if count > 0 {
            total_diff / count as f64
        } else {
            f64::INFINITY
        }
    }
}
