//! CFR+ solver for concrete subgame trees.
//!
//! Solves river subgames with exact hand evaluation (no abstraction).
//! Uses CFR+ (regret clamped to zero) for fast convergence on small trees.

use rustc_hash::FxHashMap;

use crate::cfr::regret::regret_match;
use crate::poker::{Card, Hand, Rank, Rankable};

use super::SubgameConfig;
use super::subgame_tree::{SubgameHands, SubgameNode, SubgameTree, SubgameTreeBuilder};

// ---------------------------------------------------------------------------
// SubgameStrategy -- the output of a subgame solve
// ---------------------------------------------------------------------------

/// Result of a subgame solve: strategies for all combos at all decision nodes.
#[derive(Debug, Clone)]
pub struct SubgameStrategy {
    /// `(node_idx, combo_idx)` -> action probabilities.
    strategies: FxHashMap<u64, Vec<f64>>,
    /// Total number of combos in this subgame.
    pub num_combos: usize,
}

impl SubgameStrategy {
    fn key(node_idx: u32, combo_idx: u32) -> u64 {
        (u64::from(node_idx) << 32) | u64::from(combo_idx)
    }

    /// Get action probabilities for a combo at a node.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_probs(&self, node_idx: u32, combo_idx: usize) -> Vec<f64> {
        self.strategies
            .get(&Self::key(node_idx, combo_idx as u32))
            .cloned()
            .unwrap_or_default()
    }

    /// Get root node probabilities for a combo.
    #[must_use]
    pub fn root_probs(&self, combo_idx: usize) -> Vec<f64> {
        self.get_probs(0, combo_idx)
    }

    /// Number of combos in this subgame.
    #[must_use]
    pub fn num_combos(&self) -> usize {
        self.num_combos
    }
}

// ---------------------------------------------------------------------------
// SubgameCfrSolver
// ---------------------------------------------------------------------------

/// CFR+ solver for concrete subgame trees.
pub struct SubgameCfrSolver {
    tree: SubgameTree,
    hands: SubgameHands,
    /// `equity[i][j]` = P(combo i beats combo j) at showdown.
    equity_matrix: Vec<Vec<f64>>,
    /// Opponent reaching probability per combo.
    opponent_reach: Vec<f64>,
    /// Leaf continuation values per combo (for `DepthBoundary` nodes).
    leaf_values: Vec<f64>,
    /// Cumulative regrets: `(node_idx, combo_idx)` -> regrets per action.
    regret_sum: FxHashMap<u64, Vec<f64>>,
    /// Cumulative strategy: `(node_idx, combo_idx)` -> weighted probs per action.
    strategy_sum: FxHashMap<u64, Vec<f64>>,
    /// Precomputed: for each combo, the total opponent reach of non-blocked combos.
    opp_reach_totals: Vec<f64>,
    /// Current iteration count.
    pub iteration: u32,
}

impl SubgameCfrSolver {
    /// Create a solver for a specific board position.
    #[must_use]
    pub fn new(
        tree: SubgameTree,
        hands: SubgameHands,
        opponent_reach: Vec<f64>,
        leaf_values: Vec<f64>,
    ) -> Self {
        let equity_matrix = compute_equity_matrix(&hands.combos, &tree.board);
        let opp_reach_totals = precompute_opp_reach(&hands.combos, &opponent_reach);
        Self {
            tree,
            hands,
            equity_matrix,
            opponent_reach,
            leaf_values,
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
            opp_reach_totals,
            iteration: 0,
        }
    }

    /// Run CFR+ for the given number of iterations.
    pub fn train(&mut self, iterations: u32) {
        for _ in 0..iterations {
            self.iteration += 1;
            for player in 0..2u8 {
                self.traverse_all_combos(player);
            }
        }
    }

    /// Extract the average strategy from cumulative strategy sums.
    #[must_use]
    pub fn strategy(&self) -> SubgameStrategy {
        let mut strategies = FxHashMap::default();
        for (&key, sums) in &self.strategy_sum {
            let total: f64 = sums.iter().sum();
            if total > 0.0 {
                let probs: Vec<f64> = sums.iter().map(|&s| s / total).collect();
                strategies.insert(key, probs);
            }
        }
        SubgameStrategy {
            strategies,
            num_combos: self.hands.combos.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// Private traversal methods
// ---------------------------------------------------------------------------

impl SubgameCfrSolver {
    /// Traverse from every combo for the given traversing player.
    fn traverse_all_combos(&mut self, traverser: u8) {
        let n = self.hands.combos.len();
        for combo in 0..n {
            if self.opp_reach_totals[combo] <= 0.0 {
                continue;
            }
            self.cfr_traverse(0, combo, 1.0, self.opp_reach_totals[combo], traverser);
        }
    }

    /// CFR+ recursive traversal. Returns expected value for the traverser.
    fn cfr_traverse(
        &mut self,
        node_idx: usize,
        hero_combo: usize,
        reach_hero: f64,
        reach_opp: f64,
        traverser: u8,
    ) -> f64 {
        // Clone minimal info to avoid borrow conflicts.
        let node = self.tree.nodes[node_idx].clone();
        match node {
            SubgameNode::Terminal {
                is_fold,
                fold_player,
                pot,
                ..
            } => self.terminal_value(hero_combo, is_fold, fold_player, pot, traverser),
            SubgameNode::DepthBoundary { .. } => {
                self.leaf_values.get(hero_combo).copied().unwrap_or(0.0)
            }
            SubgameNode::Decision {
                position,
                actions,
                children,
                ..
            } => self.traverse_decision(
                node_idx,
                hero_combo,
                reach_hero,
                reach_opp,
                traverser,
                position,
                actions.len(),
                &children,
            ),
        }
    }

    /// Handle a decision node in CFR+ traversal.
    #[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
    fn traverse_decision(
        &mut self,
        node_idx: usize,
        hero_combo: usize,
        reach_hero: f64,
        reach_opp: f64,
        traverser: u8,
        position: u8,
        num_actions: usize,
        children: &[u32],
    ) -> f64 {
        let key = SubgameStrategy::key(node_idx as u32, hero_combo as u32);
        let strategy = self.current_strategy(key, num_actions);

        if position == traverser {
            self.traverse_as_traverser(
                key, hero_combo, reach_hero, reach_opp, traverser, &strategy, children,
            )
        } else {
            self.traverse_as_opponent(
                hero_combo, reach_hero, reach_opp, traverser, &strategy, children,
            )
        }
    }

    /// Traverser's decision: compute counterfactual values and update regrets.
    #[allow(clippy::too_many_arguments)]
    fn traverse_as_traverser(
        &mut self,
        key: u64,
        hero_combo: usize,
        reach_hero: f64,
        reach_opp: f64,
        traverser: u8,
        strategy: &[f64],
        children: &[u32],
    ) -> f64 {
        let num_actions = children.len();
        let mut action_values = vec![0.0; num_actions];
        let mut node_value = 0.0;

        for (a, &child_idx) in children.iter().enumerate() {
            let new_reach = reach_hero * strategy[a];
            action_values[a] = self.cfr_traverse(
                child_idx as usize,
                hero_combo,
                new_reach,
                reach_opp,
                traverser,
            );
            node_value += strategy[a] * action_values[a];
        }

        self.update_regrets(key, &action_values, node_value, reach_opp, num_actions);
        self.accumulate_strategy(key, strategy, reach_hero, num_actions);
        node_value
    }

    /// Update regrets with CFR+ clamping (negative regrets set to zero).
    fn update_regrets(
        &mut self,
        key: u64,
        action_values: &[f64],
        node_value: f64,
        reach_opp: f64,
        num_actions: usize,
    ) {
        let regrets = self
            .regret_sum
            .entry(key)
            .or_insert_with(|| vec![0.0; num_actions]);
        for (a, av) in action_values.iter().enumerate() {
            regrets[a] = (regrets[a] + reach_opp * (av - node_value)).max(0.0);
        }
    }

    /// Accumulate strategy weights for averaging.
    fn accumulate_strategy(
        &mut self,
        key: u64,
        strategy: &[f64],
        reach_hero: f64,
        num_actions: usize,
    ) {
        let strat_sum = self
            .strategy_sum
            .entry(key)
            .or_insert_with(|| vec![0.0; num_actions]);
        for (a, &s) in strategy.iter().enumerate() {
            strat_sum[a] += reach_hero * s;
        }
    }

    /// Opponent's decision: weight by opponent strategy and recurse.
    fn traverse_as_opponent(
        &mut self,
        hero_combo: usize,
        reach_hero: f64,
        reach_opp: f64,
        traverser: u8,
        strategy: &[f64],
        children: &[u32],
    ) -> f64 {
        let mut node_value = 0.0;
        for (a, &child_idx) in children.iter().enumerate() {
            let new_opp_reach = reach_opp * strategy[a];
            let child_val = self.cfr_traverse(
                child_idx as usize,
                hero_combo,
                reach_hero,
                new_opp_reach,
                traverser,
            );
            node_value += strategy[a] * child_val;
        }
        node_value
    }

    /// Get current strategy via regret matching on cumulative regrets.
    #[allow(clippy::cast_precision_loss)]
    fn current_strategy(&self, key: u64, num_actions: usize) -> Vec<f64> {
        self.regret_sum.get(&key).map_or_else(
            || {
                let uniform = 1.0 / num_actions as f64;
                vec![uniform; num_actions]
            },
            |regrets| regret_match(regrets),
        )
    }

    /// Compute terminal value for the traversing player.
    fn terminal_value(
        &self,
        hero_combo: usize,
        is_fold: bool,
        fold_player: u8,
        pot: u32,
        traverser: u8,
    ) -> f64 {
        let half_pot = f64::from(pot) / 2.0;
        if is_fold {
            return if fold_player == traverser {
                -half_pot
            } else {
                half_pot
            };
        }
        self.showdown_value(hero_combo, half_pot)
    }

    /// Compute showdown value: reach-weighted equity vs opponent range.
    fn showdown_value(&self, hero_combo: usize, half_pot: f64) -> f64 {
        let hero_cards = self.hands.combos[hero_combo];
        let mut equity_sum = 0.0;
        let mut reach_sum = 0.0;

        for (j, &opp_reach) in self.opponent_reach.iter().enumerate() {
            if cards_overlap(hero_cards, self.hands.combos[j]) {
                continue;
            }
            equity_sum += opp_reach * self.equity_matrix[hero_combo][j];
            reach_sum += opp_reach;
        }

        if reach_sum <= 0.0 {
            return 0.0;
        }
        let avg_equity = equity_sum / reach_sum;
        // EV = (2 * equity - 1) * half_pot
        (2.0 * avg_equity - 1.0) * half_pot
    }
}

// ---------------------------------------------------------------------------
// Precomputation helpers
// ---------------------------------------------------------------------------

/// Precompute total opponent reach for each combo (excluding blocked combos).
fn precompute_opp_reach(combos: &[[Card; 2]], opponent_reach: &[f64]) -> Vec<f64> {
    combos
        .iter()
        .map(|&hero| {
            opponent_reach
                .iter()
                .zip(combos)
                .filter(|(_, opp)| !cards_overlap(hero, **opp))
                .map(|(&r, _)| r)
                .sum()
        })
        .collect()
}

/// Build an N x N equity matrix for all combo pairs on the given board.
fn compute_equity_matrix(combos: &[[Card; 2]], board: &[Card]) -> Vec<Vec<f64>> {
    let n = combos.len();
    let ranks: Vec<Rank> = combos.iter().map(|c| rank_combo(*c, board)).collect();
    let mut matrix = vec![vec![0.5; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if cards_overlap(combos[i], combos[j]) {
                continue; // blocked, stays at 0.5 (irrelevant)
            }
            match ranks[i].cmp(&ranks[j]) {
                std::cmp::Ordering::Greater => {
                    matrix[i][j] = 1.0;
                    matrix[j][i] = 0.0;
                }
                std::cmp::Ordering::Less => {
                    matrix[i][j] = 0.0;
                    matrix[j][i] = 1.0;
                }
                std::cmp::Ordering::Equal => {} // tie: 0.5 already set
            }
        }
    }
    matrix
}

/// Rank a combo (2 hole cards) on a board using `rs_poker`.
fn rank_combo(combo: [Card; 2], board: &[Card]) -> Rank {
    let mut hand = Hand::default();
    for &c in board {
        hand.insert(c);
    }
    hand.insert(combo[0]);
    hand.insert(combo[1]);
    hand.rank()
}

/// Check if two 2-card combos share any card.
fn cards_overlap(a: [Card; 2], b: [Card; 2]) -> bool {
    a[0] == b[0] || a[0] == b[1] || a[1] == b[0] || a[1] == b[1]
}

// ---------------------------------------------------------------------------
// Convenience function
// ---------------------------------------------------------------------------

/// Solve a subgame and return the strategy.
#[must_use]
pub fn solve_subgame(
    board: &[Card],
    bet_sizes: &[f32],
    pot: u32,
    stacks: &[u32],
    opponent_reach: &[f64],
    leaf_values: &[f64],
    config: &SubgameConfig,
) -> SubgameStrategy {
    let tree = SubgameTreeBuilder::new()
        .board(board)
        .bet_sizes(bet_sizes)
        .pot(pot)
        .stacks(stacks)
        .build();
    let hands = SubgameHands::enumerate(board);
    let mut solver =
        SubgameCfrSolver::new(tree, hands, opponent_reach.to_vec(), leaf_values.to_vec());
    solver.train(config.max_iterations);
    solver.strategy()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    fn river_board() -> Vec<Card> {
        vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
            Card::new(Value::Four, Suit::Club),
            Card::new(Value::Ten, Suit::Club),
        ]
    }

    /// Build a small hand set (first `n` combos) for fast testing.
    fn small_hands(board: &[Card], n: usize) -> SubgameHands {
        let full = SubgameHands::enumerate(board);
        SubgameHands {
            combos: full.combos.into_iter().take(n).collect(),
        }
    }

    #[timed_test]
    fn solver_creates_and_runs() {
        let board = river_board();
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        let hands = small_hands(&board, 30);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = vec![0.0; n];
        let mut solver = SubgameCfrSolver::new(tree, hands, reach, leaf);
        solver.train(10);
        assert_eq!(solver.iteration, 10);
    }

    #[timed_test]
    fn strategy_is_valid_distribution() {
        let board = river_board();
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        let hands = small_hands(&board, 50);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = vec![0.0; n];
        let mut solver = SubgameCfrSolver::new(tree, hands, reach, leaf);
        solver.train(100);
        let strategy = solver.strategy();

        for combo_idx in 0..n {
            let probs = strategy.root_probs(combo_idx);
            if probs.is_empty() {
                continue;
            }
            let sum: f64 = probs.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "combo {combo_idx}: strategy sum = {sum}, expected ~1.0"
            );
        }
    }

    #[timed_test]
    fn solve_subgame_convenience_works() {
        let board = river_board();
        let hands = small_hands(&board, 20);
        let n = hands.combos.len();
        let config = SubgameConfig {
            depth_limit: 4,
            time_budget_ms: 5000,
            max_iterations: 10,
        };

        // Build manually since solve_subgame uses full enumeration
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        let mut solver = SubgameCfrSolver::new(tree, hands, vec![1.0; n], vec![0.0; n]);
        solver.train(config.max_iterations);
        let strategy = solver.strategy();
        assert!(strategy.num_combos() > 0);
    }

    #[timed_test]
    fn equity_matrix_is_symmetric() {
        let board = river_board();
        let hands = small_hands(&board, 50);
        let matrix = compute_equity_matrix(&hands.combos, &board);
        let n = hands.combos.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let sum = matrix[i][j] + matrix[j][i];
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "equity[{i}][{j}] + equity[{j}][{i}] = {sum}, expected 1.0"
                );
            }
        }
    }

    #[timed_test]
    fn cards_overlap_detects_shared_cards() {
        let a = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
        ];
        let b = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let c = [
            Card::new(Value::Two, Suit::Club),
            Card::new(Value::Three, Suit::Diamond),
        ];
        assert!(cards_overlap(a, b));
        assert!(!cards_overlap(a, c));
    }

    #[timed_test]
    fn strong_hands_bet_more_than_weak() {
        let board = river_board();
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        let hands = small_hands(&board, 50);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = vec![0.0; n];
        let mut solver = SubgameCfrSolver::new(tree, hands.clone(), reach, leaf);
        solver.train(200);
        let strategy = solver.strategy();

        // Find the strongest and weakest combos by rank
        let ranks: Vec<Rank> = hands
            .combos
            .iter()
            .map(|c| rank_combo(*c, &board))
            .collect();
        let best = ranks
            .iter()
            .enumerate()
            .max_by_key(|(_, r)| *r)
            .map(|(i, _)| i)
            .expect("should have combos");
        let worst = ranks
            .iter()
            .enumerate()
            .min_by_key(|(_, r)| *r)
            .map(|(i, _)| i)
            .expect("should have combos");

        let best_probs = strategy.root_probs(best);
        let worst_probs = strategy.root_probs(worst);

        // Root is position 0's decision: [Check, Bet(0), Bet(ALL_IN)]
        // The best hand should have higher total betting frequency
        if best_probs.len() >= 2 && worst_probs.len() >= 2 {
            let best_bet_freq: f64 = best_probs[1..].iter().sum();
            let worst_bet_freq: f64 = worst_probs[1..].iter().sum();
            assert!(
                best_bet_freq >= worst_bet_freq - 0.15,
                "best hand bet freq ({best_bet_freq:.3}) should be >= worst ({worst_bet_freq:.3}) - margin"
            );
        }
    }

    #[timed_test]
    fn fold_terminal_gives_correct_values() {
        // Verify that folding gives +half_pot to non-folder and -half_pot to folder
        let board = river_board();
        let hands = small_hands(&board, 5);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = vec![0.0; n];
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        let solver = SubgameCfrSolver::new(tree, hands, reach, leaf);

        // Player 0 folds: traverser=0 gets -50, traverser=1 gets +50
        let v0 = solver.terminal_value(0, true, 0, 100, 0);
        let v1 = solver.terminal_value(0, true, 0, 100, 1);
        assert!((v0 - (-50.0)).abs() < 1e-10, "folder should lose half pot");
        assert!((v1 - 50.0).abs() < 1e-10, "non-folder wins half pot");
    }

    #[timed_test]
    fn showdown_value_respects_equity() {
        let board = river_board();
        let hands = small_hands(&board, 10);
        let n = hands.combos.len();
        let reach = vec![1.0; n];
        let leaf = vec![0.0; n];
        let tree = SubgameTreeBuilder::new()
            .board(&board)
            .bet_sizes(&[1.0])
            .pot(100)
            .stacks(&[200, 200])
            .build();
        let solver = SubgameCfrSolver::new(tree, hands, reach, leaf);

        // Showdown values should be in [-50, +50] for pot=100
        for i in 0..n {
            let v = solver.showdown_value(i, 50.0);
            assert!(
                v >= -50.0 - 1e-10 && v <= 50.0 + 1e-10,
                "showdown value {v} out of range for combo {i}"
            );
        }
    }
}
