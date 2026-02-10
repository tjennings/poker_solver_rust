use rustc_hash::FxHashMap;

use crate::game::{Game, Player};

use super::regret::regret_match;

/// Vanilla CFR (Counterfactual Regret Minimization) solver.
///
/// This is a reference implementation that traverses the game tree recursively.
/// For each information set, it tracks cumulative regrets and strategy sums.
pub struct VanillaCfr<G: Game> {
    game: G,
    /// Cumulative regrets per info set: `info_set_key` -> regrets per action
    regret_sum: FxHashMap<u64, Vec<f64>>,
    /// Cumulative strategy sums per info set (for averaging)
    strategy_sum: FxHashMap<u64, Vec<f64>>,
}

impl<G: Game> VanillaCfr<G> {
    /// Creates a new CFR solver for the given game.
    #[must_use]
    pub fn new(game: G) -> Self {
        Self {
            game,
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
        }
    }

    /// Runs CFR for the specified number of iterations.
    pub fn train(&mut self, iterations: u64) {
        for _ in 0..iterations {
            for initial_state in self.game.initial_states() {
                // Traverse for both players to update all info sets
                self.cfr(&initial_state, Player::Player1, 1.0, 1.0);
                self.cfr(&initial_state, Player::Player2, 1.0, 1.0);
            }
        }
    }

    /// Returns the average strategy for an information set.
    #[must_use]
    pub fn get_average_strategy(&self, info_set: u64) -> Option<Vec<f64>> {
        self.strategy_sum.get(&info_set).map(|sums| {
            let total: f64 = sums.iter().sum();
            if total > 0.0 {
                sums.iter().map(|&s| s / total).collect()
            } else {
                #[allow(clippy::cast_precision_loss)]
                let uniform = 1.0 / sums.len() as f64;
                vec![uniform; sums.len()]
            }
        })
    }

    /// Core CFR recursive function.
    ///
    /// Returns the expected utility for the traversing player from this state.
    fn cfr(
        &mut self,
        state: &G::State,
        traversing_player: Player,
        p1_reach: f64,
        p2_reach: f64,
    ) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.utility(state, traversing_player);
        }

        let current_player = self.game.player(state);
        let actions = self.game.actions(state);
        let num_actions = actions.len();
        let info_set = self.game.info_set_key(state);

        // Get current strategy from regrets
        let strategy = self.get_strategy(info_set, num_actions);

        // Compute utility for each action
        let mut action_utils = vec![0.0; num_actions];

        for (i, action) in actions.iter().enumerate() {
            let next_state = self.game.next_state(state, *action);

            let (new_p1_reach, new_p2_reach) = match current_player {
                Player::Player1 => (p1_reach * strategy[i], p2_reach),
                Player::Player2 => (p1_reach, p2_reach * strategy[i]),
            };

            action_utils[i] = self.cfr(&next_state, traversing_player, new_p1_reach, new_p2_reach);
        }

        // Expected utility under current strategy
        let node_util: f64 = action_utils
            .iter()
            .zip(strategy.iter())
            .map(|(u, p)| u * p)
            .sum();

        // Update regrets and strategy sum only for the current player
        if current_player == traversing_player {
            let opponent_reach = match current_player {
                Player::Player1 => p2_reach,
                Player::Player2 => p1_reach,
            };

            let regrets = self
                .regret_sum
                .entry(info_set)
                .or_insert_with(|| vec![0.0; num_actions]);

            for i in 0..num_actions {
                regrets[i] += opponent_reach * (action_utils[i] - node_util);
            }

            let my_reach = match current_player {
                Player::Player1 => p1_reach,
                Player::Player2 => p2_reach,
            };

            let strat_sums = self
                .strategy_sum
                .entry(info_set)
                .or_insert_with(|| vec![0.0; num_actions]);

            for i in 0..num_actions {
                strat_sums[i] += my_reach * strategy[i];
            }
        }

        node_util
    }

    /// Gets the current strategy for an information set using regret matching.
    fn get_strategy(&self, info_set: u64, num_actions: usize) -> Vec<f64> {
        if let Some(regrets) = self.regret_sum.get(&info_set) {
            regret_match(regrets)
        } else {
            #[allow(clippy::cast_precision_loss)]
            let uniform = 1.0 / num_actions as f64;
            vec![uniform; num_actions]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::KuhnPoker;
    use crate::info_key::InfoKey;
    use test_macros::timed_test;

    #[timed_test]
    fn solver_initializes_empty() {
        let game = KuhnPoker::new();
        let solver = VanillaCfr::new(game);

        assert!(solver.regret_sum.is_empty());
        assert!(solver.strategy_sum.is_empty());
    }

    #[timed_test]
    fn training_populates_info_sets() {
        let game = KuhnPoker::new();
        let mut solver = VanillaCfr::new(game);

        solver.train(10);

        // Should have info sets for each card at each decision point
        assert!(!solver.strategy_sum.is_empty());
        // K at root
        let k = InfoKey::new(2, 0, 0, 0, &[]).as_u64();
        assert!(solver.strategy_sum.contains_key(&k));
        // J after check (Jc)
        let jc = InfoKey::new(0, 0, 0, 0, &[2]).as_u64();
        assert!(solver.strategy_sum.contains_key(&jc));
    }

    #[timed_test]
    fn average_strategy_sums_to_one() {
        let game = KuhnPoker::new();
        let mut solver = VanillaCfr::new(game);

        solver.train(100);

        for &info_set in solver.strategy_sum.keys() {
            if let Some(strategy) = solver.get_average_strategy(info_set) {
                let sum: f64 = strategy.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Strategy for {info_set} sums to {sum}"
                );
            }
        }
    }

    #[timed_test]
    fn king_always_bets_or_calls() {
        let game = KuhnPoker::new();
        let mut solver = VanillaCfr::new(game);

        solver.train(10_000);

        // K at root: check or bet — betting is optimal
        let k = InfoKey::new(2, 0, 0, 0, &[]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(k) {
            assert!(
                strategy[1] > 0.5,
                "King should bet frequently, got {strategy:?}"
            );
        }

        // Kb = King facing bet, should always call
        let kb = InfoKey::new(2, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(kb) {
            assert!(
                strategy[1] > 0.99,
                "King should always call a bet, got {strategy:?}"
            );
        }
    }
}
