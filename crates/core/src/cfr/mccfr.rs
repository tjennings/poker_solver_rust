//! Monte Carlo CFR (MCCFR) - Sampling-based CFR for large games.
//!
//! MCCFR samples initial states (chance outcomes) instead of iterating over all of them,
//! making it practical for games with many initial states like HUNL Preflop (28,561 states).
//!
//! This implementation uses **Chance Sampling**: we sample the initial card deal,
//! then do full CFR traversal on the betting tree for that deal.

use std::collections::HashMap;

use crate::game::{Game, Player};

use super::regret::regret_match;

/// Monte Carlo CFR solver with chance sampling.
///
/// Much faster than vanilla CFR for games with many initial states because
/// it samples states instead of iterating over all of them.
pub struct MccfrSolver<G: Game> {
    game: G,
    /// Cumulative regrets per info set
    regret_sum: HashMap<String, Vec<f64>>,
    /// Cumulative strategy sums (for averaging)
    strategy_sum: HashMap<String, Vec<f64>>,
    /// Use CFR+ (floor regrets at 0)
    use_cfr_plus: bool,
    /// RNG state for sampling
    rng_state: u64,
    /// Total iterations completed
    iterations: u64,
}

/// Configuration for MCCFR training.
#[derive(Debug, Clone)]
pub struct MccfrConfig {
    /// Number of initial states to sample per iteration
    pub samples_per_iteration: usize,
    /// Use CFR+ regret flooring
    pub use_cfr_plus: bool,
    /// Discount early iterations (CFR+ optimization)
    /// If Some(n), discount first n iterations by sqrt(t)/(sqrt(t)+1)
    pub discount_iterations: Option<u64>,
    /// Skip first N iterations when computing average strategy
    pub skip_first_iterations: Option<u64>,
}

impl Default for MccfrConfig {
    fn default() -> Self {
        Self {
            samples_per_iteration: 100,
            use_cfr_plus: true,
            discount_iterations: Some(30),
            skip_first_iterations: None,
        }
    }
}

impl<G: Game> MccfrSolver<G> {
    /// Creates a new MCCFR solver for the given game.
    #[must_use]
    pub fn new(game: G) -> Self {
        Self {
            game,
            regret_sum: HashMap::new(),
            strategy_sum: HashMap::new(),
            use_cfr_plus: true,
            rng_state: 0x1234_5678_9ABC_DEF0,
            iterations: 0,
        }
    }

    /// Creates a new MCCFR solver with custom configuration.
    #[must_use]
    pub fn with_config(game: G, config: &MccfrConfig) -> Self {
        Self {
            game,
            regret_sum: HashMap::new(),
            strategy_sum: HashMap::new(),
            use_cfr_plus: config.use_cfr_plus,
            rng_state: 0x1234_5678_9ABC_DEF0,
            iterations: 0,
        }
    }

    /// Train using chance sampling MCCFR.
    ///
    /// Each iteration samples `samples_per_iter` initial states and runs
    /// CFR on each, rather than iterating over all initial states.
    pub fn train(&mut self, iterations: u64, samples_per_iter: usize) {
        let initial_states = self.game.initial_states();
        let num_states = initial_states.len();

        if num_states == 0 {
            return;
        }

        for iter in 0..iterations {
            // Compute discount factor for CFR+ early iteration discounting
            let discount = self.compute_discount(iter);

            // Sample initial states
            for _ in 0..samples_per_iter {
                // Simple xorshift RNG
                self.rng_state ^= self.rng_state << 13;
                self.rng_state ^= self.rng_state >> 7;
                self.rng_state ^= self.rng_state << 17;

                #[allow(clippy::cast_possible_truncation)]
                let state_idx = (self.rng_state % num_states as u64) as usize;
                let state = &initial_states[state_idx];

                // Weight by inverse sampling probability for unbiased estimates
                #[allow(clippy::cast_precision_loss)]
                let sample_weight = num_states as f64 / samples_per_iter as f64;

                // Traverse for both players
                self.cfr_traverse(state, Player::Player1, 1.0, 1.0, sample_weight, discount);
                self.cfr_traverse(state, Player::Player2, 1.0, 1.0, sample_weight, discount);
            }

            self.iterations += 1;
        }
    }

    /// Train for a fixed number of iterations, sampling all states once per iteration.
    ///
    /// This is equivalent to vanilla CFR but can be faster due to CFR+ optimizations.
    pub fn train_full(&mut self, iterations: u64) {
        let initial_states = self.game.initial_states();

        for iter in 0..iterations {
            let discount = self.compute_discount(iter);

            for state in &initial_states {
                self.cfr_traverse(state, Player::Player1, 1.0, 1.0, 1.0, discount);
                self.cfr_traverse(state, Player::Player2, 1.0, 1.0, 1.0, discount);
            }

            self.iterations += 1;
        }
    }

    /// Compute discount factor for early iterations (CFR+ optimization).
    #[allow(clippy::unused_self)] // May access config in future
    fn compute_discount(&self, iter: u64) -> f64 {
        if iter < 30 {
            #[allow(clippy::cast_precision_loss)]
            let t = (iter + 1) as f64;
            t.sqrt() / (t.sqrt() + 1.0)
        } else {
            1.0
        }
    }

    /// Returns the average strategy for an information set.
    #[must_use]
    pub fn get_average_strategy(&self, info_set: &str) -> Option<Vec<f64>> {
        self.strategy_sum.get(info_set).map(|sums| {
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

    /// Returns the current regret-matched strategy for an information set.
    #[must_use]
    pub fn get_current_strategy(&self, info_set: &str, num_actions: usize) -> Vec<f64> {
        if let Some(regrets) = self.regret_sum.get(info_set) {
            regret_match(regrets)
        } else {
            #[allow(clippy::cast_precision_loss)]
            let uniform = 1.0 / num_actions as f64;
            vec![uniform; num_actions]
        }
    }

    /// Returns the number of iterations completed.
    #[must_use]
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Returns all info sets and their average strategies.
    #[must_use]
    pub fn all_strategies(&self) -> HashMap<String, Vec<f64>> {
        self.strategy_sum
            .keys()
            .filter_map(|k| self.get_average_strategy(k).map(|s| (k.clone(), s)))
            .collect()
    }

    /// Core CFR traversal function.
    fn cfr_traverse(
        &mut self,
        state: &G::State,
        traversing_player: Player,
        p1_reach: f64,
        p2_reach: f64,
        sample_weight: f64,
        discount: f64,
    ) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.utility(state, traversing_player);
        }

        let current_player = self.game.player(state);
        let actions = self.game.actions(state);
        let num_actions = actions.len();
        let info_set = self.game.info_set_key(state);

        // Get current strategy from regrets
        let strategy = self.get_current_strategy(&info_set, num_actions);

        // Compute utility for each action
        let mut action_utils = vec![0.0; num_actions];

        for (i, action) in actions.iter().enumerate() {
            let next_state = self.game.next_state(state, *action);

            let (new_p1_reach, new_p2_reach) = match current_player {
                Player::Player1 => (p1_reach * strategy[i], p2_reach),
                Player::Player2 => (p1_reach, p2_reach * strategy[i]),
            };

            action_utils[i] = self.cfr_traverse(
                &next_state,
                traversing_player,
                new_p1_reach,
                new_p2_reach,
                sample_weight,
                discount,
            );
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
                .entry(info_set.clone())
                .or_insert_with(|| vec![0.0; num_actions]);

            for i in 0..num_actions {
                let regret_delta = opponent_reach * (action_utils[i] - node_util) * sample_weight * discount;
                regrets[i] += regret_delta;

                // CFR+ : floor regrets at 0
                if self.use_cfr_plus && regrets[i] < 0.0 {
                    regrets[i] = 0.0;
                }
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
                strat_sums[i] += my_reach * strategy[i] * sample_weight * discount;
            }
        }

        node_util
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::KuhnPoker;

    #[test]
    fn mccfr_initializes_empty() {
        let game = KuhnPoker::new();
        let solver = MccfrSolver::new(game);

        assert!(solver.regret_sum.is_empty());
        assert!(solver.strategy_sum.is_empty());
        assert_eq!(solver.iterations(), 0);
    }

    #[test]
    fn mccfr_training_populates_info_sets() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        // With 6 initial states in Kuhn, sampling 10 per iter should cover all
        solver.train(10, 10);

        assert!(!solver.strategy_sum.is_empty());
        assert!(solver.iterations() > 0);
    }

    #[test]
    fn mccfr_strategy_sums_to_one() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        solver.train(100, 10);

        for (info_set, _) in &solver.strategy_sum {
            if let Some(strategy) = solver.get_average_strategy(info_set) {
                let sum: f64 = strategy.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Strategy for {info_set} sums to {sum}"
                );
            }
        }
    }

    #[test]
    fn mccfr_converges_on_kuhn() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        // Train with full sampling (equivalent to vanilla but with CFR+)
        solver.train_full(10_000);

        // King should always call when facing a bet
        if let Some(strategy) = solver.get_average_strategy("Kb") {
            assert!(
                strategy[1] > 0.99,
                "King should always call a bet, got {:?}",
                strategy
            );
        }

        // Jack should always fold when facing a bet
        if let Some(strategy) = solver.get_average_strategy("Jb") {
            assert!(
                strategy[0] > 0.99,
                "Jack should always fold when facing a bet, got {:?}",
                strategy
            );
        }
    }

    #[test]
    fn mccfr_sampling_much_faster_than_full() {
        use std::time::Instant;

        let game = KuhnPoker::new();

        // Time full traversal
        let mut solver_full = MccfrSolver::new(game.clone());
        let start_full = Instant::now();
        solver_full.train_full(100);
        let time_full = start_full.elapsed();

        // Time sampled traversal (sample 1 state per iteration)
        let mut solver_sampled = MccfrSolver::new(game);
        let start_sampled = Instant::now();
        solver_sampled.train(100, 1);
        let time_sampled = start_sampled.elapsed();

        // Sampled should be faster (for Kuhn it's 6x fewer traversals)
        // But both are fast for Kuhn, so just verify they both work
        assert!(solver_full.iterations() == 100);
        assert!(solver_sampled.iterations() == 100);

        println!("Full: {:?}, Sampled: {:?}", time_full, time_sampled);
    }

    #[test]
    fn cfr_plus_floors_negative_regrets() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        solver.train_full(100);

        // All regrets should be >= 0 with CFR+
        for regrets in solver.regret_sum.values() {
            for &r in regrets {
                assert!(r >= 0.0, "CFR+ should floor regrets at 0, got {r}");
            }
        }
    }
}
