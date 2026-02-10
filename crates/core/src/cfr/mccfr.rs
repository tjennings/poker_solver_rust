//! Monte Carlo CFR (MCCFR) - Sampling-based CFR for large games.
//!
//! MCCFR samples initial states (chance outcomes) instead of iterating over all of them,
//! making it practical for games with many initial states like HUNL Preflop (28,561 states).
//!
//! This implementation uses **Chance Sampling**: we sample the initial card deal,
//! then do full CFR traversal on the betting tree for that deal.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::game::{Game, Player};

use super::regret::regret_match;

/// Normalize a strategy sum vector to probabilities.
///
/// Returns `None` if the total is zero (no data accumulated).
fn normalize_strategy(sums: &[f64]) -> Option<Vec<f64>> {
    let total: f64 = sums.iter().sum();
    if total > 0.0 {
        Some(sums.iter().map(|&s| s / total).collect())
    } else {
        None
    }
}

/// Advance xorshift64 RNG state in place.
fn xorshift(state: &mut u64) {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
}

/// Compute the current strategy from a regret table (pure function).
///
/// Looks up the info set in the regret snapshot and applies regret matching.
/// Returns a uniform distribution if the info set has no entry.
fn current_strategy_from(
    regret_snapshot: &FxHashMap<u64, Vec<f64>>,
    info_set: u64,
    num_actions: usize,
) -> Vec<f64> {
    if let Some(regrets) = regret_snapshot.get(&info_set) {
        regret_match(regrets)
    } else {
        #[allow(clippy::cast_precision_loss)]
        let uniform = 1.0 / num_actions as f64;
        vec![uniform; num_actions]
    }
}

/// Sample an action index from a strategy using the given RNG state.
fn sample_action_rng(rng_state: &mut u64, strategy: &[f64]) -> usize {
    xorshift(rng_state);

    #[allow(clippy::cast_precision_loss)]
    let r = (*rng_state as f64) / (u64::MAX as f64);

    let mut cumulative = 0.0;
    for (i, &prob) in strategy.iter().enumerate() {
        cumulative += prob;
        if r < cumulative {
            return i;
        }
    }

    strategy.len() - 1
}

/// Derive a deterministic per-sample seed using splitmix64 mixing.
fn per_sample_seed(base_seed: u64, iteration: u64, sample_idx: usize) -> u64 {
    let mut z = base_seed
        .wrapping_add(iteration.wrapping_mul(0x9E37_79B9_7F4A_7C15))
        .wrapping_add(sample_idx as u64);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    if z == 0 { 1 } else { z }
}

/// Monte Carlo CFR solver with chance sampling.
///
/// Much faster than vanilla CFR for games with many initial states because
/// it samples states instead of iterating over all of them.
pub struct MccfrSolver<G: Game> {
    game: G,
    /// Cumulative regrets per info set
    regret_sum: FxHashMap<u64, Vec<f64>>,
    /// Cumulative strategy sums (for averaging)
    strategy_sum: FxHashMap<u64, Vec<f64>>,
    /// Use CFR+ (floor regrets at 0)
    use_cfr_plus: bool,
    /// RNG state for sampling
    rng_state: u64,
    /// Total iterations completed
    iterations: u64,
    /// Number of early iterations to discount (CFR+ optimization).
    /// Iterations before this threshold use sqrt(t)/(sqrt(t)+1) weighting.
    discount_iterations: u64,
    /// Skip strategy accumulation until this iteration count.
    /// Implements the paper's recommendation to ignore the first ~50% of
    /// iterations for average strategy computation.
    skip_strategy_until: u64,
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
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
            use_cfr_plus: true,
            rng_state: 0x1234_5678_9ABC_DEF0,
            iterations: 0,
            discount_iterations: 30,
            skip_strategy_until: 0,
        }
    }

    /// Creates a new MCCFR solver with custom configuration.
    #[must_use]
    pub fn with_config(game: G, config: &MccfrConfig) -> Self {
        Self {
            game,
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
            use_cfr_plus: config.use_cfr_plus,
            rng_state: 0x1234_5678_9ABC_DEF0,
            iterations: 0,
            discount_iterations: config.discount_iterations.unwrap_or(0),
            skip_strategy_until: config.skip_first_iterations.unwrap_or(0),
        }
    }

    /// Set RNG seed for reproducible sampling.
    pub fn set_seed(&mut self, seed: u64) {
        self.rng_state = if seed == 0 { 1 } else { seed };
    }

    /// Train using chance sampling MCCFR.
    ///
    /// Each iteration samples `samples_per_iter` initial states and runs
    /// CFR on each, rather than iterating over all initial states.
    pub fn train(&mut self, iterations: u64, samples_per_iter: usize) {
        self.train_with_callback(iterations, samples_per_iter, |_| {});
    }

    /// Train with a per-iteration callback for progress reporting.
    ///
    /// Same as [`train`] but calls `on_iteration` after each iteration
    /// with the number of iterations completed so far in this call.
    /// Initial states are generated once and reused across all iterations.
    #[allow(clippy::missing_panics_doc)]
    pub fn train_with_callback<F>(
        &mut self,
        iterations: u64,
        samples_per_iter: usize,
        mut on_iteration: F,
    ) where
        F: FnMut(u64),
    {
        let initial_states = self.game.initial_states();
        let num_states = initial_states.len();

        if num_states == 0 {
            return;
        }

        let skip_until = self.skip_strategy_until;

        for i in 0..iterations {
            let discount = self.compute_discount();
            let accumulate = self.iterations >= skip_until;
            let player = self.traversing_player();

            for _ in 0..samples_per_iter {
                xorshift(&mut self.rng_state);

                #[allow(clippy::cast_possible_truncation)]
                let state_idx = (self.rng_state % num_states as u64) as usize;
                let state = &initial_states[state_idx];

                #[allow(clippy::cast_precision_loss)]
                let sample_weight = num_states as f64 / samples_per_iter as f64;

                self.cfr_traverse(
                    state,
                    player,
                    1.0,
                    1.0,
                    sample_weight,
                    discount,
                    accumulate,
                );
            }
            self.iterations += 1;
            on_iteration(i + 1);
        }
    }

    /// Train for a fixed number of iterations, sampling all states once per iteration.
    ///
    /// This is equivalent to vanilla CFR but can be faster due to CFR+ optimizations.
    pub fn train_full(&mut self, iterations: u64) {
        let initial_states = self.game.initial_states();
        let skip_until = self.skip_strategy_until;

        for _ in 0..iterations {
            let discount = self.compute_discount();
            let accumulate = self.iterations >= skip_until;
            let player = self.traversing_player();

            for state in &initial_states {
                self.cfr_traverse(state, player, 1.0, 1.0, 1.0, discount, accumulate);
            }

            self.iterations += 1;
        }
    }

    /// Returns the traversing player for this iteration (alternating P1/P2).
    fn traversing_player(&self) -> Player {
        if self.iterations.is_multiple_of(2) {
            Player::Player1
        } else {
            Player::Player2
        }
    }

    /// Compute discount factor for early iterations (CFR+ optimization).
    fn compute_discount(&self) -> f64 {
        if self.iterations < self.discount_iterations {
            #[allow(clippy::cast_precision_loss)]
            let t = (self.iterations + 1) as f64;
            t.sqrt() / (t.sqrt() + 1.0)
        } else {
            1.0
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

    /// Returns the current regret-matched strategy for an information set.
    #[must_use]
    pub fn get_current_strategy(&self, info_set: u64, num_actions: usize) -> Vec<f64> {
        current_strategy_from(&self.regret_sum, info_set, num_actions)
    }

    /// Returns the number of iterations completed.
    #[must_use]
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// Returns all info sets and their average strategies.
    #[must_use]
    pub fn all_strategies(&self) -> FxHashMap<u64, Vec<f64>> {
        self.strategy_sum
            .keys()
            .filter_map(|&k| self.get_average_strategy(k).map(|s| (k, s)))
            .collect()
    }

    /// Returns all info sets with best-available strategies.
    ///
    /// Uses the average strategy when available (post-skip-threshold),
    /// otherwise falls back to the current regret-matched strategy.
    /// This ensures checkpoints during the skip phase still show data.
    #[must_use]
    pub fn all_strategies_best_effort(&self) -> FxHashMap<u64, Vec<f64>> {
        let mut result = FxHashMap::default();

        // Average strategies (highest quality, only available after skip threshold)
        for (&k, s) in &self.strategy_sum {
            if let Some(avg) = normalize_strategy(s) {
                result.insert(k, avg);
            }
        }

        // Fill gaps with regret-matched strategies
        for (&k, regrets) in &self.regret_sum {
            result.entry(k).or_insert_with(|| regret_match(regrets));
        }

        result
    }

    /// External sampling MCCFR traversal.
    ///
    /// At the traversing player's nodes: explore ALL actions (full width).
    /// At the opponent's nodes: SAMPLE ONE action according to current strategy.
    /// This makes traversal linear in the opponent's branching factor.
    #[allow(clippy::too_many_arguments)]
    fn cfr_traverse(
        &mut self,
        state: &G::State,
        traversing_player: Player,
        p1_reach: f64,
        p2_reach: f64,
        sample_weight: f64,
        discount: f64,
        accumulate_strategy: bool,
    ) -> f64 {
        if self.game.is_terminal(state) {
            return self.game.utility(state, traversing_player);
        }

        let current_player = self.game.player(state);
        let actions = self.game.actions(state);
        let num_actions = actions.len();
        let info_set = self.game.info_set_key(state);

        // Get current strategy from regrets
        let strategy = current_strategy_from(&self.regret_sum, info_set, num_actions);

        if current_player == traversing_player {
            // Traversing player: explore all actions
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
                    accumulate_strategy,
                );
            }

            // Expected utility under current strategy
            let node_util: f64 = action_utils
                .iter()
                .zip(strategy.iter())
                .map(|(u, p)| u * p)
                .sum();

            // Update regrets
            let opponent_reach = match current_player {
                Player::Player1 => p2_reach,
                Player::Player2 => p1_reach,
            };

            let regrets = self
                .regret_sum
                .entry(info_set)
                .or_insert_with(|| vec![0.0; num_actions]);

            for i in 0..num_actions {
                let regret_delta =
                    opponent_reach * (action_utils[i] - node_util) * sample_weight * discount;
                regrets[i] += regret_delta;

                if self.use_cfr_plus && regrets[i] < 0.0 {
                    regrets[i] = 0.0;
                }
            }

            // Accumulate strategy after skip threshold
            if accumulate_strategy {
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
        } else {
            // Opponent's node: sample ONE action according to strategy
            let sampled_action = sample_action_rng(&mut self.rng_state, &strategy);
            let action = actions[sampled_action];
            let next_state = self.game.next_state(state, action);

            let (new_p1_reach, new_p2_reach) = match current_player {
                Player::Player1 => (p1_reach * strategy[sampled_action], p2_reach),
                Player::Player2 => (p1_reach, p2_reach * strategy[sampled_action]),
            };

            // Accumulate opponent's strategy
            if accumulate_strategy {
                let opp_reach = match current_player {
                    Player::Player1 => p1_reach,
                    Player::Player2 => p2_reach,
                };

                let strat_sums = self
                    .strategy_sum
                    .entry(info_set)
                    .or_insert_with(|| vec![0.0; num_actions]);

                for i in 0..num_actions {
                    strat_sums[i] += opp_reach * strategy[i] * sample_weight * discount;
                }
            }

            self.cfr_traverse(
                &next_state,
                traversing_player,
                new_p1_reach,
                new_p2_reach,
                sample_weight,
                discount,
                accumulate_strategy,
            )
        }
    }

    /// Train using parallel chance sampling MCCFR.
    ///
    /// Samples within each iteration are distributed across rayon threads.
    /// Each thread accumulates deltas into a thread-local map, then merges.
    /// Results are NOT bit-identical to sequential due to float addition order.
    pub fn train_parallel(&mut self, iterations: u64, samples_per_iter: usize) {
        self.train_parallel_with_callback(iterations, samples_per_iter, |_| {});
    }

    /// Parallel training with a per-iteration callback for progress reporting.
    #[allow(clippy::missing_panics_doc)]
    pub fn train_parallel_with_callback<F>(
        &mut self,
        iterations: u64,
        samples_per_iter: usize,
        mut on_iteration: F,
    ) where
        F: FnMut(u64),
    {
        let initial_states = self.game.initial_states();
        let num_states = initial_states.len();

        if num_states == 0 {
            return;
        }

        let skip_until = self.skip_strategy_until;

        for i in 0..iterations {
            let discount = self.compute_discount();
            let accumulate = self.iterations >= skip_until;
            let player = self.traversing_player();
            let base_seed = self.rng_state;

            #[allow(clippy::cast_precision_loss)]
            let sample_weight = num_states as f64 / samples_per_iter as f64;

            let merged = {
                let regret_snapshot = &self.regret_sum;
                let game = &self.game;
                let states = &initial_states;
                let iter_num = self.iterations;

                (0..samples_per_iter)
                    .into_par_iter()
                    .fold(
                        TraversalAccumulator::new,
                        |mut acc, idx| {
                            acc.rng_state = per_sample_seed(base_seed, iter_num, idx);

                            #[allow(clippy::cast_possible_truncation)]
                            let state_idx =
                                (per_sample_seed(base_seed, iter_num, idx) % num_states as u64)
                                    as usize;
                            let state = &states[state_idx];

                            cfr_traverse_pure(
                                game,
                                regret_snapshot,
                                &mut acc,
                                state,
                                player,
                                1.0,
                                1.0,
                                sample_weight,
                                discount,
                                accumulate,
                            );
                            acc
                        },
                    )
                    .reduce(TraversalAccumulator::new, TraversalAccumulator::merge)
            };

            self.merge_accumulator(merged);

            // Advance main RNG so seed changes per iteration
            xorshift(&mut self.rng_state);
            self.iterations += 1;
            on_iteration(i + 1);
        }
    }

    /// Parallel train for a fixed number of iterations, visiting all states.
    pub fn train_full_parallel(&mut self, iterations: u64) {
        let initial_states = self.game.initial_states();
        let skip_until = self.skip_strategy_until;

        for _ in 0..iterations {
            let discount = self.compute_discount();
            let accumulate = self.iterations >= skip_until;
            let player = self.traversing_player();

            let merged = {
                let regret_snapshot = &self.regret_sum;
                let game = &self.game;
                let states = &initial_states;
                let iter_num = self.iterations;

                states
                    .par_iter()
                    .enumerate()
                    .fold(
                        TraversalAccumulator::new,
                        |mut acc, (idx, state)| {
                            acc.rng_state = per_sample_seed(0xCAFE, iter_num, idx);

                            cfr_traverse_pure(
                                game,
                                regret_snapshot,
                                &mut acc,
                                state,
                                player,
                                1.0,
                                1.0,
                                1.0,
                                discount,
                                accumulate,
                            );
                            acc
                        },
                    )
                    .reduce(TraversalAccumulator::new, TraversalAccumulator::merge)
            };

            self.merge_accumulator(merged);
            self.iterations += 1;
        }
    }

    /// Merge a `TraversalAccumulator` into the solver's main tables.
    ///
    /// Applies CFR+ flooring (clamp to 0) on regrets during merge.
    fn merge_accumulator(&mut self, acc: TraversalAccumulator) {
        for (key, deltas) in acc.regret_deltas {
            let regrets = self
                .regret_sum
                .entry(key)
                .or_insert_with(|| vec![0.0; deltas.len()]);
            for (i, d) in deltas.iter().enumerate() {
                regrets[i] += d;
                if self.use_cfr_plus && regrets[i] < 0.0 {
                    regrets[i] = 0.0;
                }
            }
        }

        for (key, deltas) in acc.strategy_deltas {
            let strats = self
                .strategy_sum
                .entry(key)
                .or_insert_with(|| vec![0.0; deltas.len()]);
            for (i, d) in deltas.iter().enumerate() {
                strats[i] += d;
            }
        }
    }
}

/// Thread-local accumulator for parallel MCCFR traversals.
///
/// Collects regret and strategy deltas without CFR+ flooring.
/// Merged via tree reduction after all samples complete.
struct TraversalAccumulator {
    regret_deltas: FxHashMap<u64, Vec<f64>>,
    strategy_deltas: FxHashMap<u64, Vec<f64>>,
    rng_state: u64,
}

impl TraversalAccumulator {
    fn new() -> Self {
        Self {
            regret_deltas: FxHashMap::default(),
            strategy_deltas: FxHashMap::default(),
            rng_state: 0x1234_5678_9ABC_DEF0,
        }
    }

    /// Merge two accumulators by element-wise addition of delta vectors.
    fn merge(mut self, other: Self) -> Self {
        for (key, other_vec) in other.regret_deltas {
            let entry = self
                .regret_deltas
                .entry(key)
                .or_insert_with(|| vec![0.0; other_vec.len()]);
            for (i, &v) in other_vec.iter().enumerate() {
                entry[i] += v;
            }
        }

        for (key, other_vec) in other.strategy_deltas {
            let entry = self
                .strategy_deltas
                .entry(key)
                .or_insert_with(|| vec![0.0; other_vec.len()]);
            for (i, &v) in other_vec.iter().enumerate() {
                entry[i] += v;
            }
        }

        self
    }
}

/// Pure MCCFR traversal that writes deltas to a `TraversalAccumulator`.
///
/// Same logic as `cfr_traverse` but operates on immutable regret snapshot
/// and accumulates deltas without applying CFR+ flooring.
#[allow(clippy::too_many_arguments)]
fn cfr_traverse_pure<G: Game>(
    game: &G,
    regret_snapshot: &FxHashMap<u64, Vec<f64>>,
    acc: &mut TraversalAccumulator,
    state: &G::State,
    traversing_player: Player,
    p1_reach: f64,
    p2_reach: f64,
    sample_weight: f64,
    discount: f64,
    accumulate_strategy: bool,
) -> f64 {
    if game.is_terminal(state) {
        return game.utility(state, traversing_player);
    }

    let current_player = game.player(state);
    let actions = game.actions(state);
    let num_actions = actions.len();
    let info_set = game.info_set_key(state);

    let strategy = current_strategy_from(regret_snapshot, info_set, num_actions);

    if current_player == traversing_player {
        let mut action_utils = vec![0.0; num_actions];

        for (i, action) in actions.iter().enumerate() {
            let next_state = game.next_state(state, *action);

            let (new_p1, new_p2) = match current_player {
                Player::Player1 => (p1_reach * strategy[i], p2_reach),
                Player::Player2 => (p1_reach, p2_reach * strategy[i]),
            };

            action_utils[i] = cfr_traverse_pure(
                game,
                regret_snapshot,
                acc,
                &next_state,
                traversing_player,
                new_p1,
                new_p2,
                sample_weight,
                discount,
                accumulate_strategy,
            );
        }

        let node_util: f64 = action_utils
            .iter()
            .zip(strategy.iter())
            .map(|(u, p)| u * p)
            .sum();

        let opponent_reach = match current_player {
            Player::Player1 => p2_reach,
            Player::Player2 => p1_reach,
        };

        let regrets = acc
            .regret_deltas
            .entry(info_set)
            .or_insert_with(|| vec![0.0; num_actions]);

        for i in 0..num_actions {
            regrets[i] +=
                opponent_reach * (action_utils[i] - node_util) * sample_weight * discount;
        }

        if accumulate_strategy {
            let my_reach = match current_player {
                Player::Player1 => p1_reach,
                Player::Player2 => p2_reach,
            };

            let strat_sums = acc
                .strategy_deltas
                .entry(info_set)
                .or_insert_with(|| vec![0.0; num_actions]);

            for i in 0..num_actions {
                strat_sums[i] += my_reach * strategy[i] * sample_weight * discount;
            }
        }

        node_util
    } else {
        let sampled_action = sample_action_rng(&mut acc.rng_state, &strategy);
        let action = actions[sampled_action];
        let next_state = game.next_state(state, action);

        let (new_p1, new_p2) = match current_player {
            Player::Player1 => (p1_reach * strategy[sampled_action], p2_reach),
            Player::Player2 => (p1_reach, p2_reach * strategy[sampled_action]),
        };

        if accumulate_strategy {
            let opp_reach = match current_player {
                Player::Player1 => p1_reach,
                Player::Player2 => p2_reach,
            };

            let strat_sums = acc
                .strategy_deltas
                .entry(info_set)
                .or_insert_with(|| vec![0.0; num_actions]);

            for i in 0..num_actions {
                strat_sums[i] += opp_reach * strategy[i] * sample_weight * discount;
            }
        }

        cfr_traverse_pure(
            game,
            regret_snapshot,
            acc,
            &next_state,
            traversing_player,
            new_p1,
            new_p2,
            sample_weight,
            discount,
            accumulate_strategy,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::KuhnPoker;
    use test_macros::timed_test;

    #[timed_test]
    fn mccfr_initializes_empty() {
        let game = KuhnPoker::new();
        let solver = MccfrSolver::new(game);

        assert!(solver.regret_sum.is_empty());
        assert!(solver.strategy_sum.is_empty());
        assert_eq!(solver.iterations(), 0);
    }

    #[timed_test]
    fn mccfr_training_populates_info_sets() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        // With 6 initial states in Kuhn, sampling 10 per iter should cover all
        solver.train(10, 10);

        assert!(!solver.strategy_sum.is_empty());
        assert!(solver.iterations() > 0);
    }

    #[timed_test]
    fn mccfr_strategy_sums_to_one() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        solver.train(100, 10);

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
    fn mccfr_converges_on_kuhn() {
        use crate::info_key::InfoKey;

        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        // Train with full sampling (equivalent to vanilla but with CFR+)
        solver.train_full(10_000);

        // King (2) facing bet (4) → "Kb"
        let kb = InfoKey::new(2, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(kb) {
            assert!(
                strategy[1] > 0.99,
                "King should always call a bet, got {strategy:?}"
            );
        }

        // Jack (0) facing bet (4) → "Jb"
        let jb = InfoKey::new(0, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(jb) {
            assert!(
                strategy[0] > 0.99,
                "Jack should always fold when facing a bet, got {strategy:?}"
            );
        }
    }

    #[timed_test]
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
        assert!(time_full > time_sampled);

        println!("Full: {time_full:?}, Sampled: {time_sampled:?}");
    }

    #[timed_test]
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

    #[timed_test]
    fn set_seed_produces_deterministic_results() {
        let game = KuhnPoker::new();

        let mut solver1 = MccfrSolver::new(game.clone());
        solver1.set_seed(42);
        solver1.train(50, 3);
        let strats1 = solver1.all_strategies();

        let mut solver2 = MccfrSolver::new(game);
        solver2.set_seed(42);
        solver2.train(50, 3);
        let strats2 = solver2.all_strategies();

        // Same seed → same strategies
        assert_eq!(strats1.len(), strats2.len());
        for (key, probs1) in &strats1 {
            let probs2 = strats2.get(key).expect("same info sets");
            for (p1, p2) in probs1.iter().zip(probs2.iter()) {
                assert!(
                    (p1 - p2).abs() < 1e-10,
                    "Strategies should be identical with same seed: {p1} vs {p2}"
                );
            }
        }
    }

    #[timed_test]
    fn set_seed_zero_becomes_one() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);
        solver.set_seed(0);
        assert_eq!(solver.rng_state, 1, "Seed 0 should be mapped to 1");
    }

    #[timed_test]
    fn skip_first_iterations_delays_accumulation() {
        let game = KuhnPoker::new();

        // With skip=50, the first 50 iterations should NOT accumulate strategy
        let config = MccfrConfig {
            use_cfr_plus: true,
            skip_first_iterations: Some(50),
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game.clone(), &config);
        solver.train_full(49);

        // After 49 iterations (all < 50), strategy_sum should be empty
        assert!(
            solver.strategy_sum.is_empty(),
            "Strategy sum should be empty when all iterations are skipped, got {} entries",
            solver.strategy_sum.len()
        );

        // Train 1 more iteration (iter 49 is >= 50? No, 0-indexed: iter 49 is the 50th)
        // After 50 total iterations, the 50th call (iteration index 49) should NOT accumulate
        // because iterations is checked BEFORE incrementing
        // Let's train to 51 to be sure we get at least one accumulation
        solver.train_full(2);

        // Now we should have accumulated for iteration 50
        assert!(
            !solver.strategy_sum.is_empty(),
            "Strategy sum should be non-empty after passing skip threshold"
        );
    }

    #[timed_test]
    fn discount_iterations_from_config() {
        let config = MccfrConfig {
            discount_iterations: Some(100),
            use_cfr_plus: true,
            ..MccfrConfig::default()
        };
        let game = KuhnPoker::new();
        let solver = MccfrSolver::with_config(game, &config);

        assert_eq!(solver.discount_iterations, 100);
    }

    #[timed_test]
    fn discount_iterations_none_means_zero() {
        let config = MccfrConfig {
            discount_iterations: None,
            ..MccfrConfig::default()
        };
        let game = KuhnPoker::new();
        let solver = MccfrSolver::with_config(game, &config);

        assert_eq!(solver.discount_iterations, 0);
    }

    // --- Parallel MCCFR tests ---

    #[timed_test]
    fn accumulator_merge_is_additive() {
        let mut a = TraversalAccumulator::new();
        a.regret_deltas.insert(1, vec![1.0, 2.0]);
        a.strategy_deltas.insert(1, vec![0.5, 0.5]);

        let mut b = TraversalAccumulator::new();
        b.regret_deltas.insert(1, vec![3.0, -1.0]);
        b.regret_deltas.insert(2, vec![10.0]);
        b.strategy_deltas.insert(1, vec![0.3, 0.7]);

        let merged = a.merge(b);

        let r1 = merged.regret_deltas.get(&1).expect("key 1");
        assert!((r1[0] - 4.0).abs() < 1e-10);
        assert!((r1[1] - 1.0).abs() < 1e-10);

        let r2 = merged.regret_deltas.get(&2).expect("key 2");
        assert!((r2[0] - 10.0).abs() < 1e-10);

        let s1 = merged.strategy_deltas.get(&1).expect("key 1");
        assert!((s1[0] - 0.8).abs() < 1e-10);
        assert!((s1[1] - 1.2).abs() < 1e-10);
    }

    #[timed_test]
    fn parallel_training_populates_info_sets() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        solver.train_parallel(10, 10);

        assert!(!solver.strategy_sum.is_empty());
        assert!(solver.iterations() > 0);
    }

    #[timed_test]
    fn parallel_strategy_sums_to_one() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        solver.train_parallel(100, 10);

        for &info_set in solver.strategy_sum.keys() {
            if let Some(strategy) = solver.get_average_strategy(info_set) {
                let sum: f64 = strategy.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Parallel strategy for {info_set} sums to {sum}"
                );
            }
        }
    }

    #[timed_test]
    fn parallel_converges_on_kuhn() {
        use crate::info_key::InfoKey;

        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        solver.train_full_parallel(10_000);

        // King facing bet → always call
        let kb = InfoKey::new(2, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(kb) {
            assert!(
                strategy[1] > 0.99,
                "Parallel: King should always call a bet, got {strategy:?}"
            );
        }

        // Jack facing bet → always fold
        let jb = InfoKey::new(0, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(jb) {
            assert!(
                strategy[0] > 0.99,
                "Parallel: Jack should always fold facing a bet, got {strategy:?}"
            );
        }
    }

    #[timed_test]
    fn parallel_cfr_plus_floors_negative_regrets() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        solver.train_full_parallel(100);

        for regrets in solver.regret_sum.values() {
            for &r in regrets {
                assert!(
                    r >= 0.0,
                    "Parallel CFR+ should floor regrets at 0, got {r}"
                );
            }
        }
    }

    #[timed_test]
    fn parallel_and_sequential_converge_similarly() {
        use crate::info_key::InfoKey;

        let game = KuhnPoker::new();

        let mut seq = MccfrSolver::new(game.clone());
        seq.train_full(5_000);

        let mut par = MccfrSolver::new(game);
        par.train_full_parallel(5_000);

        // Both should converge to same Nash equilibrium points
        let kb = InfoKey::new(2, 0, 0, 0, &[4]).as_u64();
        let jb = InfoKey::new(0, 0, 0, 0, &[4]).as_u64();

        let seq_kb = seq.get_average_strategy(kb).expect("seq kb");
        let par_kb = par.get_average_strategy(kb).expect("par kb");
        assert!(
            (seq_kb[1] - par_kb[1]).abs() < 0.05,
            "King-call should converge similarly: seq={:.4} par={:.4}",
            seq_kb[1],
            par_kb[1]
        );

        let seq_jb = seq.get_average_strategy(jb).expect("seq jb");
        let par_jb = par.get_average_strategy(jb).expect("par jb");
        assert!(
            (seq_jb[0] - par_jb[0]).abs() < 0.05,
            "Jack-fold should converge similarly: seq={:.4} par={:.4}",
            seq_jb[0],
            par_jb[0]
        );
    }
}
