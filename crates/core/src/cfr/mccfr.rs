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
    /// Cached deal pool (generated once, reused across train calls)
    cached_initial_states: Option<Vec<G::State>>,
    /// RNG state for sampling
    rng_state: u64,
    /// Total iterations completed
    iterations: u64,
    /// DCFR positive regret discount exponent
    dcfr_alpha: f64,
    /// DCFR negative regret discount exponent
    dcfr_beta: f64,
    /// DCFR strategy sum discount exponent
    dcfr_gamma: f64,
    /// Regret-based pruning: skip actions below the pruning threshold.
    pruning: bool,
    /// Absolute iteration count before pruning activates.
    pruning_warmup: u64,
    /// Run a full un-pruned probe iteration every N iterations.
    pruning_probe_interval: u64,
    /// Regret threshold below which actions are pruned.
    /// With DCFR, use a negative value (e.g. -5.0) so that DCFR's
    /// decay can bring regrets back above the threshold between probes.
    pruning_threshold: f64,
    /// Number of subtree traversals skipped by pruning.
    pruned_traversals: u64,
    /// Total action traversals attempted (pruned + executed).
    total_traversals: u64,
}

/// Configuration for MCCFR training with DCFR discounting.
///
/// Implements Discounted CFR (Brown & Sandholm 2019):
/// - Positive regrets discounted by `t^α / (t^α + 1)` after each iteration
/// - Negative regrets discounted by `t^β / (t^β + 1)` after each iteration
/// - Strategy contributions weighted by `(t / (t+1))^γ` per iteration
#[derive(Debug, Clone)]
pub struct MccfrConfig {
    /// Number of initial states to sample per iteration
    pub samples_per_iteration: usize,
    /// DCFR positive regret discount exponent (default 1.5)
    pub dcfr_alpha: f64,
    /// DCFR negative regret discount exponent (default 0.5)
    pub dcfr_beta: f64,
    /// DCFR strategy sum discount exponent (default 2.0)
    pub dcfr_gamma: f64,
    /// Enable regret-based pruning (skip actions below the threshold).
    pub pruning: bool,
    /// Absolute iteration count to complete before enabling pruning.
    /// Caller should compute from a warmup fraction of total iterations.
    pub pruning_warmup: u64,
    /// Run a full un-pruned probe iteration every N iterations.
    pub pruning_probe_interval: u64,
    /// Regret threshold below which actions are pruned (default 0.0).
    /// With DCFR, use a negative value so that decay can bring regrets
    /// back above this line between probes. Recommended: -5.0 to -20.0.
    pub pruning_threshold: f64,
}

impl Default for MccfrConfig {
    fn default() -> Self {
        Self {
            samples_per_iteration: 100,
            dcfr_alpha: 1.5,
            dcfr_beta: 0.5,
            dcfr_gamma: 2.0,
            pruning: false,
            pruning_warmup: 0,
            pruning_probe_interval: 20,
            pruning_threshold: 0.0,
        }
    }
}

impl<G: Game> MccfrSolver<G> {
    /// Creates a new MCCFR solver for the given game with default DCFR parameters.
    #[must_use]
    pub fn new(game: G) -> Self {
        Self::with_config(game, &MccfrConfig::default())
    }

    /// Creates a new MCCFR solver with custom configuration.
    #[must_use]
    pub fn with_config(game: G, config: &MccfrConfig) -> Self {
        Self {
            game,
            regret_sum: FxHashMap::default(),
            strategy_sum: FxHashMap::default(),
            cached_initial_states: None,
            rng_state: 0x1234_5678_9ABC_DEF0,
            iterations: 0,
            dcfr_alpha: config.dcfr_alpha,
            dcfr_beta: config.dcfr_beta,
            dcfr_gamma: config.dcfr_gamma,
            pruning: config.pruning,
            pruning_warmup: config.pruning_warmup,
            pruning_probe_interval: config.pruning_probe_interval.max(1),
            pruning_threshold: config.pruning_threshold,
            pruned_traversals: 0,
            total_traversals: 0,
        }
    }

    /// Ensures the deal pool is generated and cached. No-op after the first call.
    fn ensure_deals_cached(&mut self) {
        if self.cached_initial_states.is_none() {
            self.cached_initial_states = Some(self.game.initial_states());
        }
    }

    /// Takes the cached deal pool out of `self`, avoiding borrow conflicts
    /// with `&mut self` methods like `cfr_traverse`. Caller must put it back
    /// via `self.cached_initial_states = Some(states)` when done.
    fn take_deals(&mut self) -> Vec<G::State> {
        self.cached_initial_states
            .take()
            .expect("ensure_deals_cached must be called first")
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
        self.ensure_deals_cached();
        let initial_states = self.take_deals();
        let num_states = initial_states.len();

        if num_states == 0 {
            self.cached_initial_states = Some(initial_states);
            return;
        }

        for i in 0..iterations {
            let strategy_discount = self.compute_strategy_discount();
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
                    strategy_discount,
                );
            }
            self.discount_regrets();
            self.iterations += 1;
            on_iteration(i + 1);
        }
        self.cached_initial_states = Some(initial_states);
    }

    /// Train for a fixed number of iterations, sampling all states once per iteration.
    pub fn train_full(&mut self, iterations: u64) {
        self.ensure_deals_cached();
        let initial_states = self.take_deals();

        for _ in 0..iterations {
            let strategy_discount = self.compute_strategy_discount();
            let player = self.traversing_player();

            for state in &initial_states {
                self.cfr_traverse(state, player, 1.0, 1.0, 1.0, strategy_discount);
            }

            self.discount_regrets();
            self.iterations += 1;
        }
        self.cached_initial_states = Some(initial_states);
    }

    /// Returns the traversing player for this iteration (alternating P1/P2).
    fn traversing_player(&self) -> Player {
        if self.iterations.is_multiple_of(2) {
            Player::Player1
        } else {
            Player::Player2
        }
    }

    /// Compute DCFR strategy discount: `(t / (t + 1))^γ`.
    ///
    /// Down-weights older strategy contributions so that early, noisy
    /// iterations contribute less to the average strategy.
    fn compute_strategy_discount(&self) -> f64 {
        #[allow(clippy::cast_precision_loss)]
        let t = self.iterations as f64;
        let ratio = t / (t + 1.0);
        ratio.powf(self.dcfr_gamma)
    }

    /// Apply DCFR cumulative regret discounting after each iteration.
    ///
    /// Positive regrets multiplied by `t^α / (t^α + 1)`.
    /// Negative regrets multiplied by `t^β / (t^β + 1)`.
    /// With α > β, negative regrets decay faster, focusing on promising actions.
    fn discount_regrets(&mut self) {
        #[allow(clippy::cast_precision_loss)]
        let t = (self.iterations + 1) as f64;
        let pos_factor = t.powf(self.dcfr_alpha) / (t.powf(self.dcfr_alpha) + 1.0);
        let neg_factor = t.powf(self.dcfr_beta) / (t.powf(self.dcfr_beta) + 1.0);

        for regrets in self.regret_sum.values_mut() {
            for r in regrets.iter_mut() {
                if *r > 0.0 {
                    *r *= pos_factor;
                } else if *r < 0.0 {
                    *r *= neg_factor;
                }
            }
        }
    }

    /// Returns pruning statistics: `(pruned, total)` traversals.
    #[must_use]
    pub fn pruning_stats(&self) -> (u64, u64) {
        (self.pruned_traversals, self.total_traversals)
    }

    /// Check whether action `action_idx` at `info_set` should be pruned.
    ///
    /// Prunes only when: pruning is enabled, past warmup, not a probe iteration,
    /// the action's regret is below the threshold, and at least one sibling has
    /// positive regret. With DCFR, a negative threshold allows decayed regrets
    /// to cross back above the line between probes.
    fn should_prune(&self, info_set: u64, action_idx: usize) -> bool {
        if !self.pruning {
            return false;
        }
        if self.iterations < self.pruning_warmup {
            return false;
        }
        if self.iterations.is_multiple_of(self.pruning_probe_interval) {
            return false; // probe iteration — explore everything
        }
        match self.regret_sum.get(&info_set) {
            Some(regrets) => {
                // Only prune if this action is below threshold AND at least one
                // sibling has positive regret. If ALL regrets are below threshold,
                // regret matching returns uniform — must explore all.
                let has_positive = regrets.iter().any(|&r| r > 0.0);
                has_positive && regrets[action_idx] <= self.pruning_threshold
            }
            None => false, // No regret data yet — don't prune
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

    /// Returns a reference to the cumulative regret sums.
    #[must_use]
    pub fn regret_sum(&self) -> &FxHashMap<u64, Vec<f64>> {
        &self.regret_sum
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
    fn cfr_traverse(
        &mut self,
        state: &G::State,
        traversing_player: Player,
        p1_reach: f64,
        p2_reach: f64,
        sample_weight: f64,
        strategy_discount: f64,
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
            let mut pruned: u32 = 0;

            for (i, action) in actions.iter().enumerate() {
                self.total_traversals += 1;

                // Regret-based pruning: skip actions with non-positive cumulative regret
                if self.should_prune(info_set, i) {
                    self.pruned_traversals += 1;
                    pruned |= 1 << i;
                    continue;
                }

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
                    strategy_discount,
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
                if pruned >> i & 1 != 0 {
                    continue;
                }
                let regret_delta =
                    opponent_reach * (action_utils[i] - node_util) * sample_weight;
                regrets[i] += regret_delta;
            }

            // Accumulate strategy with DCFR discounting
            if strategy_discount > 0.0 {
                let my_reach = match current_player {
                    Player::Player1 => p1_reach,
                    Player::Player2 => p2_reach,
                };

                let strat_sums = self
                    .strategy_sum
                    .entry(info_set)
                    .or_insert_with(|| vec![0.0; num_actions]);

                for i in 0..num_actions {
                    strat_sums[i] += my_reach * strategy[i] * sample_weight * strategy_discount;
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

            // Opponent strategy is NOT accumulated here — it's accumulated
            // when the opponent is the traversing player on alternating iterations.
            // Accumulating here would double-count with biased reach weights.

            self.cfr_traverse(
                &next_state,
                traversing_player,
                new_p1_reach,
                new_p2_reach,
                sample_weight,
                strategy_discount,
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
        self.ensure_deals_cached();
        let initial_states = self.take_deals();
        let num_states = initial_states.len();

        if num_states == 0 {
            self.cached_initial_states = Some(initial_states);
            return;
        }

        for i in 0..iterations {
            let strategy_discount = self.compute_strategy_discount();
            let player = self.traversing_player();
            let base_seed = self.rng_state;

            let pruning_ctx = PruningCtx {
                enabled: self.pruning,
                warmup: self.pruning_warmup,
                probe_interval: self.pruning_probe_interval,
                threshold: self.pruning_threshold,
                iteration: self.iterations,
            };

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
                                strategy_discount,
                                pruning_ctx,
                            );
                            acc
                        },
                    )
                    .reduce(TraversalAccumulator::new, TraversalAccumulator::merge)
            };

            self.merge_accumulator(merged);
            self.discount_regrets();

            // Advance main RNG so seed changes per iteration
            xorshift(&mut self.rng_state);
            self.iterations += 1;
            on_iteration(i + 1);
        }
        self.cached_initial_states = Some(initial_states);
    }

    /// Parallel train for a fixed number of iterations, visiting all states.
    pub fn train_full_parallel(&mut self, iterations: u64) {
        self.ensure_deals_cached();
        let initial_states = self.take_deals();

        for _ in 0..iterations {
            let strategy_discount = self.compute_strategy_discount();
            let player = self.traversing_player();

            let pruning_ctx = PruningCtx {
                enabled: self.pruning,
                warmup: self.pruning_warmup,
                probe_interval: self.pruning_probe_interval,
                threshold: self.pruning_threshold,
                iteration: self.iterations,
            };

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
                                strategy_discount,
                                pruning_ctx,
                            );
                            acc
                        },
                    )
                    .reduce(TraversalAccumulator::new, TraversalAccumulator::merge)
            };

            self.merge_accumulator(merged);
            self.discount_regrets();
            self.iterations += 1;
        }
        self.cached_initial_states = Some(initial_states);
    }

    /// Merge a `TraversalAccumulator` into the solver's main tables.
    fn merge_accumulator(&mut self, acc: TraversalAccumulator) {
        self.pruned_traversals += acc.pruned_count;
        self.total_traversals += acc.total_count;

        for (key, deltas) in acc.regret_deltas {
            let regrets = self
                .regret_sum
                .entry(key)
                .or_insert_with(|| vec![0.0; deltas.len()]);
            for (i, d) in deltas.iter().enumerate() {
                regrets[i] += d;
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

/// Immutable pruning configuration passed through parallel traversal.
#[derive(Clone, Copy)]
struct PruningCtx {
    enabled: bool,
    warmup: u64,
    probe_interval: u64,
    threshold: f64,
    iteration: u64,
}

/// Thread-local accumulator for parallel MCCFR traversals.
///
/// Collects regret and strategy deltas during parallel traversal.
/// Merged via tree reduction after all samples complete.
struct TraversalAccumulator {
    regret_deltas: FxHashMap<u64, Vec<f64>>,
    strategy_deltas: FxHashMap<u64, Vec<f64>>,
    rng_state: u64,
    pruned_count: u64,
    total_count: u64,
}

impl TraversalAccumulator {
    fn new() -> Self {
        Self {
            regret_deltas: FxHashMap::default(),
            strategy_deltas: FxHashMap::default(),
            rng_state: 0x1234_5678_9ABC_DEF0,
            pruned_count: 0,
            total_count: 0,
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

        self.pruned_count += other.pruned_count;
        self.total_count += other.total_count;

        self
    }
}

/// Check whether an action should be pruned using the frozen regret snapshot.
///
/// Same logic as `MccfrSolver::should_prune` but reads from immutable snapshot.
fn should_prune_snapshot(
    regret_snapshot: &FxHashMap<u64, Vec<f64>>,
    info_set: u64,
    action_idx: usize,
    pruning: &PruningCtx,
) -> bool {
    if !pruning.enabled || pruning.iteration < pruning.warmup {
        return false;
    }
    if pruning.iteration.is_multiple_of(pruning.probe_interval) {
        return false;
    }
    match regret_snapshot.get(&info_set) {
        Some(regrets) => {
            let has_positive = regrets.iter().any(|&r| r > 0.0);
            has_positive && regrets[action_idx] <= pruning.threshold
        }
        None => false,
    }
}

/// Pure MCCFR traversal that writes deltas to a `TraversalAccumulator`.
///
/// Same logic as `cfr_traverse` but operates on immutable regret snapshot
/// and accumulates deltas without applying CFR+ flooring.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn cfr_traverse_pure<G: Game>(
    game: &G,
    regret_snapshot: &FxHashMap<u64, Vec<f64>>,
    acc: &mut TraversalAccumulator,
    state: &G::State,
    traversing_player: Player,
    p1_reach: f64,
    p2_reach: f64,
    sample_weight: f64,
    strategy_discount: f64,
    pruning: PruningCtx,
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
        let mut pruned: u32 = 0;

        for (i, action) in actions.iter().enumerate() {
            acc.total_count += 1;

            if should_prune_snapshot(regret_snapshot, info_set, i, &pruning) {
                acc.pruned_count += 1;
                pruned |= 1 << i;
                continue;
            }

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
                strategy_discount,
                pruning,
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
            if pruned >> i & 1 != 0 {
                continue;
            }
            regrets[i] +=
                opponent_reach * (action_utils[i] - node_util) * sample_weight;
        }

        if strategy_discount > 0.0 {
            let my_reach = match current_player {
                Player::Player1 => p1_reach,
                Player::Player2 => p2_reach,
            };

            let strat_sums = acc
                .strategy_deltas
                .entry(info_set)
                .or_insert_with(|| vec![0.0; num_actions]);

            for i in 0..num_actions {
                strat_sums[i] += my_reach * strategy[i] * sample_weight * strategy_discount;
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

        // Opponent strategy is NOT accumulated here — it's accumulated
        // when the opponent is the traversing player on alternating iterations.

        cfr_traverse_pure(
            game,
            regret_snapshot,
            acc,
            &next_state,
            traversing_player,
            new_p1,
            new_p2,
            sample_weight,
            strategy_discount,
            pruning,
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

        // Train with full sampling (DCFR with external sampling)
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
    fn dcfr_strategy_discount_weights_later_iterations_more() {
        let game = KuhnPoker::new();

        let mut solver = MccfrSolver::new(game);

        // Iteration 0: discount = (0/1)^2 = 0.0, no strategy accumulated
        // Iteration 1: discount = (1/2)^2 = 0.25
        // Iteration 9: discount = (9/10)^2 = 0.81
        solver.train_full(1);

        // After iteration 0 (discount=0), strategy_sum entries should all be zero
        let all_zero = solver
            .strategy_sum
            .values()
            .all(|v| v.iter().all(|&x| x == 0.0));
        assert!(
            solver.strategy_sum.is_empty() || all_zero,
            "Iteration 0 should have zero strategy discount"
        );

        // Train more — later iterations should accumulate non-trivially
        solver.train_full(9);

        assert!(
            !solver.strategy_sum.is_empty(),
            "Strategy sum should be non-empty after multiple iterations"
        );

        let has_nonzero = solver
            .strategy_sum
            .values()
            .any(|v| v.iter().any(|&x| x > 0.0));
        assert!(has_nonzero, "Later iterations should contribute to strategy sum");
    }

    #[timed_test]
    fn dcfr_config_params_wired_correctly() {
        let config = MccfrConfig {
            dcfr_alpha: 2.0,
            dcfr_beta: 1.0,
            dcfr_gamma: 3.0,
            ..MccfrConfig::default()
        };
        let game = KuhnPoker::new();
        let solver = MccfrSolver::with_config(game, &config);

        assert!((solver.dcfr_alpha - 2.0).abs() < 1e-10);
        assert!((solver.dcfr_beta - 1.0).abs() < 1e-10);
        assert!((solver.dcfr_gamma - 3.0).abs() < 1e-10);
    }

    #[timed_test]
    fn dcfr_default_params() {
        let config = MccfrConfig::default();
        assert!((config.dcfr_alpha - 1.5).abs() < 1e-10);
        assert!((config.dcfr_beta - 0.5).abs() < 1e-10);
        assert!((config.dcfr_gamma - 2.0).abs() < 1e-10);
    }

    #[timed_test]
    fn dcfr_discounts_positive_and_negative_regrets() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        // Manually inject known regrets
        solver.regret_sum.insert(1, vec![10.0, -5.0, 0.0]);
        solver.iterations = 4; // t = 5

        // Expected factors: pos = 5^1.5 / (5^1.5 + 1) ≈ 0.918
        //                   neg = 5^0.5 / (5^0.5 + 1) ≈ 0.691
        let t = 5.0_f64;
        let expected_pos = t.powf(1.5) / (t.powf(1.5) + 1.0);
        let expected_neg = t.powf(0.5) / (t.powf(0.5) + 1.0);

        solver.discount_regrets();

        let regrets = &solver.regret_sum[&1];
        assert!(
            (regrets[0] - 10.0 * expected_pos).abs() < 1e-6,
            "Positive regret should be discounted by alpha: got {}, expected {}",
            regrets[0], 10.0 * expected_pos
        );
        assert!(
            (regrets[1] - (-5.0 * expected_neg)).abs() < 1e-6,
            "Negative regret should be discounted by beta: got {}, expected {}",
            regrets[1], -5.0 * expected_neg
        );
        assert!(
            regrets[2].abs() < 1e-10,
            "Zero regret should remain zero"
        );
    }

    #[timed_test]
    fn dcfr_negative_regrets_decay_faster_than_positive() {
        let game = KuhnPoker::new();
        let mut solver = MccfrSolver::new(game);

        // With default α=1.5, β=0.5, negative regrets decay faster
        solver.regret_sum.insert(1, vec![100.0, -100.0]);
        solver.iterations = 2; // t = 3

        solver.discount_regrets();

        let regrets = &solver.regret_sum[&1];
        let pos_retained = regrets[0] / 100.0;
        let neg_retained = regrets[1].abs() / 100.0;

        assert!(
            neg_retained < pos_retained,
            "Negative regrets should decay faster (β<α): pos_retained={pos_retained:.4}, neg_retained={neg_retained:.4}"
        );
    }

    #[timed_test]
    fn dcfr_pruning_prunes_negative_regret_actions() {
        let game = KuhnPoker::new();
        let config = MccfrConfig {
            pruning: true,
            pruning_warmup: 0,
            pruning_probe_interval: 100,
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game, &config);

        // One positive regret, one negative regret
        solver.regret_sum.insert(999, vec![5.0, -2.0]);
        solver.iterations = 50; // past warmup, not a probe

        assert!(
            solver.should_prune(999, 1),
            "Should prune action with negative regret when a sibling is positive"
        );
        assert!(
            !solver.should_prune(999, 0),
            "Should not prune the positive-regret action"
        );
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

    // --- Pruning tests ---

    #[timed_test]
    fn pruning_disabled_by_default() {
        let config = MccfrConfig::default();
        assert!(!config.pruning, "pruning should be disabled by default");
    }

    #[timed_test]
    fn pruning_skips_zero_regret_actions() {
        let game = KuhnPoker::new();
        let config = MccfrConfig {
            pruning: true,
            pruning_warmup: 0,
            pruning_probe_interval: 100, // large so no probes during test
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game, &config);

        // Train enough to build regret table with some zeros
        solver.train_full(200);

        let (pruned, total) = solver.pruning_stats();
        assert!(total > 0, "should have traversals");
        assert!(pruned > 0, "should prune some zero-regret actions after warmup");
    }

    #[timed_test]
    fn pruning_does_not_skip_during_warmup() {
        let game = KuhnPoker::new();
        let config = MccfrConfig {
            pruning: true,
            pruning_warmup: 1000, // warmup exceeds iterations
            pruning_probe_interval: 100,
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game, &config);

        solver.train_full(100);

        let (pruned, _total) = solver.pruning_stats();
        assert_eq!(pruned, 0, "no pruning should occur during warmup");
    }

    #[timed_test]
    fn probe_iterations_are_full_width() {
        let game = KuhnPoker::new();
        let config = MccfrConfig {
            pruning: true,
            pruning_warmup: 0,
            pruning_probe_interval: 1, // every iteration is a probe
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game, &config);

        solver.train_full(200);

        let (pruned, _total) = solver.pruning_stats();
        assert_eq!(
            pruned, 0,
            "no pruning should occur when every iteration is a probe"
        );
    }

    #[timed_test]
    fn pruning_does_not_skip_all_non_positive_info_set() {
        let game = KuhnPoker::new();
        let config = MccfrConfig {
            pruning: true,
            pruning_warmup: 0,
            pruning_probe_interval: 100,
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game, &config);
        solver.iterations = 101; // past warmup, not a probe (101 % 100 != 0)

        // When all regrets are zero, has_positive is false → no pruning
        solver.regret_sum.insert(999, vec![0.0, 0.0]);
        assert!(
            !solver.should_prune(999, 0),
            "should not prune when all regrets are zero"
        );
        assert!(
            !solver.should_prune(999, 1),
            "should not prune when all regrets are zero"
        );

        // When all regrets are negative, has_positive is false → no pruning
        solver.regret_sum.insert(999, vec![-3.0, -1.0]);
        assert!(
            !solver.should_prune(999, 0),
            "should not prune when all regrets are negative"
        );

        // If one is positive, the zero or negative ones should be pruned
        solver.regret_sum.insert(999, vec![5.0, 0.0, -2.0]);
        assert!(
            !solver.should_prune(999, 0),
            "should not prune the positive-regret action"
        );
        assert!(
            solver.should_prune(999, 1),
            "should prune the zero-regret action when a sibling is positive"
        );
        assert!(
            solver.should_prune(999, 2),
            "should prune the negative-regret action when a sibling is positive"
        );
    }

    #[timed_test]
    fn pruning_converges_on_kuhn() {
        use crate::info_key::InfoKey;

        let game = KuhnPoker::new();
        let config = MccfrConfig {
            pruning: true,
            pruning_warmup: 100,
            pruning_probe_interval: 20,
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game, &config);

        solver.train_full(10_000);

        // King facing bet → always call (relaxed threshold; pruning adds slight noise)
        let kb = InfoKey::new(2, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(kb) {
            assert!(
                strategy[1] > 0.95,
                "Pruning: King should always call a bet, got {strategy:?}"
            );
        }

        // Jack facing bet → always fold
        let jb = InfoKey::new(0, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(jb) {
            assert!(
                strategy[0] > 0.95,
                "Pruning: Jack should always fold facing a bet, got {strategy:?}"
            );
        }

        // Verify some pruning actually happened
        let (pruned, total) = solver.pruning_stats();
        assert!(pruned > 0, "pruning should have skipped some traversals");
        assert!(total > pruned, "not everything should be pruned");
    }

    #[timed_test]
    fn parallel_pruning_converges_on_kuhn() {
        use crate::info_key::InfoKey;

        let game = KuhnPoker::new();
        let config = MccfrConfig {
            pruning: true,
            pruning_warmup: 100,
            pruning_probe_interval: 20,
            ..MccfrConfig::default()
        };
        let mut solver = MccfrSolver::with_config(game, &config);

        solver.train_full_parallel(10_000);

        // King facing bet → always call (relaxed threshold; pruning adds slight noise)
        let kb = InfoKey::new(2, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(kb) {
            assert!(
                strategy[1] > 0.95,
                "Parallel pruning: King should always call a bet, got {strategy:?}"
            );
        }

        // Jack facing bet → always fold
        let jb = InfoKey::new(0, 0, 0, 0, &[4]).as_u64();
        if let Some(strategy) = solver.get_average_strategy(jb) {
            assert!(
                strategy[0] > 0.95,
                "Parallel pruning: Jack should always fold facing a bet, got {strategy:?}"
            );
        }

        // Verify some pruning actually happened
        let (pruned, total) = solver.pruning_stats();
        assert!(pruned > 0, "parallel pruning should have skipped some traversals");
        assert!(total > pruned, "not everything should be pruned");
    }
}
