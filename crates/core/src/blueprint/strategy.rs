//! Blueprint Strategy Storage
//!
//! Stores trained strategies extracted from the GPU solver.
//! Strategies are stored as f32 to save memory compared to f64.

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use super::BlueprintError;
use crate::info_key::InfoKey;

/// A single opponent decision point for reach computation.
///
/// Describes one node where the opponent acted: which hand they held,
/// the game state at that point, and which action they chose.
#[derive(Debug, Clone)]
pub struct ReachQuery {
    /// The 28-bit hand field for this hand (canonical index or classification bits).
    pub hand_bits: u32,
    /// Street at this decision point.
    pub street: u8,
    /// SPR bucket at this decision point.
    pub spr: u32,
    /// Action codes leading to this decision point (not including the chosen action).
    pub action_codes: Vec<u8>,
    /// Which action index the opponent chose (0-based into the blueprint prob vector).
    pub action_index: usize,
    /// How many actions were available at this node.
    pub num_actions: usize,
}

/// A decision point template for computing reach across all hands.
///
/// Like [`ReachQuery`] but uses a closure to map hand index to `hand_bits`,
/// allowing batch computation over all hands without pre-building queries.
pub struct ReachDecision {
    /// Maps hand index (`0..num_hands`) to the 28-bit hand field.
    pub hand_bits_fn: Box<dyn Fn(usize) -> u32>,
    /// Street at this decision point.
    pub street: u8,
    /// SPR bucket at this decision point.
    pub spr: u32,
    /// Action codes leading to this decision point (not including the chosen action).
    pub action_codes: Vec<u8>,
    /// Which action index the opponent chose (0-based into the blueprint prob vector).
    pub action_index: usize,
    /// How many actions were available at this node.
    pub num_actions: usize,
}

/// Stores the trained strategy for all information sets.
///
/// The strategy maps u64 information set keys to action probability vectors.
/// Probabilities are stored as f32 to reduce memory usage.
///
/// Serialization uses `HashMap<u64, Vec<f32>>` as an intermediate form
/// for compatibility with bincode.
#[derive(Debug, Clone)]
pub struct BlueprintStrategy {
    strategies: FxHashMap<u64, Vec<f32>>,
    iterations_trained: u64,
}

impl BlueprintStrategy {
    /// Creates a new empty strategy.
    #[must_use]
    pub fn new() -> Self {
        Self {
            strategies: FxHashMap::default(),
            iterations_trained: 0,
        }
    }

    /// Creates a strategy from f64 probabilities, converting to f32.
    ///
    /// This is useful when extracting strategies from the CFR solver,
    /// which operates in f64 precision. The truncation from f64 to f32
    /// is intentional to reduce memory usage; probability values are
    /// well within f32 range [0.0, 1.0].
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_strategies(strategies: FxHashMap<u64, Vec<f64>>, iterations: u64) -> Self {
        let converted = strategies
            .into_iter()
            .map(|(key, probs)| {
                let f32_probs: Vec<f32> = probs.into_iter().map(|p| p as f32).collect();
                (key, f32_probs)
            })
            .collect();

        Self {
            strategies: converted,
            iterations_trained: iterations,
        }
    }

    /// Inserts or updates a strategy for an information set.
    pub fn insert(&mut self, info_set: u64, probs: Vec<f32>) {
        self.strategies.insert(info_set, probs);
    }

    /// Sets the number of training iterations.
    pub fn set_iterations(&mut self, iterations: u64) {
        self.iterations_trained = iterations;
    }

    /// Looks up the action probabilities for an information set.
    ///
    /// Returns `None` if the information set is not in the strategy.
    #[must_use]
    pub fn lookup(&self, info_set: u64) -> Option<&[f32]> {
        self.strategies.get(&info_set).map(Vec::as_slice)
    }

    /// Returns the number of training iterations.
    #[must_use]
    pub fn iterations_trained(&self) -> u64 {
        self.iterations_trained
    }

    /// Returns the number of information sets in the strategy.
    #[must_use]
    pub fn len(&self) -> usize {
        self.strategies.len()
    }

    /// Returns true if the strategy contains no information sets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.strategies.is_empty()
    }

    /// Iterate over all (key, strategy) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &Vec<f32>)> {
        self.strategies.iter()
    }

    /// Look up the probability of a specific action for a specific hand at a decision point.
    ///
    /// Builds an info-set key from the given components and returns the blueprint
    /// probability at `action_index`. Returns uniform probability `1/num_actions`
    /// if the info set is not in the blueprint.
    #[must_use]
    pub fn action_probability(
        &self,
        hand_bits: u32,
        street: u8,
        spr: u32,
        action_codes: &[u8],
        action_index: usize,
        num_actions: usize,
    ) -> f64 {
        let key = InfoKey::new(hand_bits, street, spr, action_codes).as_u64();
        match self.lookup(key) {
            Some(probs) if action_index < probs.len() => f64::from(probs[action_index]),
            _ => {
                if num_actions == 0 {
                    0.0
                } else {
                    #[allow(clippy::cast_precision_loss)]
                    {
                        1.0 / num_actions as f64
                    }
                }
            }
        }
    }

    /// Compute opponent reach for a single hand over a sequence of decisions.
    ///
    /// Returns the product of blueprint probabilities for each action the
    /// opponent took at each of their decision points. The result is the
    /// probability that this particular hand reached the current node given
    /// the observed action sequence.
    #[must_use]
    pub fn opponent_reach_for_hand(&self, queries: &[ReachQuery]) -> f64 {
        queries.iter().fold(1.0, |reach, q| {
            reach
                * self.action_probability(
                    q.hand_bits,
                    q.street,
                    q.spr,
                    &q.action_codes,
                    q.action_index,
                    q.num_actions,
                )
        })
    }

    /// Compute opponent reach for all hands 0..`num_hands`.
    ///
    /// For each hand index, maps it to `hand_bits` via each decision's `hand_bits_fn`,
    /// then multiplies the blueprint probability of the chosen action across all
    /// decision points. Returns a vector of length `num_hands`.
    ///
    /// With no decisions, all hands have reach 1.0 (uniform prior).
    #[must_use]
    pub fn opponent_reach_all(&self, num_hands: usize, decisions: &[ReachDecision]) -> Vec<f64> {
        let mut reach = vec![1.0; num_hands];
        for decision in decisions {
            for (hand_idx, r) in reach.iter_mut().enumerate() {
                let hand_bits = (decision.hand_bits_fn)(hand_idx);
                let prob = self.action_probability(
                    hand_bits,
                    decision.street,
                    decision.spr,
                    &decision.action_codes,
                    decision.action_index,
                    decision.num_actions,
                );
                *r *= prob;
            }
        }
        reach
    }

    /// Estimate continuation value for a hand at a depth boundary.
    ///
    /// Returns 0.0 as a conservative default. A full implementation would
    /// perform a forward pass through the blueprint subtree rooted at this
    /// node, weighting each terminal by the blueprint probability of reaching it.
    ///
    // TODO: implement full forward pass through blueprint subtree
    #[must_use]
    pub fn continuation_value_estimate(
        &self,
        _hand_bits: u32,
        _street: u8,
        _spr: u32,
        _action_codes: &[u8],
    ) -> f64 {
        0.0
    }

    /// Saves the strategy to a file using bincode serialization.
    ///
    /// # Errors
    ///
    /// Returns `BlueprintError::IoError` if file operations fail.
    /// Returns `BlueprintError::SerializationError` if serialization fails.
    pub fn save(&self, path: &Path) -> Result<(), BlueprintError> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        let wire = WireStrategy {
            strategies: self
                .strategies
                .iter()
                .map(|(&k, v)| (k, v.clone()))
                .collect(),
            iterations_trained: self.iterations_trained,
        };
        bincode::serialize_into(writer, &wire)
            .map_err(|e| BlueprintError::SerializationError(e.to_string()))
    }

    /// Loads a strategy from a file.
    ///
    /// # Errors
    ///
    /// Returns `BlueprintError::IoError` if file operations fail.
    /// Returns `BlueprintError::SerializationError` if deserialization fails.
    pub fn load(path: &Path) -> Result<Self, BlueprintError> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let wire: WireStrategy = bincode::deserialize_from(reader)
            .map_err(|e| BlueprintError::SerializationError(e.to_string()))?;
        Ok(Self {
            strategies: wire.strategies.into_iter().collect(),
            iterations_trained: wire.iterations_trained,
        })
    }
}

/// Serializable wire format for `BlueprintStrategy`.
///
/// Uses `HashMap<u64, Vec<f32>>` which bincode can serialize natively,
/// unlike `FxHashMap` which lacks `Serialize`/`Deserialize`.
#[derive(Serialize, Deserialize)]
struct WireStrategy {
    strategies: HashMap<u64, Vec<f32>>,
    iterations_trained: u64,
}

impl Default for BlueprintStrategy {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::float_cmp)]
    use super::*;
    use tempfile::NamedTempFile;
    use test_macros::timed_test;

    #[timed_test]
    fn empty_strategy_returns_none() {
        let strategy = BlueprintStrategy::new();

        let result = strategy.lookup(999);

        assert!(
            result.is_none(),
            "lookup on empty strategy should return None"
        );
    }

    #[timed_test]
    fn strategy_lookup_returns_inserted_values() {
        let mut strategy = BlueprintStrategy::new();
        let probs = vec![0.3, 0.5, 0.2];
        strategy.insert(42, probs.clone());

        let result = strategy.lookup(42);

        assert!(result.is_some(), "lookup should find inserted key");
        assert_eq!(
            result.unwrap(),
            probs.as_slice(),
            "probabilities should match"
        );
    }

    #[timed_test]
    fn strategy_roundtrip_save_load() {
        let mut strategy = BlueprintStrategy::new();
        strategy.insert(100, vec![0.3, 0.5, 0.2]);
        strategy.insert(200, vec![0.1, 0.9]);
        strategy.set_iterations(1000);

        let temp_file = NamedTempFile::new().expect("failed to create temp file");
        let path = temp_file.path();

        strategy.save(path).expect("save should succeed");
        let loaded = BlueprintStrategy::load(path).expect("load should succeed");

        assert_eq!(
            loaded.len(),
            strategy.len(),
            "should have same number of entries"
        );
        assert_eq!(
            loaded.iterations_trained(),
            strategy.iterations_trained(),
            "iterations should match"
        );
        assert_eq!(
            loaded.lookup(100),
            strategy.lookup(100),
            "strategy values should match"
        );
        assert_eq!(
            loaded.lookup(200),
            strategy.lookup(200),
            "strategy values should match"
        );
    }

    #[timed_test]
    fn from_strategies_converts_f64_to_f32() {
        let mut f64_strategies: FxHashMap<u64, Vec<f64>> = FxHashMap::default();
        f64_strategies.insert(10, vec![0.25, 0.75]);
        f64_strategies.insert(20, vec![0.1, 0.3, 0.6]);

        let strategy = BlueprintStrategy::from_strategies(f64_strategies, 500);

        assert_eq!(strategy.iterations_trained(), 500);
        assert_eq!(strategy.len(), 2);

        let probs_10 = strategy.lookup(10).expect("should find key 10");
        assert_eq!(probs_10.len(), 2);
        assert!(
            (probs_10[0] - 0.25_f32).abs() < f32::EPSILON,
            "first prob should be ~0.25"
        );
        assert!(
            (probs_10[1] - 0.75_f32).abs() < f32::EPSILON,
            "second prob should be ~0.75"
        );

        let probs_20 = strategy.lookup(20).expect("should find key 20");
        assert_eq!(probs_20.len(), 3);
        assert!((probs_20[0] - 0.1_f32).abs() < f32::EPSILON);
        assert!((probs_20[1] - 0.3_f32).abs() < f32::EPSILON);
        assert!((probs_20[2] - 0.6_f32).abs() < f32::EPSILON);
    }

    #[timed_test]
    fn len_and_is_empty() {
        let mut strategy = BlueprintStrategy::new();

        assert!(strategy.is_empty(), "new strategy should be empty");
        assert_eq!(strategy.len(), 0, "new strategy should have len 0");

        strategy.insert(1, vec![0.5, 0.5]);

        assert!(
            !strategy.is_empty(),
            "strategy with entry should not be empty"
        );
        assert_eq!(strategy.len(), 1, "strategy should have len 1");

        strategy.insert(2, vec![1.0]);

        assert_eq!(strategy.len(), 2, "strategy should have len 2");
    }

    #[timed_test]
    fn default_creates_empty_strategy() {
        let strategy = BlueprintStrategy::default();

        assert!(strategy.is_empty(), "default strategy should be empty");
        assert_eq!(
            strategy.iterations_trained(),
            0,
            "default iterations should be 0"
        );
    }

    #[timed_test]
    fn insert_overwrites_existing_key() {
        let mut strategy = BlueprintStrategy::new();
        strategy.insert(1, vec![0.5, 0.5]);
        strategy.insert(1, vec![0.3, 0.7]);

        let probs = strategy.lookup(1).expect("should find key");
        assert_eq!(probs, &[0.3, 0.7], "should have updated values");
        assert_eq!(strategy.len(), 1, "should still have one entry");
    }

    #[timed_test]
    fn load_nonexistent_file_returns_error() {
        let result = BlueprintStrategy::load(Path::new("/nonexistent/path/strategy.bin"));

        assert!(result.is_err(), "loading nonexistent file should fail");
        match result.unwrap_err() {
            BlueprintError::IoError(_) => {}
            other => panic!("expected IoError, got {other:?}"),
        }
    }

    // === Task 12: Opponent reach computation ===

    #[timed_test]
    fn action_probability_returns_correct_value() {
        let mut bp = BlueprintStrategy::new();
        let key = InfoKey::new(42, 0, 15, &[]).as_u64();
        bp.insert(key, vec![0.3, 0.5, 0.2]);

        let prob = bp.action_probability(42, 0, 15, &[], 1, 3);

        assert!((prob - 0.5).abs() < 0.001, "expected 0.5, got {prob}");
    }

    #[timed_test]
    fn action_probability_returns_uniform_when_missing() {
        let bp = BlueprintStrategy::new();

        let prob = bp.action_probability(42, 0, 15, &[], 1, 3);

        assert!(
            (prob - 1.0 / 3.0).abs() < 0.001,
            "expected ~0.333, got {prob}"
        );
    }

    #[timed_test]
    fn action_probability_returns_zero_for_zero_actions() {
        let bp = BlueprintStrategy::new();

        let prob = bp.action_probability(42, 0, 15, &[], 0, 0);

        assert!(
            prob.abs() < 0.001,
            "expected 0.0 for zero actions, got {prob}"
        );
    }

    #[timed_test]
    fn opponent_reach_for_hand_single_decision() {
        let mut bp = BlueprintStrategy::new();
        let key = InfoKey::new(10, 0, 15, &[]).as_u64();
        bp.insert(key, vec![0.2, 0.3, 0.5]);

        let queries = vec![ReachQuery {
            hand_bits: 10,
            street: 0,
            spr: 15,
            action_codes: vec![],
            action_index: 2,
            num_actions: 3,
        }];

        let reach = bp.opponent_reach_for_hand(&queries);

        assert!((reach - 0.5).abs() < 0.001, "expected 0.5, got {reach}");
    }

    #[timed_test]
    fn opponent_reach_for_hand_two_decisions_multiplies() {
        let mut bp = BlueprintStrategy::new();
        // First decision: 50% call
        let key1 = InfoKey::new(5, 0, 15, &[]).as_u64();
        bp.insert(key1, vec![0.5, 0.5]);
        // Second decision: 80% bet (after call action code 3)
        let key2 = InfoKey::new(5, 0, 15, &[3]).as_u64();
        bp.insert(key2, vec![0.2, 0.8]);

        let queries = vec![
            ReachQuery {
                hand_bits: 5,
                street: 0,
                spr: 15,
                action_codes: vec![],
                action_index: 1, // call
                num_actions: 2,
            },
            ReachQuery {
                hand_bits: 5,
                street: 0,
                spr: 15,
                action_codes: vec![3], // after call
                action_index: 1,       // bet
                num_actions: 2,
            },
        ];

        let reach = bp.opponent_reach_for_hand(&queries);

        assert!(
            (reach - 0.4).abs() < 0.001,
            "expected 0.5 * 0.8 = 0.4, got {reach}"
        );
    }

    #[timed_test]
    fn opponent_reach_for_hand_empty_decisions_is_one() {
        let bp = BlueprintStrategy::new();

        let reach = bp.opponent_reach_for_hand(&[]);

        assert!(
            (reach - 1.0).abs() < 0.001,
            "expected 1.0 with no decisions, got {reach}"
        );
    }

    #[timed_test]
    fn opponent_reach_all_hands_uniform_with_no_decisions() {
        let bp = BlueprintStrategy::new();

        let reach = bp.opponent_reach_all(169, &[]);

        assert_eq!(reach.len(), 169);
        for &r in &reach {
            assert!(
                (r - 1.0).abs() < 0.001,
                "expected 1.0 with no decisions, got {r}"
            );
        }
    }

    #[timed_test]
    fn opponent_reach_all_narrows_after_action() {
        let mut bp = BlueprintStrategy::new();
        // Hand 0: raises 100% (action index 2)
        let key0 = InfoKey::new(0, 0, 15, &[]).as_u64();
        bp.insert(key0, vec![0.0, 0.0, 1.0]);
        // Hand 1: folds 100% (action index 0)
        let key1 = InfoKey::new(1, 0, 15, &[]).as_u64();
        bp.insert(key1, vec![1.0, 0.0, 0.0]);

        let decisions = vec![ReachDecision {
            hand_bits_fn: Box::new(|h| h as u32),
            street: 0,
            spr: 15,
            action_codes: vec![],
            action_index: 2, // opponent raised
            num_actions: 3,
        }];

        let reach = bp.opponent_reach_all(2, &decisions);

        assert!(
            (reach[0] - 1.0).abs() < 0.01,
            "hand 0 raised, should have full reach, got {}",
            reach[0]
        );
        assert!(
            reach[1] < 0.01,
            "hand 1 folds, should have zero reach after raise, got {}",
            reach[1]
        );
    }

    #[timed_test]
    fn opponent_reach_all_multiple_decisions() {
        let mut bp = BlueprintStrategy::new();
        // Decision 1: hand 0 calls 80%, hand 1 calls 40%
        let k0_d1 = InfoKey::new(0, 0, 15, &[]).as_u64();
        bp.insert(k0_d1, vec![0.2, 0.8]);
        let k1_d1 = InfoKey::new(1, 0, 15, &[]).as_u64();
        bp.insert(k1_d1, vec![0.6, 0.4]);
        // Decision 2 (after call=3): hand 0 checks 50%, hand 1 checks 100%
        let k0_d2 = InfoKey::new(0, 0, 15, &[3]).as_u64();
        bp.insert(k0_d2, vec![0.5, 0.5]);
        let k1_d2 = InfoKey::new(1, 0, 15, &[3]).as_u64();
        bp.insert(k1_d2, vec![1.0, 0.0]);

        let decisions = vec![
            ReachDecision {
                hand_bits_fn: Box::new(|h| h as u32),
                street: 0,
                spr: 15,
                action_codes: vec![],
                action_index: 1, // call
                num_actions: 2,
            },
            ReachDecision {
                hand_bits_fn: Box::new(|h| h as u32),
                street: 0,
                spr: 15,
                action_codes: vec![3], // after call
                action_index: 0,       // check
                num_actions: 2,
            },
        ];

        let reach = bp.opponent_reach_all(2, &decisions);

        assert!(
            (reach[0] - 0.4).abs() < 0.01,
            "hand 0: 0.8 * 0.5 = 0.4, got {}",
            reach[0]
        );
        assert!(
            (reach[1] - 0.4).abs() < 0.01,
            "hand 1: 0.4 * 1.0 = 0.4, got {}",
            reach[1]
        );
    }

    // === Task 13: Continuation values ===

    #[timed_test]
    fn continuation_value_returns_zero_for_unknown() {
        let bp = BlueprintStrategy::new();

        let val = bp.continuation_value_estimate(42, 1, 15, &[]);

        assert!(
            val.abs() < 0.001,
            "expected 0.0 for unknown hand, got {val}"
        );
    }

    #[timed_test]
    fn continuation_value_returns_zero_for_known() {
        let mut bp = BlueprintStrategy::new();
        let key = InfoKey::new(42, 1, 15, &[]).as_u64();
        bp.insert(key, vec![0.1, 0.9]);

        let val = bp.continuation_value_estimate(42, 1, 15, &[]);

        assert!(
            val.abs() < 0.001,
            "placeholder should return 0.0, got {val}"
        );
    }
}
