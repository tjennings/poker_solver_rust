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
            strategies: self.strategies.iter().map(|(&k, v)| (k, v.clone())).collect(),
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
}
