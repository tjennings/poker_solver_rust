//! Precomputed equity table for canonical hand matchups.
//!
//! Stores a 169x169 matrix of equities and card-removal weights for
//! all canonical preflop hand pairings. The full computation is expensive;
//! use `new_uniform()` for testing with equal equities.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use crate::equity::calculate_equity;
use crate::hands::CanonicalHand;

const NUM_CANONICAL_HANDS: usize = 169;

/// A precomputed equity table for 169 canonical preflop hands.
#[derive(Debug, Clone)]
pub struct EquityTable {
    /// Equity of hand i vs hand j: `equities[i][j]`.
    equities: Vec<Vec<f64>>,
    /// Card-removal weight for hand i vs hand j: `weights[i][j]`.
    weights: Vec<Vec<f64>>,
}

impl EquityTable {
    /// Creates a uniform table where all equities are 0.5 and all weights are 1.0.
    /// Useful for testing without running the full equity computation.
    #[must_use]
    pub fn new_uniform() -> Self {
        Self {
            equities: vec![vec![0.5; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS],
            weights: vec![vec![1.0; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS],
        }
    }

    /// Returns the equity of hand `hand_i` vs hand `hand_j`.
    #[must_use]
    pub fn equity(&self, hand_i: usize, hand_j: usize) -> f64 {
        self.equities[hand_i][hand_j]
    }

    /// Returns the card-removal weight for hand `hand_i` vs hand `hand_j`.
    #[must_use]
    pub fn weight(&self, hand_i: usize, hand_j: usize) -> f64 {
        self.weights[hand_i][hand_j]
    }

    /// Returns the number of canonical hands (always 169).
    #[must_use]
    pub fn num_hands(&self) -> usize {
        self.equities.len()
    }

    /// Compute real equities for all 169x169 canonical hand matchups.
    ///
    /// Uses Monte Carlo simulation with `samples` per matchup.
    /// Calls `on_progress(pairs_done)` after each unique pair is computed.
    /// Computation is parallelized across all available cores via rayon.
    ///
    /// # Panics
    /// Panics if a canonical hand index (0..168) is invalid (should never happen).
    #[must_use]
    pub fn new_computed(samples: u32, on_progress: impl Fn(usize) + Sync + Send) -> Self {
        let pairs: Vec<(usize, usize)> = (0..NUM_CANONICAL_HANDS)
            .flat_map(|i| ((i + 1)..NUM_CANONICAL_HANDS).map(move |j| (i, j)))
            .collect();

        let done = AtomicUsize::new(0);

        let results: Vec<(usize, usize, f64)> = pairs
            .par_iter()
            .map(|&(i, j)| {
                // indices 0..168 are always valid for CanonicalHand
                let h1 = CanonicalHand::from_index(i).expect("valid canonical index");
                let h2 = CanonicalHand::from_index(j).expect("valid canonical index");
                let eq = calculate_equity(h1, h2, samples);
                let count = done.fetch_add(1, Ordering::Relaxed) + 1;
                on_progress(count);
                (i, j, eq)
            })
            .collect();

        let mut equities = vec![vec![0.5; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS];
        for (i, j, eq) in results {
            equities[i][j] = eq;
            equities[j][i] = 1.0 - eq;
        }
        let weights = vec![vec![1.0; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS];

        Self { equities, weights }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hands::CanonicalHand as CH;
    use test_macros::timed_test;

    #[test]
    #[ignore] // ~60s in debug mode (14,196 MC equity computations)
    fn computed_table_has_real_equities() {
        let table = EquityTable::new_computed(500, |_| {});
        let aa = CH::parse("AA").unwrap().index();
        let seven_two = CH::parse("72o").unwrap().index();
        let eq = table.equity(aa, seven_two);
        assert!(eq > 0.80, "AA vs 72o equity should be > 0.80, got {eq}");
    }

    #[test]
    #[ignore] // ~25s in debug mode
    fn computed_equities_are_symmetric() {
        let table = EquityTable::new_computed(200, |_| {});
        for i in 0..NUM_CANONICAL_HANDS {
            for j in (i + 1)..NUM_CANONICAL_HANDS {
                let sum = table.equity(i, j) + table.equity(j, i);
                assert!(
                    (sum - 1.0).abs() < 1e-10,
                    "equity({i},{j}) + equity({j},{i}) = {sum}, expected 1.0"
                );
            }
        }
    }

    #[test]
    #[ignore] // ~25s in debug mode
    fn computed_table_self_matchup_is_half() {
        let table = EquityTable::new_computed(200, |_| {});
        for i in 0..NUM_CANONICAL_HANDS {
            let eq = table.equity(i, i);
            assert!(
                (eq - 0.5).abs() < f64::EPSILON,
                "self-matchup equity({i},{i}) = {eq}, expected 0.5"
            );
        }
    }

    #[timed_test]
    fn uniform_table_is_169x169() {
        let table = EquityTable::new_uniform();
        assert_eq!(table.num_hands(), 169);
        assert_eq!(table.equities.len(), 169);
        assert_eq!(table.weights.len(), 169);
        for row in &table.equities {
            assert_eq!(row.len(), 169);
        }
        for row in &table.weights {
            assert_eq!(row.len(), 169);
        }
    }

    #[timed_test]
    fn uniform_equity_is_half() {
        let table = EquityTable::new_uniform();
        for i in 0..169 {
            for j in 0..169 {
                assert!(
                    (table.equity(i, j) - 0.5).abs() < f64::EPSILON,
                    "Expected equity 0.5 at ({}, {}), got {}",
                    i,
                    j,
                    table.equity(i, j)
                );
            }
        }
    }

    #[timed_test]
    fn equity_lookup_works() {
        let table = EquityTable::new_uniform();
        assert!((table.equity(0, 0) - 0.5).abs() < f64::EPSILON);
        assert!((table.equity(168, 168) - 0.5).abs() < f64::EPSILON);
        assert!((table.weight(0, 0) - 1.0).abs() < f64::EPSILON);
        assert!((table.weight(84, 42) - 1.0).abs() < f64::EPSILON);
    }
}
