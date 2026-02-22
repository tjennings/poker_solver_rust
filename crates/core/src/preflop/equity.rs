//! Precomputed equity table for canonical hand matchups.
//!
//! Stores a 169x169 matrix of equities and card-removal weights for
//! all canonical preflop hand pairings. The full computation is expensive;
//! use `new_uniform()` for testing with equal equities.

use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

use serde::{Deserialize, Serialize};

use crate::equity::calculate_equity;
use crate::hands::CanonicalHand;
use crate::poker::Card;

const NUM_CANONICAL_HANDS: usize = 169;

/// A precomputed equity table for 169 canonical preflop hands.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

    /// Returns the average equity of `hand_i` vs all other hands,
    /// weighted by card-removal counts.
    #[must_use]
    pub fn avg_equity(&self, hand_i: usize) -> f64 {
        let n = self.equities.len();
        let mut total_eq = 0.0;
        let mut total_weight = 0.0;
        for j in 0..n {
            if j == hand_i {
                continue;
            }
            let w = self.weights[hand_i][j];
            total_eq += self.equities[hand_i][j] * w;
            total_weight += w;
        }
        if total_weight > 0.0 {
            total_eq / total_weight
        } else {
            0.5
        }
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
        let weights = compute_card_removal_weights();

        Self { equities, weights }
    }
}

/// Count non-overlapping combo pairs for two canonical hands.
fn count_non_overlapping(c1: &[(Card, Card)], c2: &[(Card, Card)]) -> usize {
    c1.iter()
        .flat_map(|a| c2.iter().map(move |b| (a, b)))
        .filter(|((a1, a2), (b1, b2))| *a1 != *b1 && *a1 != *b2 && *a2 != *b1 && *a2 != *b2)
        .count()
}

/// Compute the 169x169 card-removal weight matrix.
///
/// `weights[i][j]` = number of non-overlapping (`combo_a`, `combo_b`) pairs
/// for canonical hands i and j. Symmetric: `weights[i][j] == weights[j][i]`.
#[allow(clippy::cast_precision_loss)] // max value is 144 (12×12), fits in f64 exactly
fn compute_card_removal_weights() -> Vec<Vec<f64>> {
    let combos: Vec<Vec<(Card, Card)>> = (0..NUM_CANONICAL_HANDS)
        .map(|i| {
            CanonicalHand::from_index(i)
                .expect("valid canonical index")
                .combos()
        })
        .collect();

    let mut weights = vec![vec![0.0; NUM_CANONICAL_HANDS]; NUM_CANONICAL_HANDS];
    for i in 0..NUM_CANONICAL_HANDS {
        for j in i..NUM_CANONICAL_HANDS {
            let w = count_non_overlapping(&combos[i], &combos[j]) as f64;
            weights[i][j] = w;
            weights[j][i] = w;
        }
    }
    weights
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

    #[timed_test]
    fn weight_aa_vs_kk_is_36() {
        let weights = compute_card_removal_weights();
        let aa = CH::parse("AA").unwrap().index();
        let kk = CH::parse("KK").unwrap().index();
        assert_eq!(weights[aa][kk], 36.0, "AA vs KK: 6x6, no shared ranks");
    }

    #[timed_test]
    fn weight_aa_vs_aks_is_12() {
        let weights = compute_card_removal_weights();
        let aa = CH::parse("AA").unwrap().index();
        let aks = CH::parse("AKs").unwrap().index();
        assert_eq!(
            weights[aa][aks], 12.0,
            "AA vs AKs: shared ace reduces combos"
        );
    }

    #[timed_test]
    fn weight_aks_vs_ako_is_24() {
        let weights = compute_card_removal_weights();
        let aks = CH::parse("AKs").unwrap().index();
        let ako = CH::parse("AKo").unwrap().index();
        assert_eq!(weights[aks][ako], 24.0, "AKs vs AKo: shared A and K");
    }

    #[timed_test]
    fn weight_aa_vs_aa_is_6() {
        let weights = compute_card_removal_weights();
        let aa = CH::parse("AA").unwrap().index();
        // C(4,2) ways to split 4 aces into two non-overlapping pairs = 3
        // But we count ordered pairs: (combo_a, combo_b) where a != b
        // 6 combos x 6 combos = 36, minus 6 self-overlaps, minus 24 card-overlaps = 6
        assert_eq!(weights[aa][aa], 6.0);
    }

    #[timed_test]
    fn weight_symmetry() {
        let weights = compute_card_removal_weights();
        for i in 0..NUM_CANONICAL_HANDS {
            for j in (i + 1)..NUM_CANONICAL_HANDS {
                assert_eq!(
                    weights[i][j], weights[j][i],
                    "weight[{i}][{j}] != weight[{j}][{i}]"
                );
            }
        }
    }

    #[timed_test]
    fn weight_no_shared_ranks_is_product() {
        let weights = compute_card_removal_weights();
        // 72o vs 83o: no shared ranks → 12 * 12 = 144
        let seven_two = CH::parse("72o").unwrap();
        let eight_three = CH::parse("83o").unwrap();
        let w = weights[seven_two.index()][eight_three.index()];
        let expected = f64::from(seven_two.num_combos()) * f64::from(eight_three.num_combos());
        assert_eq!(w, expected, "no shared ranks should give full product");
    }

    #[timed_test]
    fn self_weight_suited_is_12() {
        let weights = compute_card_removal_weights();
        let aks = CH::parse("AKs").unwrap().index();
        // 4 suited combos: AsKs, AhKh, AdKd, AcKc
        // Ordered pairs (a, b) with different suits: 4 * 3 = 12
        assert_eq!(weights[aks][aks], 12.0);
    }
}
