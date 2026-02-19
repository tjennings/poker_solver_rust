//! Precomputed equity table for canonical hand matchups.
//!
//! Stores a 169x169 matrix of equities and card-removal weights for
//! all canonical preflop hand pairings. The full computation is expensive;
//! use `new_uniform()` for testing with equal equities.

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

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
