//! Leaf evaluator trait for depth-boundary CFV estimation.
//!
//! Used by subgame solvers to evaluate counterfactual values at depth
//! boundaries where the game tree is truncated.

use crate::poker::Card;

/// Evaluates combos at a depth boundary, producing per-combo CFVs.
///
/// Implementations may solve sub-trees exactly (e.g. all river runouts)
/// or use a neural network approximation.
pub trait LeafEvaluator {
    /// Evaluate counterfactual values for a set of combos at a boundary.
    ///
    /// Returns a `Vec<f64>` of length `combos.len()`, where each entry is
    /// the expected value (in pot fractions) for the traversing player.
    ///
    /// # Arguments
    ///
    /// * `combos` - hole card combos active at this boundary
    /// * `board` - current board cards (3-5 cards)
    /// * `pot` - total pot at the boundary
    /// * `effective_stack` - remaining stack for each player
    /// * `oop_range` - OOP per-combo reach probabilities
    /// * `ip_range` - IP per-combo reach probabilities
    /// * `traverser` - which player (0=OOP, 1=IP) is the traverser
    fn evaluate(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        pot: f64,
        effective_stack: f64,
        oop_range: &[f64],
        ip_range: &[f64],
        traverser: u8,
    ) -> Vec<f64>;

    /// Batch-evaluate multiple boundary requests in one call.
    ///
    /// Each request specifies `(pot, effective_stack, traverser)`
    /// for a different boundary node. All requests share the same combos,
    /// board, and ranges.
    ///
    /// Returns one `Vec<f64>` per request (same length as `combos`).
    ///
    /// Default implementation calls [`evaluate`] sequentially. GPU-backed
    /// implementations should override to batch into a single forward pass.
    fn evaluate_boundaries(
        &self,
        combos: &[[Card; 2]],
        board: &[Card],
        oop_range: &[f64],
        ip_range: &[f64],
        requests: &[(f64, f64, u8)], // (pot, effective_stack, traverser)
    ) -> Vec<Vec<f64>> {
        requests
            .iter()
            .map(|&(pot, eff_stack, traverser)| {
                self.evaluate(combos, board, pot, eff_stack, oop_range, ip_range, traverser)
            })
            .collect()
    }
}
