//! Opponent range tracking via 1326-combo weight vectors.
//!
//! Each combo corresponds to one of the C(52,2) hole-card pairs. Weights start
//! at 1.0 (uniform range) and are narrowed by multiplying with blueprint action
//! probabilities and zeroing combos that conflict with known board/hero cards.

use crate::poker::{Card, Suit, Value};

/// Number of two-card combinations from a 52-card deck: C(52,2) = 1326.
pub const NUM_COMBOS: usize = 1326;

/// Tracks opponent range as a 1326-element weight vector.
///
/// Weights start uniform at 1.0 and are narrowed over the course of a hand
/// by two operations:
/// - **Bayesian update**: multiply each combo's weight by the blueprint
///   probability that the opponent takes the observed action with that combo.
/// - **Card removal**: zero any combo that shares a card with the board or
///   hero's hole cards.
pub struct RangeNarrower {
    weights: Vec<f64>,
}

impl RangeNarrower {
    /// Create a new uniform range (all 1326 weights = 1.0).
    #[must_use]
    pub fn new() -> Self {
        Self {
            weights: vec![1.0; NUM_COMBOS],
        }
    }

    /// Borrow the weight vector.
    #[must_use]
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// Multiply each combo weight by the corresponding action probability.
    ///
    /// `action_probs` must have length `NUM_COMBOS`. Each entry is the
    /// probability (0.0..=1.0) that the opponent takes the observed action
    /// when holding that combo.
    ///
    /// # Panics
    ///
    /// Panics if `action_probs.len() != NUM_COMBOS`.
    pub fn update(&mut self, action_probs: &[f64]) {
        assert_eq!(
            action_probs.len(),
            NUM_COMBOS,
            "action_probs length must be {NUM_COMBOS}"
        );
        for (w, &p) in self.weights.iter_mut().zip(action_probs) {
            *w *= p;
        }
    }

    /// Zero the weight of every combo that shares a card with the board or
    /// hero's hole cards.
    pub fn apply_card_removal(&mut self, board: &[Card], hero: &[Card; 2]) {
        let mut blocked = [false; 52];
        for &card in board {
            blocked[card_index(&card)] = true;
        }
        for card in hero {
            blocked[card_index(card)] = true;
        }

        for (idx, w) in self.weights.iter_mut().enumerate() {
            let (c1, c2) = combo_cards(idx);
            if blocked[c1] || blocked[c2] {
                *w = 0.0;
            }
        }
    }

    /// Count of combos with nonzero weight.
    #[must_use]
    pub fn live_combo_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }

    /// Reset all weights to 1.0 (uniform range).
    pub fn reset(&mut self) {
        self.weights.fill(1.0);
    }
}

impl Default for RangeNarrower {
    fn default() -> Self {
        Self::new()
    }
}

/// Map a `Card` to a unique index in 0..52.
///
/// Layout: `value_index * 4 + suit_index` where values run Two=0..Ace=12
/// and suits run Spade=0, Heart=1, Diamond=2, Club=3.
///
/// This matches the canonical ordering used by [`crate::blueprint_v2::cluster_pipeline::card_to_deck_index`].
#[must_use]
pub fn card_index(card: &Card) -> usize {
    let value_idx = match card.value {
        Value::Two => 0,
        Value::Three => 1,
        Value::Four => 2,
        Value::Five => 3,
        Value::Six => 4,
        Value::Seven => 5,
        Value::Eight => 6,
        Value::Nine => 7,
        Value::Ten => 8,
        Value::Jack => 9,
        Value::Queen => 10,
        Value::King => 11,
        Value::Ace => 12,
    };
    let suit_idx = match card.suit {
        Suit::Spade => 0,
        Suit::Heart => 1,
        Suit::Diamond => 2,
        Suit::Club => 3,
    };
    value_idx * 4 + suit_idx
}

/// Decompose a combo index (0..1326) into its two card indices (c1, c2)
/// where c1 < c2.
///
/// This is the inverse of [`combo_index`].
#[must_use]
pub fn combo_cards(combo_idx: usize) -> (usize, usize) {
    debug_assert!(combo_idx < NUM_COMBOS, "combo_idx out of range");
    // Binary search for c1: the first card index such that the number of
    // combos starting at or before c1 exceeds combo_idx.
    // Combos with first card < c1: c1 * (103 - c1) / 2
    let mut c1 = 0;
    for candidate in 0..52 {
        let next_start = (candidate + 1) * (103 - (candidate + 1)) / 2;
        if next_start > combo_idx {
            c1 = candidate;
            break;
        }
    }
    let row_start = c1 * (103 - c1) / 2;
    let c2 = combo_idx - row_start + c1 + 1;
    (c1, c2)
}

/// Map a pair of card indices to a combo index (0..1326).
///
/// Requires `c1 < c2` and both in 0..52.
///
/// # Panics
///
/// Panics (in debug) if `c1 >= c2` or either index is >= 52.
#[must_use]
pub fn combo_index(c1: usize, c2: usize) -> usize {
    debug_assert!(c1 < c2, "c1 must be less than c2");
    debug_assert!(c2 < 52, "card indices must be < 52");
    c1 * (103 - c1) / 2 + c2 - c1 - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};

    #[test]
    fn test_new_starts_uniform() {
        let rn = RangeNarrower::new();
        assert_eq!(rn.weights().len(), NUM_COMBOS);
        for &w in rn.weights() {
            assert!((w - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_update_multiplies_weights() {
        let mut rn = RangeNarrower::new();

        // Set up action probs: 0.5 for index 0, 0.0 for index 1, 1.0 elsewhere
        let mut probs = vec![1.0; NUM_COMBOS];
        probs[0] = 0.5;
        probs[1] = 0.0;

        rn.update(&probs);

        assert!((rn.weights()[0] - 0.5).abs() < f64::EPSILON);
        assert!((rn.weights()[1] - 0.0).abs() < f64::EPSILON);
        assert!((rn.weights()[2] - 1.0).abs() < f64::EPSILON);

        // Apply again — weights should compound
        rn.update(&probs);
        assert!((rn.weights()[0] - 0.25).abs() < f64::EPSILON);
        assert!((rn.weights()[1] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_card_removal_zeros_blocked_combos() {
        let mut rn = RangeNarrower::new();

        // Board: As Kh 7d (3 cards), Hero: Ac Qd (2 cards)
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        ];
        let hero = [
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::Queen, Suit::Diamond),
        ];

        rn.apply_card_removal(&board, &hero);

        // Count how many dead cards we have
        let dead_cards: Vec<usize> = board
            .iter()
            .chain(hero.iter())
            .map(|c| card_index(c))
            .collect();
        assert_eq!(dead_cards.len(), 5);

        // Every combo involving a dead card should be zero
        for (idx, &w) in rn.weights().iter().enumerate() {
            let (c1, c2) = combo_cards(idx);
            let blocked = dead_cards.contains(&c1) || dead_cards.contains(&c2);
            if blocked {
                assert_eq!(w, 0.0, "combo {idx} ({c1},{c2}) should be blocked");
            } else {
                assert_eq!(w, 1.0, "combo {idx} ({c1},{c2}) should be live");
            }
        }

        // With 5 dead cards: C(47,2) = 1081 live combos
        assert_eq!(rn.live_combo_count(), 1081);
    }

    #[test]
    fn test_live_combo_count() {
        let mut rn = RangeNarrower::new();
        assert_eq!(rn.live_combo_count(), 1326);

        // Zero out a few combos manually
        rn.weights[0] = 0.0;
        rn.weights[1] = 0.0;
        rn.weights[2] = 0.0;
        assert_eq!(rn.live_combo_count(), 1323);
    }

    #[test]
    fn test_combo_index_roundtrip() {
        for i in 0..NUM_COMBOS {
            let (c1, c2) = combo_cards(i);
            assert!(c1 < c2, "combo_cards({i}) returned c1={c1} >= c2={c2}");
            assert!(c2 < 52, "combo_cards({i}) returned c2={c2} >= 52");
            let back = combo_index(c1, c2);
            assert_eq!(back, i, "roundtrip failed for index {i}: cards ({c1},{c2})");
        }
    }

    #[test]
    fn test_combo_index_enumeration_order() {
        // Verify that combo indices match the canonical enumeration order
        let mut expected = 0;
        for c1 in 0..52 {
            for c2 in (c1 + 1)..52 {
                assert_eq!(
                    combo_index(c1, c2),
                    expected,
                    "combo_index({c1},{c2}) != {expected}"
                );
                expected += 1;
            }
        }
        assert_eq!(expected, NUM_COMBOS);
    }

    #[test]
    fn test_reset() {
        let mut rn = RangeNarrower::new();

        // Mess up the weights
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        ];
        let hero = [
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        rn.apply_card_removal(&board, &hero);
        assert!(rn.live_combo_count() < NUM_COMBOS);

        rn.reset();
        assert_eq!(rn.live_combo_count(), NUM_COMBOS);
        for &w in rn.weights() {
            assert!((w - 1.0).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn test_card_index_values() {
        // Two of Spades = index 0
        assert_eq!(card_index(&Card::new(Value::Two, Suit::Spade)), 0);
        // Two of Club = index 3
        assert_eq!(card_index(&Card::new(Value::Two, Suit::Club)), 3);
        // Ace of Spade = index 48
        assert_eq!(card_index(&Card::new(Value::Ace, Suit::Spade)), 48);
        // Ace of Club = index 51
        assert_eq!(card_index(&Card::new(Value::Ace, Suit::Club)), 51);
    }
}
