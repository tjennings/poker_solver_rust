//! Hand Strength Metrics for Card Abstraction
//!
//! Provides types for measuring hand strength including Expected Hand Strength (EHS),
//! potential metrics, and the EHS2 formula that combines current strength with
//! hand potential.

use crate::poker::{Card, Hand, Rankable, Suit, Value};
use std::collections::HashSet;

/// Hand strength metrics for poker hand evaluation.
///
/// This struct captures both the current hand strength and the potential
/// for improvement or degradation as more cards are dealt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandStrength {
    /// Expected Hand Strength - equity vs random opponent now
    pub ehs: f32,
    /// Positive potential - P(behind now, ahead after runout)
    pub ppot: f32,
    /// Negative potential - P(ahead now, behind after runout)
    pub npot: f32,
    /// `EHS2 = EHS + (1-EHS)*PPot - EHS*NPot`
    pub ehs2: f32,
}

impl HandStrength {
    /// Create a new `HandStrength` with all metrics.
    ///
    /// Computes EHS2 automatically from the provided values using:
    /// `EHS2 = EHS + (1-EHS)*PPot - EHS*NPot`
    ///
    /// # Arguments
    /// * `ehs` - Expected Hand Strength (current equity)
    /// * `ppot` - Positive potential (probability of improving from behind)
    /// * `npot` - Negative potential (probability of falling behind from ahead)
    ///
    /// # Examples
    /// ```
    /// use poker_solver_core::abstraction::HandStrength;
    ///
    /// let hs = HandStrength::new(0.5, 0.2, 0.1);
    /// assert!((hs.ehs2 - 0.55).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn new(ehs: f32, ppot: f32, npot: f32) -> Self {
        let ehs2 = ehs + (1.0 - ehs) * ppot - ehs * npot;
        Self {
            ehs,
            ppot,
            npot,
            ehs2,
        }
    }

    /// Create a river hand strength (no potential, just EHS).
    ///
    /// On the river, there are no more cards to come, so positive and negative
    /// potential are both zero, and EHS2 equals EHS.
    ///
    /// # Arguments
    /// * `ehs` - Expected Hand Strength (current equity)
    ///
    /// # Examples
    /// ```
    /// use poker_solver_core::abstraction::HandStrength;
    ///
    /// let hs = HandStrength::river(0.75);
    /// assert_eq!(hs.ehs, 0.75);
    /// assert_eq!(hs.ehs2, 0.75);
    /// assert_eq!(hs.ppot, 0.0);
    /// assert_eq!(hs.npot, 0.0);
    /// ```
    #[must_use]
    pub fn river(ehs: f32) -> Self {
        Self {
            ehs,
            ppot: 0.0,
            npot: 0.0,
            ehs2: ehs,
        }
    }
}

/// Calculator for hand strength metrics.
///
/// Provides methods for computing Expected Hand Strength (EHS) and related
/// metrics at different streets.
pub struct HandStrengthCalculator;

impl HandStrengthCalculator {
    /// Create a new `HandStrengthCalculator`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Calculate EHS on the river using exhaustive enumeration.
    ///
    /// Enumerates all possible opponent holdings and calculates the probability
    /// of winning against a random opponent hand.
    ///
    /// # Arguments
    /// * `board` - The 5 community cards on the river
    /// * `holding` - The player's two hole cards
    ///
    /// # Returns
    /// A `HandStrength` with river EHS (no potential, since no more cards to come).
    #[must_use]
    pub fn calculate_river(&self, board: &[Card], holding: (Card, Card)) -> HandStrength {
        let (h1, h2) = holding;

        // Build set of dead cards
        let mut dead: HashSet<Card> = HashSet::new();
        dead.insert(h1);
        dead.insert(h2);
        for &card in board {
            dead.insert(card);
        }

        // Our 7-card hand
        let mut our_cards: Vec<Card> = board.to_vec();
        our_cards.push(h1);
        our_cards.push(h2);
        let our_hand = Hand::new_with_cards(our_cards);
        let our_rank = our_hand.rank();

        // Enumerate all opponent holdings
        let mut wins = 0u32;
        let mut ties = 0u32;
        let mut losses = 0u32;

        let all_cards_vec: Vec<Card> = all_cards().collect();

        for (i, &opp1) in all_cards_vec.iter().enumerate() {
            if dead.contains(&opp1) {
                continue;
            }
            for &opp2 in all_cards_vec.iter().skip(i + 1) {
                if dead.contains(&opp2) {
                    continue;
                }

                // Opponent's 7-card hand
                let mut opp_cards: Vec<Card> = board.to_vec();
                opp_cards.push(opp1);
                opp_cards.push(opp2);
                let opp_hand = Hand::new_with_cards(opp_cards);
                let opp_rank = opp_hand.rank();

                match our_rank.cmp(&opp_rank) {
                    std::cmp::Ordering::Greater => wins += 1,
                    std::cmp::Ordering::Less => losses += 1,
                    std::cmp::Ordering::Equal => ties += 1,
                }
            }
        }

        let total = wins + ties + losses;
        #[allow(clippy::cast_precision_loss)]
        let ehs = if total > 0 {
            // Precision loss is acceptable for EHS calculation (max ~990 opponent combinations)
            (wins as f32 + ties as f32 / 2.0) / total as f32
        } else {
            0.5
        };

        HandStrength::river(ehs)
    }
}

impl Default for HandStrengthCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate an iterator over all 52 cards in a standard deck.
fn all_cards() -> impl Iterator<Item = Card> {
    const VALUES: [Value; 13] = [
        Value::Two,
        Value::Three,
        Value::Four,
        Value::Five,
        Value::Six,
        Value::Seven,
        Value::Eight,
        Value::Nine,
        Value::Ten,
        Value::Jack,
        Value::Queen,
        Value::King,
        Value::Ace,
    ];
    const SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

    VALUES
        .into_iter()
        .flat_map(move |v| SUITS.into_iter().map(move |s| Card::new(v, s)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn river_ehs2_equals_ehs() {
        let hs = HandStrength::river(0.75);
        assert_eq!(hs.ehs2, hs.ehs);
        assert_eq!(hs.ppot, 0.0);
        assert_eq!(hs.npot, 0.0);
    }

    #[test]
    fn ehs2_formula_correct() {
        let hs = HandStrength::new(0.5, 0.2, 0.1);
        // EHS2 = 0.5 + (1-0.5)*0.2 - 0.5*0.1 = 0.5 + 0.1 - 0.05 = 0.55
        assert!((hs.ehs2 - 0.55).abs() < 0.001);
    }

    #[test]
    fn river_with_zero_ehs() {
        let hs = HandStrength::river(0.0);
        assert_eq!(hs.ehs, 0.0);
        assert_eq!(hs.ehs2, 0.0);
    }

    #[test]
    fn river_with_full_ehs() {
        let hs = HandStrength::river(1.0);
        assert_eq!(hs.ehs, 1.0);
        assert_eq!(hs.ehs2, 1.0);
    }

    #[test]
    fn ehs2_with_high_positive_potential() {
        // A drawing hand with high positive potential
        let hs = HandStrength::new(0.3, 0.5, 0.0);
        // EHS2 = 0.3 + (1-0.3)*0.5 - 0.3*0.0 = 0.3 + 0.35 = 0.65
        assert!((hs.ehs2 - 0.65).abs() < 0.001);
    }

    #[test]
    fn ehs2_with_high_negative_potential() {
        // A vulnerable hand with high negative potential
        let hs = HandStrength::new(0.8, 0.0, 0.3);
        // EHS2 = 0.8 + (1-0.8)*0.0 - 0.8*0.3 = 0.8 - 0.24 = 0.56
        assert!((hs.ehs2 - 0.56).abs() < 0.001);
    }

    #[test]
    fn hand_strength_is_copy() {
        let hs1 = HandStrength::new(0.5, 0.2, 0.1);
        let hs2 = hs1;
        assert_eq!(hs1, hs2);
    }

    #[test]
    fn hand_strength_is_clone() {
        let hs1 = HandStrength::new(0.5, 0.2, 0.1);
        let hs2 = hs1.clone();
        assert_eq!(hs1, hs2);
    }

    #[test]
    fn hand_strength_debug() {
        let hs = HandStrength::river(0.5);
        let debug_str = format!("{:?}", hs);
        assert!(debug_str.contains("HandStrength"));
        assert!(debug_str.contains("0.5"));
    }

    #[test]
    fn river_ehs_nut_flush_near_one() {
        // As Ks on 2s 5s 8s Tc Qd - nut flush
        let board = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Five, Suit::Spade),
            Card::new(Value::Eight, Suit::Spade),
            Card::new(Value::Ten, Suit::Club),
            Card::new(Value::Queen, Suit::Diamond),
        ];
        let holding = (
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_river(&board, holding);

        // Nut flush should have very high equity
        assert!(hs.ehs > 0.90, "Expected EHS > 0.90, got {}", hs.ehs);
    }

    #[test]
    fn river_ehs_weak_hand_low() {
        // 7h 2c on As Ks Qs Js 9d - no pair, no flush
        let board = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Nine, Suit::Diamond),
        ];
        let holding = (
            Card::new(Value::Seven, Suit::Heart),
            Card::new(Value::Two, Suit::Club),
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_river(&board, holding);

        // Weak hand on scary board should have low equity
        assert!(hs.ehs < 0.30, "Expected EHS < 0.30, got {}", hs.ehs);
    }

    #[test]
    fn all_cards_generates_52_cards() {
        let cards: Vec<Card> = all_cards().collect();
        assert_eq!(cards.len(), 52);
    }

    #[test]
    fn all_cards_generates_unique_cards() {
        let cards: Vec<Card> = all_cards().collect();
        let unique: HashSet<Card> = cards.into_iter().collect();
        assert_eq!(unique.len(), 52);
    }

    #[test]
    fn calculator_default_trait() {
        let calc = HandStrengthCalculator::default();
        // Just verify we can create via default
        let board = vec![
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Three, Suit::Heart),
            Card::new(Value::Four, Suit::Diamond),
            Card::new(Value::Five, Suit::Club),
            Card::new(Value::Six, Suit::Spade),
        ];
        let holding = (
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Heart),
        );
        let hs = calc.calculate_river(&board, holding);
        // Just check it computed something valid
        assert!(hs.ehs >= 0.0 && hs.ehs <= 1.0);
    }
}
