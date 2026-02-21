//! Shared card utility functions.
//!
//! Centralizes common card-level operations used across hand classification,
//! flop analysis, isomorphism, and hand evaluation.

use rs_poker::core::{Card, Hand, Rank, Rankable, Value};

/// Convert a `Value` to a numeric rank (Two=2, ..., Ace=14).
#[must_use]
pub fn value_rank(v: Value) -> u8 {
    u8::from(v) + 2
}

/// Evaluate the best 5-card hand from a 2-card holding plus board cards.
#[must_use]
pub fn hand_rank(holding: [Card; 2], board: &[Card]) -> Rank {
    let mut h = Hand::default();
    for &c in board {
        h.insert(c);
    }
    for c in holding {
        h.insert(c);
    }
    h.rank()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rs_poker::core::Suit;
    use test_macros::timed_test;

    #[timed_test]
    fn value_rank_two_is_2() {
        assert_eq!(value_rank(Value::Two), 2);
    }

    #[timed_test]
    fn value_rank_ace_is_14() {
        assert_eq!(value_rank(Value::Ace), 14);
    }

    #[timed_test]
    fn value_rank_all_values_ascending() {
        let values = [
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
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(value_rank(v), (i + 2) as u8);
        }
    }

    #[timed_test]
    fn hand_rank_evaluates_seven_cards() {
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let board = [
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Jack, Suit::Spade),
            Card::new(Value::Ten, Suit::Spade),
            Card::new(Value::Two, Suit::Heart),
            Card::new(Value::Three, Suit::Diamond),
        ];
        let rank = hand_rank(holding, &board);
        assert!(matches!(rank, Rank::StraightFlush(_)));
    }

    #[timed_test]
    fn hand_rank_works_with_five_card_board() {
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::Ace, Suit::Heart),
        ];
        let board = [
            Card::new(Value::Ace, Suit::Diamond),
            Card::new(Value::Ace, Suit::Club),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Two, Suit::Heart),
            Card::new(Value::Three, Suit::Diamond),
        ];
        let rank = hand_rank(holding, &board);
        assert!(matches!(rank, Rank::FourOfAKind(_)));
    }

    #[timed_test]
    fn hand_rank_works_with_partial_board() {
        let holding = [
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ];
        let board = [
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Diamond),
            Card::new(Value::Ten, Suit::Club),
        ];
        let rank = hand_rank(holding, &board);
        assert!(matches!(rank, Rank::Straight(_)));
    }
}
