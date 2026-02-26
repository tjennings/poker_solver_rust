//! Shared utilities for 169-canonical-hand postflop abstraction.
//!
//! Provides combo mapping, board-conflict detection, card parsing, and canonical
//! flop enumeration used by both MCCFR and Exhaustive postflop backends.

use crate::hands::{all_hands, CanonicalHand};
use crate::poker::{Card, Suit, Value};

/// Number of strategically distinct preflop hand categories.
pub const NUM_CANONICAL_HANDS: usize = 169;

/// Check whether either hole card conflicts with any board card.
#[must_use]
pub fn board_conflicts(hand: [Card; 2], board: &[Card]) -> bool {
    board.iter().any(|c| *c == hand[0] || *c == hand[1])
}

/// All 52 cards as a `Vec`.
#[must_use]
pub fn all_cards_vec() -> Vec<Card> {
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
        .flat_map(|v| SUITS.into_iter().map(move |s| Card::new(v, s)))
        .collect()
}

/// For each canonical hand (0..168), list all concrete `(Card, Card)` combos
/// that do not conflict with `flop`.
#[must_use]
pub fn build_combo_map(flop: &[Card; 3]) -> Vec<Vec<(Card, Card)>> {
    all_hands()
        .map(|hand: CanonicalHand| {
            hand.combos()
                .into_iter()
                .filter(|&(c1, c2)| !board_conflicts([c1, c2], flop))
                .collect()
        })
        .collect()
}

/// Parse a two-character card string like `"Ah"` into a [`Card`].
///
/// Format: value char (`2`-`9`, `T`, `J`, `Q`, `K`, `A`) followed by suit char
/// (`s`, `h`, `d`, `c`).
#[must_use]
pub fn parse_card(s: &str) -> Option<Card> {
    let mut chars = s.chars();
    let value = Value::from_char(chars.next()?)?;
    let suit = Suit::from_char(chars.next()?)?;
    if chars.next().is_some() {
        return None; // trailing chars
    }
    Some(Card::new(value, suit))
}

/// Parse a six-character flop string like `"AhKd2s"` into three cards.
#[must_use]
pub fn parse_flop(s: &str) -> Option<[Card; 3]> {
    if s.len() != 6 {
        return None;
    }
    let c1 = parse_card(&s[0..2])?;
    let c2 = parse_card(&s[2..4])?;
    let c3 = parse_card(&s[4..6])?;
    if c1 == c2 || c2 == c3 || c1 == c3 {
        return None;
    }
    Some([c1, c2, c3])
}

/// Parse a slice of flop name strings into validated flop arrays.
///
/// # Errors
///
/// Returns an error if any string is malformed or contains duplicate cards
/// within a single flop.
pub fn parse_flops(names: &[String]) -> Result<Vec<[Card; 3]>, String> {
    let mut result = Vec::with_capacity(names.len());
    for (i, name) in names.iter().enumerate() {
        let flop =
            parse_flop(name).ok_or_else(|| format!("invalid flop at index {i}: {name:?}"))?;
        result.push(flop);
    }
    Ok(result)
}

/// All canonical flops (1755 strategically distinct three-card boards).
///
/// Delegates to `flops::all_flops()` which uses suit-frequency-based
/// canonicalization via `CanonicalBoard::from_cards()`.
#[must_use]
pub fn canonical_flops() -> Vec<[Card; 3]> {
    crate::flops::all_flops()
        .into_iter()
        .map(|f| *f.cards())
        .collect()
}

/// Return a random sample of `n` canonical flops.
///
/// If `n == 0` or `n >= total`, returns all canonical flops.
#[must_use]
pub fn sample_canonical_flops(n: usize) -> Vec<[Card; 3]> {
    use rand::seq::SliceRandom;

    let mut all = canonical_flops();
    if n == 0 || n >= all.len() {
        return all;
    }
    let mut rng = rand::rng();
    all.shuffle(&mut rng);
    all.truncate(n);
    all
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};
    use test_macros::timed_test;

    fn card(v: Value, s: Suit) -> Card {
        Card::new(v, s)
    }

    #[timed_test]
    fn build_combo_map_has_169_entries() {
        let flop = [
            card(Value::Two, Suit::Spade),
            card(Value::Seven, Suit::Heart),
            card(Value::Queen, Suit::Diamond),
        ];
        let map = build_combo_map(&flop);
        assert_eq!(map.len(), 169);
    }

    #[timed_test]
    fn combo_map_no_board_conflicts() {
        let flop = [
            card(Value::Two, Suit::Spade),
            card(Value::Seven, Suit::Heart),
            card(Value::Queen, Suit::Diamond),
        ];
        let map = build_combo_map(&flop);
        for combos in &map {
            for &(c1, c2) in combos {
                assert!(
                    !board_conflicts([c1, c2], &flop),
                    "hand ({c1}, {c2}) conflicts with flop"
                );
            }
        }
    }

    #[timed_test]
    fn combo_map_paired_hand_has_at_most_6() {
        let flop = [
            card(Value::Two, Suit::Spade),
            card(Value::Seven, Suit::Heart),
            card(Value::Queen, Suit::Diamond),
        ];
        let map = build_combo_map(&flop);
        for i in 0..13 {
            assert!(
                map[i].len() <= 6,
                "pair hand {i} has {} combos",
                map[i].len()
            );
        }
    }

    #[timed_test]
    fn all_cards_vec_has_52() {
        assert_eq!(all_cards_vec().len(), 52);
    }

    #[timed_test]
    fn board_conflicts_detects_overlap() {
        let flop = [
            card(Value::Two, Suit::Spade),
            card(Value::Seven, Suit::Heart),
            card(Value::Queen, Suit::Diamond),
        ];
        assert!(board_conflicts(
            [
                card(Value::Two, Suit::Spade),
                card(Value::Ace, Suit::Club)
            ],
            &flop,
        ));
        assert!(!board_conflicts(
            [
                card(Value::Ace, Suit::Spade),
                card(Value::King, Suit::Club)
            ],
            &flop,
        ));
    }

    #[timed_test]
    fn canonical_flops_count() {
        assert_eq!(canonical_flops().len(), 1755);
    }

    #[timed_test]
    fn parse_flop_valid() {
        let flop = parse_flop("AhKd2s").unwrap();
        assert_eq!(flop[0].value, Value::Ace);
        assert_eq!(flop[0].suit, Suit::Heart);
    }

    #[timed_test]
    fn parse_flops_valid() {
        let flops = parse_flops(&["AhKd2s".to_string(), "7c8c9c".to_string()]).unwrap();
        assert_eq!(flops.len(), 2);
    }

    #[timed_test]
    fn sample_canonical_flops_zero_returns_all() {
        let all = sample_canonical_flops(0);
        assert_eq!(all.len(), 1755);
    }
}
