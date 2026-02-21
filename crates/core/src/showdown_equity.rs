//! Showdown equity computation for postflop hands.
//!
//! Enumerates all possible opponent 2-card combos from the remaining deck
//! and computes the fraction of matchups our hand wins (plus half-ties).
//! Used to bin hands into equity buckets for info-set key encoding.

use std::cmp::Ordering;

use crate::poker::{Card, Hand, Rank, Rankable};

/// Compute showdown equity of `hole` against a random opponent on `board`.
///
/// Enumerates all C(remaining, 2) opponent combos from the 52-card deck
/// minus the known cards (hole + board). Returns a value in [0.0, 1.0].
///
/// Board must have 3, 4, or 5 cards.
#[must_use]
pub fn compute_equity(hole: [Card; 2], board: &[Card]) -> f64 {
    let our_rank = rank_hand(hole, board);
    let remaining = remaining_cards(hole, board);

    let (wins, ties, total) = remaining
        .iter()
        .enumerate()
        .flat_map(|(i, &c1)| remaining[i + 1..].iter().map(move |&c2| [c1, c2]))
        .fold((0u32, 0u32, 0u32), |(w, t, n), opp| {
            let opp_rank = rank_hand(opp, board);
            match our_rank.cmp(&opp_rank) {
                Ordering::Greater => (w + 1, t, n + 1),
                Ordering::Equal => (w, t + 1, n + 1),
                Ordering::Less => (w, t, n + 1),
            }
        });

    if total == 0 {
        return 0.5;
    }
    (f64::from(wins) + f64::from(ties) * 0.5) / f64::from(total)
}

/// Map an equity value in \[0.0, 1.0\] to a bin index in \[0, `num_bins`\).
///
/// Linearly maps equity to bin, clamping at `num_bins - 1`.
#[must_use]
pub fn equity_bin(equity: f64, num_bins: u8) -> u8 {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let bin = (equity * f64::from(num_bins)) as u8;
    bin.min(num_bins - 1)
}

/// Rank a hand (hole + board) using `rs_poker`.
fn rank_hand(hole: [Card; 2], board: &[Card]) -> Rank {
    let mut hand = Hand::default();
    for &c in &hole {
        hand.insert(c);
    }
    for &c in board {
        hand.insert(c);
    }
    hand.rank()
}

/// Map a card to a bit index (0..51) for the 52-card bitset.
///
/// Encoding: `value as u8 * 4 + suit as u8` where Value and Suit
/// use their `#[repr(u8)]` discriminants.
fn card_bit(card: Card) -> u32 {
    card.value as u32 * 4 + card.suit as u32
}

/// Collect all 52-card deck cards not in `hole` or `board`.
///
/// Uses a u64 bitset for O(1) exclusion checks and returns a
/// stack-allocated `ArrayVec` to avoid heap allocation.
fn remaining_cards(hole: [Card; 2], board: &[Card]) -> arrayvec::ArrayVec<Card, 50> {
    let mut used = (1u64 << card_bit(hole[0])) | (1u64 << card_bit(hole[1]));
    for &c in board {
        used |= 1u64 << card_bit(c);
    }

    let mut remaining = arrayvec::ArrayVec::new();
    for &v in &crate::poker::ALL_VALUES {
        for &s in &crate::poker::ALL_SUITS {
            let c = Card::new(v, s);
            if used & (1u64 << card_bit(c)) == 0 {
                remaining.push(c);
            }
        }
    }
    remaining
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Suit, Value};
    use test_macros::timed_test;

    fn card(value: Value, suit: Suit) -> Card {
        Card::new(value, suit)
    }

    use Suit::{Club, Diamond, Heart, Spade};
    use Value::{Ace, Eight, Five, Four, Jack, King, Nine, Queen, Seven, Six, Ten, Three, Two};

    #[timed_test]
    fn aces_on_dry_board_high_equity() {
        let hole = [card(Ace, Spade), card(Ace, Heart)];
        let board = [
            card(King, Diamond),
            card(Seven, Club),
            card(Three, Spade),
            card(Nine, Heart),
            card(Two, Diamond),
        ];
        let eq = compute_equity(hole, &board);
        // AA on a dry board should have very high equity (>85%)
        assert!(eq > 0.85, "AA equity: {eq}");
    }

    #[timed_test]
    fn nut_flush_high_equity() {
        let hole = [card(Ace, Heart), card(King, Heart)];
        let board = [
            card(Queen, Heart),
            card(Jack, Heart),
            card(Five, Heart),
            card(Three, Club),
            card(Two, Diamond),
        ];
        let eq = compute_equity(hole, &board);
        // Nut flush should beat almost everything
        assert!(eq > 0.95, "Nut flush equity: {eq}");
    }

    #[timed_test]
    fn weak_hand_low_equity() {
        let hole = [card(Seven, Club), card(Two, Diamond)];
        let board = [
            card(Ace, Spade),
            card(King, Heart),
            card(Queen, Diamond),
            card(Jack, Club),
            card(Nine, Spade),
        ];
        let eq = compute_equity(hole, &board);
        // 72o on AKQJ9 — very weak
        assert!(eq < 0.20, "72o equity: {eq}");
    }

    #[timed_test]
    fn equity_on_flop() {
        let hole = [card(Ace, Spade), card(King, Spade)];
        let board = [card(Queen, Heart), card(Seven, Diamond), card(Three, Club)];
        let eq = compute_equity(hole, &board);
        // AKo on Q73 rainbow — overcards, reasonable equity
        assert!(eq > 0.30 && eq < 0.70, "AK on Q73 equity: {eq}");
    }

    #[timed_test]
    fn equity_bin_boundaries() {
        assert_eq!(equity_bin(0.0, 16), 0);
        assert_eq!(equity_bin(0.999, 16), 15);
        assert_eq!(equity_bin(1.0, 16), 15); // clamped
        assert_eq!(equity_bin(0.5, 16), 8);
        assert_eq!(equity_bin(0.0, 8), 0);
        assert_eq!(equity_bin(0.99, 8), 7);
    }

    #[timed_test]
    fn equity_bin_with_4_bins() {
        assert_eq!(equity_bin(0.0, 4), 0);
        assert_eq!(equity_bin(0.24, 4), 0);
        assert_eq!(equity_bin(0.25, 4), 1);
        assert_eq!(equity_bin(0.74, 4), 2);
        assert_eq!(equity_bin(0.75, 4), 3);
        assert_eq!(equity_bin(1.0, 4), 3);
    }

    #[timed_test]
    fn remaining_cards_count() {
        let hole = [card(Ace, Spade), card(King, Heart)];
        let board = [card(Queen, Diamond), card(Jack, Club), card(Ten, Spade)];
        let remaining = remaining_cards(hole, &board);
        assert_eq!(remaining.len(), 47); // 52 - 5
    }

    #[timed_test]
    fn remaining_cards_excludes_known() {
        let hole = [card(Ace, Spade), card(King, Heart)];
        let board = [card(Queen, Diamond), card(Jack, Club), card(Ten, Spade)];
        let remaining = remaining_cards(hole, &board);
        assert!(!remaining.contains(&hole[0]));
        assert!(!remaining.contains(&hole[1]));
        for &b in &board {
            assert!(!remaining.contains(&b));
        }
    }

    #[timed_test]
    fn card_bit_unique_for_all_52() {
        let values = [
            Two, Three, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace,
        ];
        let suits = [Spade, Heart, Diamond, Club];
        let mut seen = 0u64;
        for &v in &values {
            for &s in &suits {
                let bit = card_bit(Card::new(v, s));
                assert!(bit < 52, "bit index out of range: {bit}");
                assert_eq!(seen & (1u64 << bit), 0, "duplicate bit for {v:?} {s:?}");
                seen |= 1u64 << bit;
            }
        }
        assert_eq!(seen.count_ones(), 52);
    }
}
