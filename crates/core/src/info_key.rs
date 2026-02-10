//! Numeric u64 info set key encoding for fast hashing and zero-allocation lookups.
//!
//! ## Bit Layout
//!
//! ```text
//! Bits 63-36: hand/bucket   (28 bits)
//! Bits 35-34: street         (2 bits)
//! Bits 33-29: pot_bucket     (5 bits)
//! Bits 28-25: stack_bucket   (4 bits)
//! Bit  24:    (reserved)
//! Bits 23-0:  action slots  (24 bits) — up to 6 actions × 4 bits
//! ```
//!
//! Action encoding (4 bits): 0=empty, 1=fold, 2=check, 3=call,
//! 4-7=bet idx 0-3, 8-11=raise idx 0-3, 12=bet all-in, 13=raise all-in.

use crate::game::{Action, ALL_IN};
use crate::poker::{Card, Suit, Value};

const HAND_SHIFT: u32 = 36;
const STREET_SHIFT: u32 = 34;
const POT_SHIFT: u32 = 29;
const STACK_SHIFT: u32 = 25;

/// A packed u64 information set key.
///
/// Encodes hand/bucket, street, pot/stack buckets, and up to 6 actions
/// in a single 64-bit integer for allocation-free hashing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InfoKey(u64);

impl InfoKey {
    /// Build a key from its components.
    ///
    /// # Arguments
    /// * `hand_or_bucket` - Canonical hand index (0-168) or classification bits (up to 28 bits)
    /// * `street` - 0=Preflop, 1=Flop, 2=Turn, 3=River
    /// * `pot_bucket` - pot / 20 (5 bits, max 31)
    /// * `stack_bucket` - `eff_stack` / 20 (4 bits, max 15)
    /// * `actions` - Slice of encoded action codes (from `encode_action`)
    #[must_use]
    pub fn new(
        hand_or_bucket: u32,
        street: u8,
        pot_bucket: u32,
        stack_bucket: u32,
        actions: &[u8],
    ) -> Self {
        let mut key: u64 = 0;
        key |= (u64::from(hand_or_bucket) & 0xFFF_FFFF) << HAND_SHIFT;
        key |= (u64::from(street) & 0x3) << STREET_SHIFT;
        key |= (u64::from(pot_bucket) & 0x1F) << POT_SHIFT;
        key |= (u64::from(stack_bucket) & 0xF) << STACK_SHIFT;

        // Pack up to 6 actions into bits 23..0 (4 bits each, MSB-first)
        // Action 0 → bits 23-20, action 1 → bits 19-16, ..., action 5 → bits 3-0
        for (i, &code) in actions.iter().take(6).enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            let shift = 20 - (i as u32) * 4;
            key |= (u64::from(code) & 0xF) << shift;
        }

        Self(key)
    }

    /// Wrap a raw u64 as an `InfoKey`.
    #[must_use]
    #[inline]
    pub const fn from_raw(raw: u64) -> Self {
        Self(raw)
    }

    /// Extract the raw u64 value.
    #[must_use]
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }

    /// Extract pot bucket (5 bits).
    #[must_use]
    pub const fn pot_bucket(self) -> u32 {
        ((self.0 >> POT_SHIFT) & 0x1F) as u32
    }

    /// Extract stack bucket (4 bits).
    #[must_use]
    pub const fn stack_bucket(self) -> u32 {
        ((self.0 >> STACK_SHIFT) & 0xF) as u32
    }

    /// Return a new key with modified pot and stack buckets.
    #[must_use]
    pub const fn with_buckets(self, pot_bucket: u32, stack_bucket: u32) -> Self {
        let mask = !((0x1F << POT_SHIFT) | (0xF << STACK_SHIFT));
        let cleared = self.0 & mask;
        let new_bits = ((pot_bucket as u64 & 0x1F) << POT_SHIFT)
            | ((stack_bucket as u64 & 0xF) << STACK_SHIFT);
        Self(cleared | new_bits)
    }
}

/// Encode an [`Action`] into a 4-bit code.
///
/// 0=empty, 1=fold, 2=check, 3=call, 4-7=bet idx 0-3,
/// 8-11=raise idx 0-3, 12=bet all-in, 13=raise all-in.
#[must_use]
pub fn encode_action(action: Action) -> u8 {
    match action {
        Action::Fold => 1,
        Action::Check => 2,
        Action::Call => 3,
        Action::Bet(idx) if idx == ALL_IN => 12,
        Action::Bet(idx) => 4 + idx.min(3) as u8,
        Action::Raise(idx) if idx == ALL_IN => 13,
        Action::Raise(idx) => 8 + idx.min(3) as u8,
    }
}

/// Map a canonical hand to a unique index in 0..169.
///
/// Canonical hands: 13 pairs + 78 suited combos + 78 offsuit combos.
/// Index layout: pairs first (AA=0..22=12), then suited (AKs=13..),
/// then offsuit (...=91..168).
#[must_use]
pub fn canonical_hand_index(holding: [Card; 2]) -> u16 {
    let r1 = rank_ordinal(holding[0].value);
    let r2 = rank_ordinal(holding[1].value);
    let (high, low) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };
    let suited = holding[0].suit == holding[1].suit;

    if high == low {
        // Pair: index 0..12 (A=0, K=1, ..., 2=12)
        u16::from(high)
    } else if suited {
        // Suited: 13 pairs already used, then upper triangle
        // For (high, low) with high > low: index = 13 + triangle_offset(high, low)
        13 + triangle_index(high, low)
    } else {
        // Offsuit: 13 + 78 suited = 91, then same triangle
        91 + triangle_index(high, low)
    }
}

/// Map a canonical hand string (e.g. "AKs", "QQ", "72o") to its index.
///
/// Returns `None` if the string is not a valid canonical hand.
#[must_use]
pub fn canonical_hand_index_from_str(hand: &str) -> Option<u16> {
    let chars: Vec<char> = hand.chars().collect();
    if chars.len() < 2 || chars.len() > 3 {
        return None;
    }

    let r1 = rank_ordinal_from_char(chars[0])?;
    let r2 = rank_ordinal_from_char(chars[1])?;
    let (high, low) = if r1 >= r2 { (r1, r2) } else { (r2, r1) };

    if high == low {
        Some(u16::from(high))
    } else {
        let suited = chars.get(2) == Some(&'s');
        if suited {
            Some(13 + triangle_index(high, low))
        } else {
            Some(91 + triangle_index(high, low))
        }
    }
}

/// Upper-triangle index for (high, low) where high > low.
///
/// Enumerates pairs (high, low) with high in 0..12, low in 0..high.
/// For high=1,low=0 → 0; high=2,low=0 → 1; high=2,low=1 → 2; ...
fn triangle_index(high: u8, low: u8) -> u16 {
    // Number of pairs before row `high` = high*(high-1)/2
    let base = u16::from(high) * (u16::from(high) - 1) / 2;
    base + u16::from(low)
}

/// Map card rank to ordinal: A=0, K=1, Q=2, ..., 2=12.
fn rank_ordinal(value: Value) -> u8 {
    match value {
        Value::Ace => 0,
        Value::King => 1,
        Value::Queen => 2,
        Value::Jack => 3,
        Value::Ten => 4,
        Value::Nine => 5,
        Value::Eight => 6,
        Value::Seven => 7,
        Value::Six => 8,
        Value::Five => 9,
        Value::Four => 10,
        Value::Three => 11,
        Value::Two => 12,
    }
}

/// Map a rank character to its ordinal.
fn rank_ordinal_from_char(c: char) -> Option<u8> {
    match c {
        'A' => Some(0),
        'K' => Some(1),
        'Q' => Some(2),
        'J' => Some(3),
        'T' => Some(4),
        '9' => Some(5),
        '8' => Some(6),
        '7' => Some(7),
        '6' => Some(8),
        '5' => Some(9),
        '4' => Some(10),
        '3' => Some(11),
        '2' => Some(12),
        _ => None,
    }
}

/// Map a rank character and suited flag to representative cards.
///
/// Uses Spade for first card; second card is same suit if suited,
/// Heart if offsuit. For pairs, uses Spade+Heart.
#[must_use]
pub fn cards_from_rank_chars(rank1: char, rank2: char, suited: bool) -> Option<[Card; 2]> {
    let v1 = value_from_char(rank1)?;
    let v2 = value_from_char(rank2)?;
    let suit2 = if suited { Suit::Spade } else { Suit::Heart };
    Some([Card::new(v1, Suit::Spade), Card::new(v2, suit2)])
}

fn value_from_char(c: char) -> Option<Value> {
    match c {
        'A' => Some(Value::Ace),
        'K' => Some(Value::King),
        'Q' => Some(Value::Queen),
        'J' => Some(Value::Jack),
        'T' => Some(Value::Ten),
        '9' => Some(Value::Nine),
        '8' => Some(Value::Eight),
        '7' => Some(Value::Seven),
        '6' => Some(Value::Six),
        '5' => Some(Value::Five),
        '4' => Some(Value::Four),
        '3' => Some(Value::Three),
        '2' => Some(Value::Two),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_macros::timed_test;

    #[timed_test]
    fn round_trip_key_components() {
        let key = InfoKey::new(42, 2, 15, 9, &[1, 3, 5]);
        assert_eq!(key.pot_bucket(), 15);
        assert_eq!(key.stack_bucket(), 9);
    }

    #[timed_test]
    fn with_buckets_replaces_correctly() {
        let key = InfoKey::new(42, 1, 10, 5, &[2, 3]);
        let modified = key.with_buckets(20, 8);
        assert_eq!(modified.pot_bucket(), 20);
        assert_eq!(modified.stack_bucket(), 8);
        // Hand and street bits should be unchanged
        assert_eq!(
            key.as_u64() >> HAND_SHIFT,
            modified.as_u64() >> HAND_SHIFT,
        );
    }

    #[timed_test]
    fn all_169_canonical_hands_unique() {
        let mut seen = std::collections::HashSet::new();
        let values = [
            Value::Ace,
            Value::King,
            Value::Queen,
            Value::Jack,
            Value::Ten,
            Value::Nine,
            Value::Eight,
            Value::Seven,
            Value::Six,
            Value::Five,
            Value::Four,
            Value::Three,
            Value::Two,
        ];

        let mut count = 0;
        for (i, &v1) in values.iter().enumerate() {
            for &v2 in &values[i..] {
                if v1 == v2 {
                    // Pair
                    let idx = canonical_hand_index([
                        Card::new(v1, Suit::Spade),
                        Card::new(v2, Suit::Heart),
                    ]);
                    assert!(seen.insert(idx), "Duplicate index {idx} for pair {v1:?}");
                    count += 1;
                } else {
                    // Suited
                    let idx_s = canonical_hand_index([
                        Card::new(v1, Suit::Spade),
                        Card::new(v2, Suit::Spade),
                    ]);
                    assert!(
                        seen.insert(idx_s),
                        "Duplicate index {idx_s} for suited {v1:?}{v2:?}"
                    );
                    count += 1;

                    // Offsuit
                    let idx_o = canonical_hand_index([
                        Card::new(v1, Suit::Spade),
                        Card::new(v2, Suit::Heart),
                    ]);
                    assert!(
                        seen.insert(idx_o),
                        "Duplicate index {idx_o} for offsuit {v1:?}{v2:?}"
                    );
                    count += 1;
                }
            }
        }

        assert_eq!(count, 169);
        assert_eq!(seen.len(), 169);
        // All indices should be in 0..169
        for &idx in &seen {
            assert!(idx < 169, "Index {idx} out of range");
        }
    }

    #[timed_test]
    fn canonical_hand_index_symmetry() {
        // AKs should give the same index regardless of card order
        let idx1 = canonical_hand_index([
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ]);
        let idx2 = canonical_hand_index([
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Ace, Suit::Spade),
        ]);
        assert_eq!(idx1, idx2);
    }

    #[timed_test]
    fn canonical_hand_index_from_str_examples() {
        assert_eq!(canonical_hand_index_from_str("AA"), Some(0));
        assert_eq!(canonical_hand_index_from_str("KK"), Some(1));
        assert_eq!(canonical_hand_index_from_str("22"), Some(12));
        assert_eq!(
            canonical_hand_index_from_str("AKs"),
            Some(13)
        );
        assert_eq!(
            canonical_hand_index_from_str("AKo"),
            Some(91)
        );
    }

    #[timed_test]
    fn canonical_hand_index_str_matches_card_index() {
        // AKs from string should match AKs from cards
        let from_str = canonical_hand_index_from_str("AKs").unwrap();
        let from_cards = canonical_hand_index([
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
        ]);
        assert_eq!(from_str, from_cards);

        // QQ from string should match QQ from cards
        let from_str = canonical_hand_index_from_str("QQ").unwrap();
        let from_cards = canonical_hand_index([
            Card::new(Value::Queen, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
        ]);
        assert_eq!(from_str, from_cards);
    }

    #[timed_test]
    fn encode_action_covers_all_variants() {
        assert_eq!(encode_action(Action::Fold), 1);
        assert_eq!(encode_action(Action::Check), 2);
        assert_eq!(encode_action(Action::Call), 3);
        assert_eq!(encode_action(Action::Bet(0)), 4);
        assert_eq!(encode_action(Action::Bet(1)), 5);
        assert_eq!(encode_action(Action::Bet(2)), 6);
        assert_eq!(encode_action(Action::Bet(3)), 7);
        assert_eq!(encode_action(Action::Raise(0)), 8);
        assert_eq!(encode_action(Action::Raise(1)), 9);
        assert_eq!(encode_action(Action::Raise(2)), 10);
        assert_eq!(encode_action(Action::Raise(3)), 11);
        assert_eq!(encode_action(Action::Bet(ALL_IN)), 12);
        assert_eq!(encode_action(Action::Raise(ALL_IN)), 13);
    }

    #[timed_test]
    fn different_streets_produce_different_keys() {
        let k1 = InfoKey::new(0, 0, 0, 0, &[]);
        let k2 = InfoKey::new(0, 1, 0, 0, &[]);
        assert_ne!(k1.as_u64(), k2.as_u64());
    }

    #[timed_test]
    fn different_actions_produce_different_keys() {
        let k1 = InfoKey::new(0, 0, 0, 0, &[1]);
        let k2 = InfoKey::new(0, 0, 0, 0, &[2]);
        assert_ne!(k1.as_u64(), k2.as_u64());
    }

    #[timed_test]
    fn action_packing_order_matters() {
        let k1 = InfoKey::new(0, 0, 0, 0, &[1, 2]);
        let k2 = InfoKey::new(0, 0, 0, 0, &[2, 1]);
        assert_ne!(k1.as_u64(), k2.as_u64());
    }

    #[timed_test]
    fn actions_do_not_overlap_with_stack_bucket() {
        // Verify that setting all action bits to max doesn't corrupt stack_bucket.
        let key = InfoKey::new(0, 0, 0, 15, &[15, 15, 15, 15, 15, 15]);
        assert_eq!(key.stack_bucket(), 15, "Stack bucket corrupted by action bits");

        // Verify that max stack_bucket doesn't corrupt first action
        let k1 = InfoKey::new(0, 0, 0, 15, &[1]);
        let k2 = InfoKey::new(0, 0, 0, 15, &[2]);
        assert_ne!(k1.as_u64(), k2.as_u64(), "Actions indistinguishable with max stack_bucket");
    }

    #[timed_test]
    fn invalid_str_returns_none() {
        assert_eq!(canonical_hand_index_from_str(""), None);
        assert_eq!(canonical_hand_index_from_str("X"), None);
        assert_eq!(canonical_hand_index_from_str("AAAA"), None);
    }
}
