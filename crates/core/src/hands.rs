//! 169 canonical preflop hand representations.
//!
//! In Hold'em, there are 1326 possible 2-card combinations, but only 169
//! strategically distinct hand types:
//! - 13 pocket pairs (`AA`, `KK`, ..., `22`)
//! - 78 suited hands (`AKs`, `AQs`, ..., `32s`)
//! - 78 off-suit hands (`AKo`, `AQo`, ..., `32o`)

use std::fmt;

use crate::poker::{Card, Suit, Value};

/// The four suits in a fixed order for combo enumeration.
const ALL_SUITS: [Suit; 4] = [Suit::Spade, Suit::Heart, Suit::Diamond, Suit::Club];

/// A canonical preflop hand category.
///
/// Represents one of the 169 strategically distinct preflop hands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalHand {
    /// Higher value (or equal for pairs)
    high: Value,
    /// Lower value (or equal for pairs)
    low: Value,
    /// Hand type: pair, suited, or offsuit
    hand_type: HandType,
}

/// The type of canonical hand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HandType {
    /// Pocket pair (e.g., `AA`, `KK`)
    Pair,
    /// Suited non-pair (e.g., `AKs`)
    Suited,
    /// Off-suit non-pair (e.g., `AKo`)
    Offsuit,
}

impl CanonicalHand {
    /// Create a new canonical hand.
    ///
    /// Automatically orders values so high >= low.
    #[must_use]
    pub fn new(value1: Value, value2: Value, suited: bool) -> Self {
        let (high, low) = if value1 >= value2 {
            (value1, value2)
        } else {
            (value2, value1)
        };

        let hand_type = if high == low {
            HandType::Pair
        } else if suited {
            HandType::Suited
        } else {
            HandType::Offsuit
        };

        Self {
            high,
            low,
            hand_type,
        }
    }

    /// Create a canonical hand from two cards.
    #[must_use]
    pub fn from_cards(card1: Card, card2: Card) -> Self {
        let suited = card1.suit == card2.suit;
        Self::new(card1.value, card2.value, suited)
    }

    /// Get the higher value.
    #[must_use]
    pub fn high_value(&self) -> Value {
        self.high
    }

    /// Get the lower value.
    #[must_use]
    pub fn low_value(&self) -> Value {
        self.low
    }

    /// Get the hand type.
    #[must_use]
    pub fn hand_type(&self) -> HandType {
        self.hand_type
    }

    /// Check if this is a pocket pair.
    #[must_use]
    pub fn is_pair(&self) -> bool {
        self.hand_type == HandType::Pair
    }

    /// Check if this is suited.
    #[must_use]
    pub fn is_suited(&self) -> bool {
        self.hand_type == HandType::Suited
    }

    /// Get the number of combinations this hand represents.
    ///
    /// - Pairs: 6 combos (4 choose 2)
    /// - Suited: 4 combos (one per suit)
    /// - Offsuit: 12 combos (4 * 3)
    #[must_use]
    pub fn num_combos(&self) -> u8 {
        match self.hand_type {
            HandType::Pair => 6,
            HandType::Suited => 4,
            HandType::Offsuit => 12,
        }
    }

    /// Enumerate all specific 2-card combos for this canonical hand.
    ///
    /// - Pairs: 6 combos (all C(4,2) suit pairs)
    /// - Suited: 4 combos (one per suit)
    /// - Offsuit: 12 combos (all suit1 != suit2)
    #[must_use]
    pub fn combos(&self) -> Vec<(Card, Card)> {
        match self.hand_type {
            HandType::Pair => {
                let mut out = Vec::with_capacity(6);
                for (i, &s1) in ALL_SUITS.iter().enumerate() {
                    for &s2 in &ALL_SUITS[i + 1..] {
                        out.push((Card::new(self.high, s1), Card::new(self.high, s2)));
                    }
                }
                out
            }
            HandType::Suited => ALL_SUITS
                .iter()
                .map(|&s| (Card::new(self.high, s), Card::new(self.low, s)))
                .collect(),
            HandType::Offsuit => {
                let mut out = Vec::with_capacity(12);
                for &s1 in &ALL_SUITS {
                    for &s2 in &ALL_SUITS {
                        if s1 != s2 {
                            out.push((Card::new(self.high, s1), Card::new(self.low, s2)));
                        }
                    }
                }
                out
            }
        }
    }

    /// Get the index of this hand in the standard 169 ordering.
    ///
    /// Ordering: pairs first (AA=0 to 22=12), then suited, then offsuit.
    #[must_use]
    pub fn index(&self) -> usize {
        let high_idx = value_to_index(self.high);
        let low_idx = value_to_index(self.low);

        match self.hand_type {
            HandType::Pair => high_idx,
            HandType::Suited => {
                // 13 pairs + suited hands
                let mut idx = 13; // Skip pairs
                for h in 0..high_idx {
                    idx += 12 - h; // Number of suited hands with high value h
                }
                idx + (low_idx - high_idx - 1)
            }
            HandType::Offsuit => {
                // 13 pairs + 78 suited + offsuit hands
                let mut idx = 13 + 78; // Skip pairs and suited
                for h in 0..high_idx {
                    idx += 12 - h; // Number of offsuit hands with high value h
                }
                idx + (low_idx - high_idx - 1)
            }
        }
    }

    /// Create a canonical hand from its index (0-168).
    #[must_use]
    pub fn from_index(index: usize) -> Option<Self> {
        if index >= 169 {
            return None;
        }

        if index < 13 {
            // Pair
            let value = index_to_value(index);
            Some(Self::new(value, value, false))
        } else if index < 13 + 78 {
            // Suited
            let suited_idx = index - 13;
            let (high_idx, low_idx) = linear_index_to_hand_indices(suited_idx);
            Some(Self::new(
                index_to_value(high_idx),
                index_to_value(low_idx),
                true,
            ))
        } else {
            // Offsuit
            let offsuit_idx = index - 13 - 78;
            let (high_idx, low_idx) = linear_index_to_hand_indices(offsuit_idx);
            Some(Self::new(
                index_to_value(high_idx),
                index_to_value(low_idx),
                false,
            ))
        }
    }

    /// Get the matrix position for this hand.
    ///
    /// Returns (row, col) where:
    /// - Pairs are on the diagonal
    /// - Suited hands are above the diagonal (row < col)
    /// - Offsuit hands are below the diagonal (row > col)
    #[must_use]
    pub fn matrix_position(&self) -> (usize, usize) {
        let high_idx = value_to_index(self.high);
        let low_idx = value_to_index(self.low);

        match self.hand_type {
            HandType::Pair => (high_idx, high_idx),
            HandType::Suited => (high_idx, low_idx), // Above diagonal
            HandType::Offsuit => (low_idx, high_idx), // Below diagonal
        }
    }

    /// Create a canonical hand from matrix position.
    #[must_use]
    pub fn from_matrix_position(row: usize, col: usize) -> Option<Self> {
        if row >= 13 || col >= 13 {
            return None;
        }

        match row.cmp(&col) {
            std::cmp::Ordering::Equal => {
                let value = index_to_value(row);
                Some(Self::new(value, value, false))
            }
            std::cmp::Ordering::Less => {
                // Above diagonal = suited
                Some(Self::new(index_to_value(row), index_to_value(col), true))
            }
            std::cmp::Ordering::Greater => {
                // Below diagonal = offsuit
                Some(Self::new(index_to_value(col), index_to_value(row), false))
            }
        }
    }

    /// Parse a canonical hand from string notation.
    ///
    /// Accepts formats like `AKs`, `AKo`, `AA`, `72o`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The string length is not 2-3 characters
    /// - The value characters are invalid
    /// - A non-pair hand is missing the 's' or 'o' suffix
    /// - A pair is marked as suited
    pub fn parse(s: &str) -> Result<Self, ParseHandError> {
        let s = s.trim();
        if s.len() < 2 || s.len() > 3 {
            return Err(ParseHandError::InvalidLength);
        }

        let chars: Vec<char> = s.chars().collect();
        let value1 = Value::from_char(chars[0]).ok_or(ParseHandError::InvalidValue(chars[0]))?;
        let value2 = Value::from_char(chars[1]).ok_or(ParseHandError::InvalidValue(chars[1]))?;

        if s.len() == 2 {
            // Must be a pair
            if value1 != value2 {
                return Err(ParseHandError::MissingSuitedness);
            }
            Ok(Self::new(value1, value2, false))
        } else {
            let suited = match chars[2] {
                's' | 'S' => true,
                'o' | 'O' => false,
                c => return Err(ParseHandError::InvalidSuitedness(c)),
            };

            if value1 == value2 && suited {
                return Err(ParseHandError::PairCannotBeSuited);
            }
            Ok(Self::new(value1, value2, suited))
        }
    }
}

impl fmt::Display for CanonicalHand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let high = self.high.to_char();
        let low = self.low.to_char();

        match self.hand_type {
            HandType::Pair => write!(f, "{high}{low}"),
            HandType::Suited => write!(f, "{high}{low}s"),
            HandType::Offsuit => write!(f, "{high}{low}o"),
        }
    }
}

/// Error parsing a canonical hand string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseHandError {
    /// Invalid string length
    InvalidLength,
    /// Invalid value character
    InvalidValue(char),
    /// Non-pair hand missing 's' or 'o' suffix
    MissingSuitedness,
    /// Invalid suitedness character
    InvalidSuitedness(char),
    /// Pairs cannot be suited
    PairCannotBeSuited,
}

impl fmt::Display for ParseHandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength => write!(f, "hand must be 2-3 characters"),
            Self::InvalidValue(c) => write!(f, "invalid value: {c}"),
            Self::MissingSuitedness => write!(f, "non-pair hands must end with 's' or 'o'"),
            Self::InvalidSuitedness(c) => {
                write!(f, "invalid suitedness: {c} (expected 's' or 'o')")
            }
            Self::PairCannotBeSuited => write!(f, "pairs cannot be suited"),
        }
    }
}

impl std::error::Error for ParseHandError {}

/// Iterator over all 169 canonical hands.
pub struct AllHands {
    index: usize,
}

impl AllHands {
    /// Create a new iterator over all canonical hands.
    #[must_use]
    pub fn new() -> Self {
        Self { index: 0 }
    }
}

impl Default for AllHands {
    fn default() -> Self {
        Self::new()
    }
}

impl Iterator for AllHands {
    type Item = CanonicalHand;

    fn next(&mut self) -> Option<Self::Item> {
        let hand = CanonicalHand::from_index(self.index)?;
        self.index += 1;
        Some(hand)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = 169 - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for AllHands {}

/// Get an iterator over all 169 canonical hands.
#[must_use]
pub fn all_hands() -> AllHands {
    AllHands::new()
}

// Helper functions for value/index conversion

/// Convert Value to index (Ace=0, King=1, ..., Two=12)
fn value_to_index(value: Value) -> usize {
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

/// Convert index to Value (0=Ace, 1=King, ..., 12=Two)
fn index_to_value(index: usize) -> Value {
    match index {
        0 => Value::Ace,
        1 => Value::King,
        2 => Value::Queen,
        3 => Value::Jack,
        4 => Value::Ten,
        5 => Value::Nine,
        6 => Value::Eight,
        7 => Value::Seven,
        8 => Value::Six,
        9 => Value::Five,
        10 => Value::Four,
        11 => Value::Three,
        _ => Value::Two,
    }
}

/// Convert a linear index to `(high_idx, low_idx)` for non-pair hands.
fn linear_index_to_hand_indices(linear_idx: usize) -> (usize, usize) {
    // For suited/offsuit hands, we iterate: (0,1), (0,2), ..., (0,12), (1,2), (1,3), ...
    let mut idx = 0;
    for high in 0..12 {
        let count = 12 - high;
        if idx + count > linear_idx {
            let low = high + 1 + (linear_idx - idx);
            return (high, low);
        }
        idx += count;
    }
    (11, 12) // Should never reach here for valid indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::Suit;
    use test_macros::timed_test;

    #[timed_test]
    fn test_all_hands_count() {
        assert_eq!(all_hands().count(), 169);
    }

    #[timed_test]
    fn test_pairs_count() {
        let pairs: Vec<_> = all_hands().filter(super::CanonicalHand::is_pair).collect();
        assert_eq!(pairs.len(), 13);
    }

    #[timed_test]
    fn test_suited_count() {
        let suited: Vec<_> = all_hands()
            .filter(super::CanonicalHand::is_suited)
            .collect();
        assert_eq!(suited.len(), 78);
    }

    #[timed_test]
    fn test_offsuit_count() {
        let offsuit: Vec<_> = all_hands()
            .filter(|h| !h.is_pair() && !h.is_suited())
            .collect();
        assert_eq!(offsuit.len(), 78);
    }

    #[timed_test]
    fn test_total_combos() {
        let total: u32 = all_hands().map(|h| u32::from(h.num_combos())).sum();
        assert_eq!(total, 1326); // 52 choose 2
    }

    #[timed_test]
    fn test_parse_pairs() {
        let aa = CanonicalHand::parse("AA").unwrap();
        assert!(aa.is_pair());
        assert_eq!(aa.high_value(), Value::Ace);
        assert_eq!(aa.to_string(), "AA");

        let twos = CanonicalHand::parse("22").unwrap();
        assert!(twos.is_pair());
        assert_eq!(twos.high_value(), Value::Two);
    }

    #[timed_test]
    fn test_parse_suited() {
        let aks = CanonicalHand::parse("AKs").unwrap();
        assert!(aks.is_suited());
        assert_eq!(aks.high_value(), Value::Ace);
        assert_eq!(aks.low_value(), Value::King);
        assert_eq!(aks.to_string(), "AKs");
    }

    #[timed_test]
    fn test_parse_offsuit() {
        let ako = CanonicalHand::parse("AKo").unwrap();
        assert!(!ako.is_suited());
        assert!(!ako.is_pair());
        assert_eq!(ako.to_string(), "AKo");
    }

    #[timed_test]
    fn test_parse_reversed_order() {
        let ka = CanonicalHand::parse("KAs").unwrap();
        assert_eq!(ka.high_value(), Value::Ace);
        assert_eq!(ka.low_value(), Value::King);
    }

    #[timed_test]
    fn test_index_roundtrip() {
        for hand in all_hands() {
            let idx = hand.index();
            let recovered = CanonicalHand::from_index(idx).unwrap();
            assert_eq!(hand, recovered, "Index {idx} failed roundtrip");
        }
    }

    #[timed_test]
    fn test_matrix_position_roundtrip() {
        for hand in all_hands() {
            let (row, col) = hand.matrix_position();
            let recovered = CanonicalHand::from_matrix_position(row, col).unwrap();
            assert_eq!(hand, recovered);
        }
    }

    #[timed_test]
    fn test_matrix_diagonal_is_pairs() {
        for i in 0..13 {
            let hand = CanonicalHand::from_matrix_position(i, i).unwrap();
            assert!(hand.is_pair());
        }
    }

    #[timed_test]
    fn test_matrix_upper_triangle_is_suited() {
        for row in 0..13 {
            for col in (row + 1)..13 {
                let hand = CanonicalHand::from_matrix_position(row, col).unwrap();
                assert!(hand.is_suited(), "({row}, {col}) should be suited");
            }
        }
    }

    #[timed_test]
    fn test_matrix_lower_triangle_is_offsuit() {
        for row in 1..13 {
            for col in 0..row {
                let hand = CanonicalHand::from_matrix_position(row, col).unwrap();
                assert!(
                    !hand.is_pair() && !hand.is_suited(),
                    "({row}, {col}) should be offsuit"
                );
            }
        }
    }

    #[timed_test]
    fn test_from_cards() {
        let ace_spades = Card::new(Value::Ace, Suit::Spade);
        let king_spades = Card::new(Value::King, Suit::Spade);
        let king_hearts = Card::new(Value::King, Suit::Heart);

        let aks = CanonicalHand::from_cards(ace_spades, king_spades);
        assert!(aks.is_suited());
        assert_eq!(aks.to_string(), "AKs");

        let ako = CanonicalHand::from_cards(ace_spades, king_hearts);
        assert!(!ako.is_suited());
        assert_eq!(ako.to_string(), "AKo");
    }

    #[timed_test]
    fn test_num_combos() {
        assert_eq!(CanonicalHand::parse("AA").unwrap().num_combos(), 6);
        assert_eq!(CanonicalHand::parse("AKs").unwrap().num_combos(), 4);
        assert_eq!(CanonicalHand::parse("AKo").unwrap().num_combos(), 12);
    }

    #[timed_test]
    fn combos_pair_returns_six() {
        let aa = CanonicalHand::parse("AA").unwrap();
        let combos = aa.combos();
        assert_eq!(combos.len(), 6);
        // All combos should have Ace value
        for (c1, c2) in &combos {
            assert_eq!(c1.value, Value::Ace);
            assert_eq!(c2.value, Value::Ace);
            assert_ne!(c1.suit, c2.suit);
        }
    }

    #[timed_test]
    fn combos_suited_returns_four() {
        let aks = CanonicalHand::parse("AKs").unwrap();
        let combos = aks.combos();
        assert_eq!(combos.len(), 4);
        for (c1, c2) in &combos {
            assert_eq!(c1.value, Value::Ace);
            assert_eq!(c2.value, Value::King);
            assert_eq!(c1.suit, c2.suit);
        }
    }

    #[timed_test]
    fn combos_offsuit_returns_twelve() {
        let ako = CanonicalHand::parse("AKo").unwrap();
        let combos = ako.combos();
        assert_eq!(combos.len(), 12);
        for (c1, c2) in &combos {
            assert_eq!(c1.value, Value::Ace);
            assert_eq!(c2.value, Value::King);
            assert_ne!(c1.suit, c2.suit);
        }
    }

    #[timed_test]
    fn combos_no_duplicates() {
        for hand in all_hands() {
            let combos = hand.combos();
            for (i, a) in combos.iter().enumerate() {
                for b in &combos[i + 1..] {
                    assert_ne!(a, b, "duplicate combo in {hand}");
                }
            }
        }
    }
}
