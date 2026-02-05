use crate::poker::{Card, Suit};

/// Canonical suit ordering (used for isomorphism)
/// Spade=0, Heart=1, Diamond=2, Club=3
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CanonicalSuit {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
}

impl CanonicalSuit {
    #[must_use]
    pub fn to_suit(self) -> Suit {
        match self {
            CanonicalSuit::First => Suit::Spade,
            CanonicalSuit::Second => Suit::Heart,
            CanonicalSuit::Third => Suit::Diamond,
            CanonicalSuit::Fourth => Suit::Club,
        }
    }
}

/// Mapping from original suits to canonical suits
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SuitMapping {
    mapping: [CanonicalSuit; 4], // indexed by original Suit
}

impl SuitMapping {
    /// Create identity mapping
    #[must_use]
    pub fn identity() -> Self {
        Self {
            mapping: [
                CanonicalSuit::First,  // Spade -> First
                CanonicalSuit::Second, // Heart -> Second
                CanonicalSuit::Third,  // Diamond -> Third
                CanonicalSuit::Fourth, // Club -> Fourth
            ],
        }
    }

    /// Map a suit through this mapping
    #[must_use]
    pub fn map_suit(&self, suit: Suit) -> Suit {
        let idx = suit_to_index(suit);
        self.mapping[idx].to_suit()
    }

    /// Map a card through this mapping
    #[must_use]
    pub fn map_card(&self, card: Card) -> Card {
        Card::new(card.value, self.map_suit(card.suit))
    }
}

/// Convert Suit to array index
fn suit_to_index(suit: Suit) -> usize {
    match suit {
        Suit::Spade => 0,
        Suit::Heart => 1,
        Suit::Diamond => 2,
        Suit::Club => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::Value;

    #[test]
    fn identity_mapping_preserves_suits() {
        let mapping = SuitMapping::identity();
        assert_eq!(mapping.map_suit(Suit::Spade), Suit::Spade);
        assert_eq!(mapping.map_suit(Suit::Heart), Suit::Heart);
        assert_eq!(mapping.map_suit(Suit::Diamond), Suit::Diamond);
        assert_eq!(mapping.map_suit(Suit::Club), Suit::Club);
    }

    #[test]
    fn identity_mapping_preserves_cards() {
        let mapping = SuitMapping::identity();
        let card = Card::new(Value::Ace, Suit::Heart);
        let mapped = mapping.map_card(card);
        assert_eq!(mapped.value, Value::Ace);
        assert_eq!(mapped.suit, Suit::Heart);
    }
}
