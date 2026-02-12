use crate::abstraction::AbstractionError;
use crate::card_utils::value_rank;
use crate::poker::{Card, Suit, Value};

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


/// Compare two slices of values lexicographically by rank
fn compare_value_slices(a: &[Value], b: &[Value]) -> std::cmp::Ordering {
    for (va, vb) in a.iter().zip(b.iter()) {
        let cmp = value_rank(*va).cmp(&value_rank(*vb));
        if cmp != std::cmp::Ordering::Equal {
            return cmp;
        }
    }
    a.len().cmp(&b.len())
}

/// A board in canonical suit form
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalBoard {
    pub cards: Vec<Card>,
    pub mapping: SuitMapping,
}

impl CanonicalBoard {
    /// Canonicalize a board by reordering suits
    ///
    /// Rules:
    /// 1. Suits ordered by frequency on board (most common first)
    /// 2. Ties broken by highest card in that suit
    /// 3. Further ties broken by second-highest, etc.
    ///
    /// # Errors
    /// Returns `AbstractionError::InvalidBoardSize` if board is empty or has more than 5 cards.
    pub fn from_cards(board: &[Card]) -> Result<Self, AbstractionError> {
        if board.is_empty() || board.len() > 5 {
            return Err(AbstractionError::InvalidBoardSize {
                expected: 3,
                got: board.len(),
            });
        }

        // Collect cards by suit
        let mut suit_cards: [Vec<Value>; 4] = Default::default();
        for card in board {
            let idx = suit_to_index(card.suit);
            suit_cards[idx].push(card.value);
        }

        // Sort cards within each suit (descending by value)
        for cards in &mut suit_cards {
            cards.sort_by_key(|v| std::cmp::Reverse(value_rank(*v)));
        }

        // Create suit priority list: (original_suit_idx, count, highest_cards)
        let mut suit_priority: Vec<(usize, usize, &[Value])> = (0..4)
            .map(|i| (i, suit_cards[i].len(), suit_cards[i].as_slice()))
            .collect();

        // Sort by: count desc, then by card ranks desc (lexicographic)
        suit_priority.sort_by(|a, b| {
            match b.1.cmp(&a.1) {
                std::cmp::Ordering::Equal => {
                    // Compare cards lexicographically (highest first)
                    compare_value_slices(b.2, a.2)
                }
                other => other,
            }
        });

        // Build the mapping: suit_priority[0] -> First, [1] -> Second, etc.
        let mut mapping_arr = [CanonicalSuit::First; 4];
        for (canonical_idx, (original_idx, _, _)) in suit_priority.iter().enumerate() {
            let canonical = match canonical_idx {
                0 => CanonicalSuit::First,
                1 => CanonicalSuit::Second,
                2 => CanonicalSuit::Third,
                _ => CanonicalSuit::Fourth,
            };
            mapping_arr[*original_idx] = canonical;
        }

        let mapping = SuitMapping {
            mapping: mapping_arr,
        };

        // Apply mapping to board
        let canonical_cards: Vec<Card> = board.iter().map(|c| mapping.map_card(*c)).collect();

        Ok(Self {
            cards: canonical_cards,
            mapping,
        })
    }

    /// Map a holding (two hole cards) using the same suit mapping
    #[must_use]
    pub fn canonicalize_holding(&self, card1: Card, card2: Card) -> (Card, Card) {
        (self.mapping.map_card(card1), self.mapping.map_card(card2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::Value;
    use test_macros::timed_test;

    #[timed_test]
    fn identity_mapping_preserves_suits() {
        let mapping = SuitMapping::identity();
        assert_eq!(mapping.map_suit(Suit::Spade), Suit::Spade);
        assert_eq!(mapping.map_suit(Suit::Heart), Suit::Heart);
        assert_eq!(mapping.map_suit(Suit::Diamond), Suit::Diamond);
        assert_eq!(mapping.map_suit(Suit::Club), Suit::Club);
    }

    #[timed_test]
    fn identity_mapping_preserves_cards() {
        let mapping = SuitMapping::identity();
        let card = Card::new(Value::Ace, Suit::Heart);
        let mapped = mapping.map_card(card);
        assert_eq!(mapped.value, Value::Ace);
        assert_eq!(mapped.suit, Suit::Heart);
    }

    #[timed_test]
    fn canonicalize_rainbow_flop_orders_by_highest_card() {
        // Ah Kd 7c -> all different suits, order by highest card per suit
        let board = vec![
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Seven, Suit::Club),
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        // Canonical board should be As Kh 7d
        assert_eq!(canonical.cards[0], Card::new(Value::Ace, Suit::Spade));
        assert_eq!(canonical.cards[1], Card::new(Value::King, Suit::Heart));
        assert_eq!(canonical.cards[2], Card::new(Value::Seven, Suit::Diamond));
    }

    #[timed_test]
    fn canonicalize_monotone_flop() {
        // All hearts: Ah Kh 7h
        let board = vec![
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Heart),
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        // All should be spades (first canonical suit)
        assert_eq!(canonical.cards[0].suit, Suit::Spade);
        assert_eq!(canonical.cards[1].suit, Suit::Spade);
        assert_eq!(canonical.cards[2].suit, Suit::Spade);
    }

    #[timed_test]
    fn canonicalize_two_tone_flop() {
        // Ah Kh 7c - two hearts, one club
        let board = vec![
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Club),
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        assert_eq!(canonical.cards[0], Card::new(Value::Ace, Suit::Spade));
        assert_eq!(canonical.cards[1], Card::new(Value::King, Suit::Spade));
        assert_eq!(canonical.cards[2], Card::new(Value::Seven, Suit::Heart));
    }

    #[timed_test]
    fn isomorphic_boards_same_canonical() {
        // Ah Kd 7c and As Kh 7d should produce same canonical form
        let board1 = vec![
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Seven, Suit::Club),
        ];
        let board2 = vec![
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Heart),
            Card::new(Value::Seven, Suit::Diamond),
        ];

        let canonical1 = CanonicalBoard::from_cards(&board1).unwrap();
        let canonical2 = CanonicalBoard::from_cards(&board2).unwrap();

        assert_eq!(canonical1.cards, canonical2.cards);
    }

    #[timed_test]
    fn holding_mapped_consistently_with_board() {
        let board = vec![
            Card::new(Value::Ace, Suit::Heart),
            Card::new(Value::King, Suit::Diamond),
            Card::new(Value::Seven, Suit::Club),
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        // Holding with hearts should map to spades (same as Ace)
        let (c1, c2) = canonical.canonicalize_holding(
            Card::new(Value::Queen, Suit::Heart),
            Card::new(Value::Jack, Suit::Heart),
        );

        assert_eq!(c1.suit, Suit::Spade);
        assert_eq!(c2.suit, Suit::Spade);
    }
}
