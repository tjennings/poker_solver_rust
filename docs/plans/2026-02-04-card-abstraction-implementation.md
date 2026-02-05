# Card Abstraction System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a card abstraction system for full HUNL poker with EHS2-based bucketing and suit isomorphism reduction.

**Architecture:** Three modules (isomorphism, hand_strength, buckets) with a unified CardAbstraction API. Offline bucket boundary generation via exhaustive enumeration, runtime O(log n) bucket lookup.

**Tech Stack:** Rust, rs_poker (card types, hand evaluation), thiserror (errors), serde (serialization)

---

## Task 1: Module Scaffolding and Error Types

**Files:**
- Create: `crates/core/src/abstraction/mod.rs`
- Create: `crates/core/src/abstraction/error.rs`
- Modify: `crates/core/src/lib.rs`

**Step 1: Create the abstraction module directory**

Run: `mkdir -p crates/core/src/abstraction`

**Step 2: Write error types with failing compilation test**

Create `crates/core/src/abstraction/error.rs`:

```rust
use crate::poker::Card;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AbstractionError {
    #[error("Invalid board size: expected {expected} cards, got {got}")]
    InvalidBoardSize { expected: usize, got: usize },

    #[error("Duplicate card: {0}")]
    DuplicateCard(Card),

    #[error("Failed to load abstraction: {0}")]
    LoadError(#[from] std::io::Error),

    #[error("Invalid boundary data: {reason}")]
    InvalidBoundaries { reason: String },

    #[error("Serialization error: {0}")]
    SerializationError(String),
}
```

**Step 3: Create module file**

Create `crates/core/src/abstraction/mod.rs`:

```rust
mod error;

pub use error::AbstractionError;

/// Street in poker (determines bucket count and calculation)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Street {
    Flop,
    Turn,
    River,
}

impl Street {
    /// Determine street from board card count
    pub fn from_board_len(len: usize) -> Result<Self, AbstractionError> {
        match len {
            3 => Ok(Street::Flop),
            4 => Ok(Street::Turn),
            5 => Ok(Street::River),
            n => Err(AbstractionError::InvalidBoardSize {
                expected: 3,
                got: n,
            }),
        }
    }
}
```

**Step 4: Register module in lib.rs**

Add to `crates/core/src/lib.rs` after other module declarations:

```rust
pub mod abstraction;
```

**Step 5: Verify compilation**

Run: `cargo build -p poker-solver-core`
Expected: Successful compilation

**Step 6: Commit**

```bash
git add crates/core/src/abstraction/ crates/core/src/lib.rs
git commit -m "feat(abstraction): add module scaffolding and error types"
```

---

## Task 2: Suit Type and Basic Isomorphism Types

**Files:**
- Create: `crates/core/src/abstraction/isomorphism.rs`
- Modify: `crates/core/src/abstraction/mod.rs`

**Step 1: Write failing test for suit ordering**

Create `crates/core/src/abstraction/isomorphism.rs`:

```rust
use crate::poker::{Card, Suit, Value};

/// Canonical suit ordering (used for isomorphism)
/// Spades=0, Hearts=1, Diamonds=2, Clubs=3
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum CanonicalSuit {
    First = 0,
    Second = 1,
    Third = 2,
    Fourth = 3,
}

impl CanonicalSuit {
    pub fn to_suit(self) -> Suit {
        match self {
            CanonicalSuit::First => Suit::Spades,
            CanonicalSuit::Second => Suit::Hearts,
            CanonicalSuit::Third => Suit::Diamonds,
            CanonicalSuit::Fourth => Suit::Clubs,
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
    pub fn identity() -> Self {
        Self {
            mapping: [
                CanonicalSuit::First,  // Spades -> First
                CanonicalSuit::Second, // Hearts -> Second
                CanonicalSuit::Third,  // Diamonds -> Third
                CanonicalSuit::Fourth, // Clubs -> Fourth
            ],
        }
    }

    /// Map a suit through this mapping
    pub fn map_suit(&self, suit: Suit) -> Suit {
        let idx = suit_to_index(suit);
        self.mapping[idx].to_suit()
    }

    /// Map a card through this mapping
    pub fn map_card(&self, card: Card) -> Card {
        Card {
            value: card.value,
            suit: self.map_suit(card.suit),
        }
    }
}

/// Convert Suit to array index
fn suit_to_index(suit: Suit) -> usize {
    match suit {
        Suit::Spades => 0,
        Suit::Hearts => 1,
        Suit::Diamonds => 2,
        Suit::Clubs => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_mapping_preserves_suits() {
        let mapping = SuitMapping::identity();
        assert_eq!(mapping.map_suit(Suit::Spades), Suit::Spades);
        assert_eq!(mapping.map_suit(Suit::Hearts), Suit::Hearts);
        assert_eq!(mapping.map_suit(Suit::Diamonds), Suit::Diamonds);
        assert_eq!(mapping.map_suit(Suit::Clubs), Suit::Clubs);
    }

    #[test]
    fn identity_mapping_preserves_cards() {
        let mapping = SuitMapping::identity();
        let card = Card {
            value: Value::Ace,
            suit: Suit::Hearts,
        };
        let mapped = mapping.map_card(card);
        assert_eq!(mapped.value, Value::Ace);
        assert_eq!(mapped.suit, Suit::Hearts);
    }
}
```

**Step 2: Register module and run tests**

Add to `crates/core/src/abstraction/mod.rs`:

```rust
mod isomorphism;

pub use isomorphism::{CanonicalSuit, SuitMapping};
```

Run: `cargo test -p poker-solver-core isomorphism -- --nocapture`
Expected: 2 tests pass

**Step 3: Commit**

```bash
git add crates/core/src/abstraction/
git commit -m "feat(abstraction): add suit mapping types for isomorphism"
```

---

## Task 3: Board Canonicalization

**Files:**
- Modify: `crates/core/src/abstraction/isomorphism.rs`

**Step 1: Write failing test for canonicalization**

Add to `crates/core/src/abstraction/isomorphism.rs` tests:

```rust
    #[test]
    fn canonicalize_rainbow_flop_orders_by_highest_card() {
        // Ah Kd 7c -> all different suits, order by highest card per suit
        // A(hearts) > K(diamonds) > 7(clubs)
        // So: hearts->spades, diamonds->hearts, clubs->diamonds
        let board = vec![
            Card { value: Value::Ace, suit: Suit::Hearts },
            Card { value: Value::King, suit: Suit::Diamonds },
            Card { value: Value::Seven, suit: Suit::Clubs },
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        // Canonical board should be As Kh 7d
        assert_eq!(canonical.cards[0], Card { value: Value::Ace, suit: Suit::Spades });
        assert_eq!(canonical.cards[1], Card { value: Value::King, suit: Suit::Hearts });
        assert_eq!(canonical.cards[2], Card { value: Value::Seven, suit: Suit::Diamonds });
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core canonicalize_rainbow -- --nocapture`
Expected: FAIL with "cannot find struct `CanonicalBoard`"

**Step 3: Implement CanonicalBoard**

Add to `crates/core/src/abstraction/isomorphism.rs` (before tests module):

```rust
use crate::abstraction::AbstractionError;

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
            cards.sort_by(|a, b| value_rank(*b).cmp(&value_rank(*a)));
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

        let mapping = SuitMapping { mapping: mapping_arr };

        // Apply mapping to board
        let canonical_cards: Vec<Card> = board.iter().map(|c| mapping.map_card(*c)).collect();

        Ok(Self {
            cards: canonical_cards,
            mapping,
        })
    }

    /// Map a holding (two hole cards) using the same suit mapping
    pub fn canonicalize_holding(&self, card1: Card, card2: Card) -> (Card, Card) {
        (self.mapping.map_card(card1), self.mapping.map_card(card2))
    }
}

/// Get numeric rank for value comparison (Ace=14, King=13, ..., Two=2)
fn value_rank(v: Value) -> u8 {
    match v {
        Value::Ace => 14,
        Value::King => 13,
        Value::Queen => 12,
        Value::Jack => 11,
        Value::Ten => 10,
        Value::Nine => 9,
        Value::Eight => 8,
        Value::Seven => 7,
        Value::Six => 6,
        Value::Five => 5,
        Value::Four => 4,
        Value::Three => 3,
        Value::Two => 2,
    }
}

/// Compare two slices of values lexicographically by rank
fn compare_value_slices(a: &[Value], b: &[Value]) -> std::cmp::Ordering {
    for (va, vb) in a.iter().zip(b.iter()) {
        match value_rank(*va).cmp(&value_rank(*vb)) {
            std::cmp::Ordering::Equal => continue,
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}
```

**Step 4: Export CanonicalBoard**

Update `crates/core/src/abstraction/mod.rs`:

```rust
pub use isomorphism::{CanonicalBoard, CanonicalSuit, SuitMapping};
```

**Step 5: Run test to verify it passes**

Run: `cargo test -p poker-solver-core canonicalize_rainbow -- --nocapture`
Expected: PASS

**Step 6: Add more canonicalization tests**

Add to tests module:

```rust
    #[test]
    fn canonicalize_monotone_flop() {
        // All hearts: Ah Kh 7h
        // Hearts has 3 cards, others have 0
        // Hearts -> First (spades)
        let board = vec![
            Card { value: Value::Ace, suit: Suit::Hearts },
            Card { value: Value::King, suit: Suit::Hearts },
            Card { value: Value::Seven, suit: Suit::Hearts },
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        // All should be spades
        assert_eq!(canonical.cards[0].suit, Suit::Spades);
        assert_eq!(canonical.cards[1].suit, Suit::Spades);
        assert_eq!(canonical.cards[2].suit, Suit::Spades);
    }

    #[test]
    fn canonicalize_two_tone_flop() {
        // Ah Kh 7c - two hearts, one club
        // Hearts (2 cards) -> First (spades)
        // Clubs (1 card) -> Second (hearts)
        let board = vec![
            Card { value: Value::Ace, suit: Suit::Hearts },
            Card { value: Value::King, suit: Suit::Hearts },
            Card { value: Value::Seven, suit: Suit::Clubs },
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        assert_eq!(canonical.cards[0], Card { value: Value::Ace, suit: Suit::Spades });
        assert_eq!(canonical.cards[1], Card { value: Value::King, suit: Suit::Spades });
        assert_eq!(canonical.cards[2], Card { value: Value::Seven, suit: Suit::Hearts });
    }

    #[test]
    fn isomorphic_boards_same_canonical() {
        // Ah Kd 7c and As Kh 7d should produce same canonical form
        let board1 = vec![
            Card { value: Value::Ace, suit: Suit::Hearts },
            Card { value: Value::King, suit: Suit::Diamonds },
            Card { value: Value::Seven, suit: Suit::Clubs },
        ];
        let board2 = vec![
            Card { value: Value::Ace, suit: Suit::Spades },
            Card { value: Value::King, suit: Suit::Hearts },
            Card { value: Value::Seven, suit: Suit::Diamonds },
        ];

        let canonical1 = CanonicalBoard::from_cards(&board1).unwrap();
        let canonical2 = CanonicalBoard::from_cards(&board2).unwrap();

        assert_eq!(canonical1.cards, canonical2.cards);
    }

    #[test]
    fn holding_mapped_consistently_with_board() {
        let board = vec![
            Card { value: Value::Ace, suit: Suit::Hearts },
            Card { value: Value::King, suit: Suit::Diamonds },
            Card { value: Value::Seven, suit: Suit::Clubs },
        ];

        let canonical = CanonicalBoard::from_cards(&board).unwrap();

        // Holding with hearts should map to spades (same as Ace)
        let (c1, c2) = canonical.canonicalize_holding(
            Card { value: Value::Queen, suit: Suit::Hearts },
            Card { value: Value::Jack, suit: Suit::Hearts },
        );

        assert_eq!(c1.suit, Suit::Spades);
        assert_eq!(c2.suit, Suit::Spades);
    }
```

**Step 7: Run all isomorphism tests**

Run: `cargo test -p poker-solver-core isomorphism -- --nocapture`
Expected: All 6 tests pass

**Step 8: Commit**

```bash
git add crates/core/src/abstraction/
git commit -m "feat(abstraction): implement board canonicalization for suit isomorphism"
```

---

## Task 4: Hand Strength Types and EHS Calculation

**Files:**
- Create: `crates/core/src/abstraction/hand_strength.rs`
- Modify: `crates/core/src/abstraction/mod.rs`

**Step 1: Write failing test for basic EHS**

Create `crates/core/src/abstraction/hand_strength.rs`:

```rust
use crate::poker::{Card, Hand, Rankable, Suit, Value};

/// Hand strength metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandStrength {
    /// Expected Hand Strength - equity vs random opponent now
    pub ehs: f32,
    /// Positive potential - P(behind now, ahead after runout)
    pub ppot: f32,
    /// Negative potential - P(ahead now, behind after runout)
    pub npot: f32,
    /// EHS2 = EHS + (1-EHS)*PPot - EHS*NPot
    pub ehs2: f32,
}

impl HandStrength {
    pub fn new(ehs: f32, ppot: f32, npot: f32) -> Self {
        let ehs2 = ehs + (1.0 - ehs) * ppot - ehs * npot;
        Self { ehs, ppot, npot, ehs2 }
    }

    /// River hand strength (no potential, just EHS)
    pub fn river(ehs: f32) -> Self {
        Self { ehs, ppot: 0.0, npot: 0.0, ehs2: ehs }
    }
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
}
```

**Step 2: Register module and run tests**

Add to `crates/core/src/abstraction/mod.rs`:

```rust
mod hand_strength;

pub use hand_strength::HandStrength;
```

Run: `cargo test -p poker-solver-core hand_strength -- --nocapture`
Expected: 2 tests pass

**Step 3: Commit**

```bash
git add crates/core/src/abstraction/
git commit -m "feat(abstraction): add HandStrength type with EHS2 formula"
```

---

## Task 5: River EHS Calculator

**Files:**
- Modify: `crates/core/src/abstraction/hand_strength.rs`

**Step 1: Write failing test for river EHS**

Add to tests in `hand_strength.rs`:

```rust
    #[test]
    fn river_ehs_nut_flush_near_one() {
        // As Ks on 2s 5s 8s Tc Qd - nut flush
        let board = vec![
            Card { value: Value::Two, suit: Suit::Spades },
            Card { value: Value::Five, suit: Suit::Spades },
            Card { value: Value::Eight, suit: Suit::Spades },
            Card { value: Value::Ten, suit: Suit::Clubs },
            Card { value: Value::Queen, suit: Suit::Diamonds },
        ];
        let holding = (
            Card { value: Value::Ace, suit: Suit::Spades },
            Card { value: Value::King, suit: Suit::Spades },
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
            Card { value: Value::Ace, suit: Suit::Spades },
            Card { value: Value::King, suit: Suit::Spades },
            Card { value: Value::Queen, suit: Suit::Spades },
            Card { value: Value::Jack, suit: Suit::Spades },
            Card { value: Value::Nine, suit: Suit::Diamonds },
        ];
        let holding = (
            Card { value: Value::Seven, suit: Suit::Hearts },
            Card { value: Value::Two, suit: Suit::Clubs },
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_river(&board, holding);

        // Weak hand on scary board should have low equity
        assert!(hs.ehs < 0.30, "Expected EHS < 0.30, got {}", hs.ehs);
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core river_ehs -- --nocapture`
Expected: FAIL with "cannot find struct `HandStrengthCalculator`"

**Step 3: Implement HandStrengthCalculator for river**

Add to `hand_strength.rs` before tests:

```rust
use std::collections::HashSet;

/// Calculator for hand strength metrics
pub struct HandStrengthCalculator;

impl HandStrengthCalculator {
    pub fn new() -> Self {
        Self
    }

    /// Calculate EHS on the river (exhaustive enumeration)
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
        let our_hand = Hand::new_with_cards(our_cards.clone());
        let our_rank = our_hand.rank();

        // Enumerate all opponent holdings
        let mut wins = 0u32;
        let mut ties = 0u32;
        let mut losses = 0u32;

        for opp1_val in all_values() {
            for opp1_suit in all_suits() {
                let opp1 = Card { value: opp1_val, suit: opp1_suit };
                if dead.contains(&opp1) {
                    continue;
                }

                for opp2_val in all_values() {
                    for opp2_suit in all_suits() {
                        let opp2 = Card { value: opp2_val, suit: opp2_suit };
                        if dead.contains(&opp2) || opp2 == opp1 {
                            continue;
                        }
                        // Avoid double-counting (opp1, opp2) and (opp2, opp1)
                        if (opp1_val as u8, opp1_suit as u8) >= (opp2_val as u8, opp2_suit as u8) {
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
            }
        }

        let total = wins + ties + losses;
        let ehs = if total > 0 {
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

fn all_values() -> impl Iterator<Item = Value> {
    [
        Value::Two, Value::Three, Value::Four, Value::Five, Value::Six,
        Value::Seven, Value::Eight, Value::Nine, Value::Ten,
        Value::Jack, Value::Queen, Value::King, Value::Ace,
    ].into_iter()
}

fn all_suits() -> impl Iterator<Item = Suit> {
    [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs].into_iter()
}
```

**Step 4: Export calculator**

Update `crates/core/src/abstraction/mod.rs`:

```rust
pub use hand_strength::{HandStrength, HandStrengthCalculator};
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core river_ehs -- --nocapture`
Expected: 2 tests pass

**Step 6: Commit**

```bash
git add crates/core/src/abstraction/
git commit -m "feat(abstraction): implement river EHS calculation via exhaustive enumeration"
```

---

## Task 6: Turn EHS2 Calculator (with river runout)

**Files:**
- Modify: `crates/core/src/abstraction/hand_strength.rs`

**Step 1: Write failing test for turn EHS2**

Add to tests:

```rust
    #[test]
    fn turn_ehs2_flush_draw_higher_than_ehs() {
        // Flush draw: As 5s on 2s 8s Tc Qd (4 to flush)
        let board = vec![
            Card { value: Value::Two, suit: Suit::Spades },
            Card { value: Value::Eight, suit: Suit::Spades },
            Card { value: Value::Ten, suit: Suit::Clubs },
            Card { value: Value::Queen, suit: Suit::Diamonds },
        ];
        let holding = (
            Card { value: Value::Ace, suit: Suit::Spades },
            Card { value: Value::Five, suit: Suit::Spades },
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_turn(&board, holding);

        // Flush draw should have positive potential
        assert!(hs.ppot > 0.1, "Expected PPot > 0.1, got {}", hs.ppot);
        assert!(hs.ehs2 > hs.ehs, "Expected EHS2 > EHS for flush draw");
    }

    #[test]
    fn turn_ehs2_made_hand_npot() {
        // Top pair on draw-heavy board: Ah Kc on As 8s 7s 2d
        let board = vec![
            Card { value: Value::Ace, suit: Suit::Spades },
            Card { value: Value::Eight, suit: Suit::Spades },
            Card { value: Value::Seven, suit: Suit::Spades },
            Card { value: Value::Two, suit: Suit::Diamonds },
        ];
        let holding = (
            Card { value: Value::Ace, suit: Suit::Hearts },
            Card { value: Value::King, suit: Suit::Clubs },
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_turn(&board, holding);

        // Made hand on flush board should have negative potential
        assert!(hs.npot > 0.05, "Expected NPot > 0.05, got {}", hs.npot);
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core turn_ehs2 -- --nocapture`
Expected: FAIL with "no method named `calculate_turn`"

**Step 3: Implement calculate_turn**

Add to `HandStrengthCalculator` impl:

```rust
    /// Calculate EHS2 on the turn (enumerate river cards)
    pub fn calculate_turn(&self, board: &[Card], holding: (Card, Card)) -> HandStrength {
        let (h1, h2) = holding;

        // Build set of dead cards
        let mut dead: HashSet<Card> = HashSet::new();
        dead.insert(h1);
        dead.insert(h2);
        for &card in board {
            dead.insert(card);
        }

        // Counters for EHS
        let mut total_ahead = 0u64;
        let mut total_tied = 0u64;
        let mut total_behind = 0u64;

        // Counters for potential
        let mut ahead_stays_ahead = 0u64;
        let mut ahead_falls_behind = 0u64;
        let mut behind_moves_ahead = 0u64;
        let mut behind_stays_behind = 0u64;
        let mut tied_count = 0u64;

        // Enumerate all opponent holdings
        for opp1 in all_cards() {
            if dead.contains(&opp1) {
                continue;
            }
            for opp2 in all_cards() {
                if dead.contains(&opp2) || opp2 <= opp1 {
                    continue;
                }

                // Current hand comparison (on turn)
                let mut our_turn: Vec<Card> = board.to_vec();
                our_turn.push(h1);
                our_turn.push(h2);
                let our_turn_rank = Hand::new_with_cards(our_turn.clone()).rank();

                let mut opp_turn: Vec<Card> = board.to_vec();
                opp_turn.push(opp1);
                opp_turn.push(opp2);
                let opp_turn_rank = Hand::new_with_cards(opp_turn.clone()).rank();

                let currently_ahead = our_turn_rank > opp_turn_rank;
                let currently_behind = our_turn_rank < opp_turn_rank;
                let currently_tied = our_turn_rank == opp_turn_rank;

                if currently_ahead {
                    total_ahead += 1;
                } else if currently_behind {
                    total_behind += 1;
                } else {
                    total_tied += 1;
                }

                // Enumerate river cards
                for river in all_cards() {
                    if dead.contains(&river) || river == opp1 || river == opp2 {
                        continue;
                    }

                    // Final hand comparison
                    let mut our_river = our_turn.clone();
                    our_river.push(river);
                    let our_river_rank = Hand::new_with_cards(our_river).rank();

                    let mut opp_river = opp_turn.clone();
                    opp_river.push(river);
                    let opp_river_rank = Hand::new_with_cards(opp_river).rank();

                    let finally_ahead = our_river_rank > opp_river_rank;
                    let finally_behind = our_river_rank < opp_river_rank;

                    if currently_ahead {
                        if finally_ahead || our_river_rank == opp_river_rank {
                            ahead_stays_ahead += 1;
                        } else {
                            ahead_falls_behind += 1;
                        }
                    } else if currently_behind {
                        if finally_ahead || our_river_rank == opp_river_rank {
                            behind_moves_ahead += 1;
                        } else {
                            behind_stays_behind += 1;
                        }
                    } else {
                        tied_count += 1;
                    }
                }
            }
        }

        let total = total_ahead + total_tied + total_behind;
        let ehs = if total > 0 {
            (total_ahead as f32 + total_tied as f32 / 2.0) / total as f32
        } else {
            0.5
        };

        let ppot = if total_behind > 0 {
            let behind_outcomes = behind_moves_ahead + behind_stays_behind;
            if behind_outcomes > 0 {
                behind_moves_ahead as f32 / behind_outcomes as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        let npot = if total_ahead > 0 {
            let ahead_outcomes = ahead_stays_ahead + ahead_falls_behind;
            if ahead_outcomes > 0 {
                ahead_falls_behind as f32 / ahead_outcomes as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        HandStrength::new(ehs, ppot, npot)
    }
```

Also add helper function:

```rust
fn all_cards() -> impl Iterator<Item = Card> {
    all_values().flat_map(|v| all_suits().map(move |s| Card { value: v, suit: s }))
}

impl PartialOrd for Card {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Card {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self.value as u8).cmp(&(other.value as u8)) {
            std::cmp::Ordering::Equal => (self.suit as u8).cmp(&(other.suit as u8)),
            other => other,
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core turn_ehs2 -- --nocapture`
Expected: 2 tests pass

**Step 5: Commit**

```bash
git add crates/core/src/abstraction/
git commit -m "feat(abstraction): implement turn EHS2 calculation with river enumeration"
```

---

## Task 7: Flop EHS2 Calculator (with turn+river runout)

**Files:**
- Modify: `crates/core/src/abstraction/hand_strength.rs`

**Step 1: Write failing test for flop EHS2**

Add to tests:

```rust
    #[test]
    fn flop_ehs2_open_ended_straight_draw() {
        // 9h 8h on 7c 6d 2s - open-ended straight draw
        let board = vec![
            Card { value: Value::Seven, suit: Suit::Clubs },
            Card { value: Value::Six, suit: Suit::Diamonds },
            Card { value: Value::Two, suit: Suit::Spades },
        ];
        let holding = (
            Card { value: Value::Nine, suit: Suit::Hearts },
            Card { value: Value::Eight, suit: Suit::Hearts },
        );

        let calc = HandStrengthCalculator::new();
        let hs = calc.calculate_flop(&board, holding);

        // Straight draw should have significant positive potential
        assert!(hs.ppot > 0.15, "Expected PPot > 0.15 for OESD, got {}", hs.ppot);
        assert!(hs.ehs2 > hs.ehs, "Expected EHS2 > EHS for draw");
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core flop_ehs2 -- --nocapture`
Expected: FAIL with "no method named `calculate_flop`"

**Step 3: Implement calculate_flop**

Add to `HandStrengthCalculator` impl:

```rust
    /// Calculate EHS2 on the flop (enumerate turn+river cards)
    /// This is the most expensive calculation (~1M evaluations per holding)
    pub fn calculate_flop(&self, board: &[Card], holding: (Card, Card)) -> HandStrength {
        let (h1, h2) = holding;

        // Build set of dead cards
        let mut dead: HashSet<Card> = HashSet::new();
        dead.insert(h1);
        dead.insert(h2);
        for &card in board {
            dead.insert(card);
        }

        // Counters for EHS
        let mut total_ahead = 0u64;
        let mut total_tied = 0u64;
        let mut total_behind = 0u64;

        // Counters for potential
        let mut ahead_stays_ahead = 0u64;
        let mut ahead_falls_behind = 0u64;
        let mut behind_moves_ahead = 0u64;
        let mut behind_stays_behind = 0u64;

        // Enumerate all opponent holdings
        for opp1 in all_cards() {
            if dead.contains(&opp1) {
                continue;
            }
            for opp2 in all_cards() {
                if dead.contains(&opp2) || opp2 <= opp1 {
                    continue;
                }

                // Current hand comparison (on flop)
                let mut our_flop: Vec<Card> = board.to_vec();
                our_flop.push(h1);
                our_flop.push(h2);
                let our_flop_rank = Hand::new_with_cards(our_flop.clone()).rank();

                let mut opp_flop: Vec<Card> = board.to_vec();
                opp_flop.push(opp1);
                opp_flop.push(opp2);
                let opp_flop_rank = Hand::new_with_cards(opp_flop.clone()).rank();

                let currently_ahead = our_flop_rank > opp_flop_rank;
                let currently_behind = our_flop_rank < opp_flop_rank;

                if currently_ahead {
                    total_ahead += 1;
                } else if currently_behind {
                    total_behind += 1;
                } else {
                    total_tied += 1;
                }

                // Enumerate turn cards
                for turn in all_cards() {
                    if dead.contains(&turn) || turn == opp1 || turn == opp2 {
                        continue;
                    }

                    // Enumerate river cards
                    for river in all_cards() {
                        if dead.contains(&river) || river == opp1 || river == opp2 || river == turn {
                            continue;
                        }
                        if river <= turn {
                            continue; // Avoid counting same runout twice
                        }

                        // Final hand comparison
                        let mut our_river = our_flop.clone();
                        our_river.push(turn);
                        our_river.push(river);
                        let our_river_rank = Hand::new_with_cards(our_river).rank();

                        let mut opp_river = opp_flop.clone();
                        opp_river.push(turn);
                        opp_river.push(river);
                        let opp_river_rank = Hand::new_with_cards(opp_river).rank();

                        let finally_ahead = our_river_rank > opp_river_rank;

                        if currently_ahead {
                            if finally_ahead || our_river_rank == opp_river_rank {
                                ahead_stays_ahead += 1;
                            } else {
                                ahead_falls_behind += 1;
                            }
                        } else if currently_behind {
                            if finally_ahead || our_river_rank == opp_river_rank {
                                behind_moves_ahead += 1;
                            } else {
                                behind_stays_behind += 1;
                            }
                        }
                    }
                }
            }
        }

        let total = total_ahead + total_tied + total_behind;
        let ehs = if total > 0 {
            (total_ahead as f32 + total_tied as f32 / 2.0) / total as f32
        } else {
            0.5
        };

        let ppot = if total_behind > 0 {
            let behind_outcomes = behind_moves_ahead + behind_stays_behind;
            if behind_outcomes > 0 {
                behind_moves_ahead as f32 / behind_outcomes as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        let npot = if total_ahead > 0 {
            let ahead_outcomes = ahead_stays_ahead + ahead_falls_behind;
            if ahead_outcomes > 0 {
                ahead_falls_behind as f32 / ahead_outcomes as f32
            } else {
                0.0
            }
        } else {
            0.0
        };

        HandStrength::new(ehs, ppot, npot)
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core flop_ehs2 -- --nocapture --ignored`
(Note: This test is slow, ~30 seconds)

Expected: PASS

**Step 5: Mark flop test as ignored for normal runs**

Update test to add `#[ignore]` attribute for CI:

```rust
    #[test]
    #[ignore] // Slow: ~30 seconds
    fn flop_ehs2_open_ended_straight_draw() {
```

**Step 6: Commit**

```bash
git add crates/core/src/abstraction/
git commit -m "feat(abstraction): implement flop EHS2 calculation with turn+river enumeration"
```

---

## Task 8: Bucket Boundaries and Assignment

**Files:**
- Create: `crates/core/src/abstraction/buckets.rs`
- Modify: `crates/core/src/abstraction/mod.rs`

**Step 1: Write failing test for bucket boundaries**

Create `crates/core/src/abstraction/buckets.rs`:

```rust
use crate::abstraction::{AbstractionError, Street};
use serde::{Deserialize, Serialize};

/// Bucket boundaries for each street
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BucketBoundaries {
    pub flop: Vec<f32>,
    pub turn: Vec<f32>,
    pub river: Vec<f32>,
}

impl BucketBoundaries {
    /// Create boundaries from sorted EHS2 samples (percentile-based)
    pub fn from_samples(
        flop_samples: &mut [f32],
        turn_samples: &mut [f32],
        river_samples: &mut [f32],
        flop_buckets: usize,
        turn_buckets: usize,
        river_buckets: usize,
    ) -> Self {
        Self {
            flop: compute_percentile_boundaries(flop_samples, flop_buckets),
            turn: compute_percentile_boundaries(turn_samples, turn_buckets),
            river: compute_percentile_boundaries(river_samples, river_buckets),
        }
    }
}

fn compute_percentile_boundaries(samples: &mut [f32], num_buckets: usize) -> Vec<f32> {
    if samples.is_empty() || num_buckets == 0 {
        return vec![];
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut boundaries = Vec::with_capacity(num_buckets - 1);
    for i in 1..num_buckets {
        let idx = (i * samples.len()) / num_buckets;
        let idx = idx.min(samples.len() - 1);
        boundaries.push(samples[idx]);
    }

    boundaries
}

/// Assigns EHS2 values to bucket indices
#[derive(Debug, Clone)]
pub struct BucketAssigner {
    boundaries: BucketBoundaries,
}

impl BucketAssigner {
    pub fn new(boundaries: BucketBoundaries) -> Self {
        Self { boundaries }
    }

    /// Get bucket index for an EHS2 value on a given street
    /// O(log n) via binary search
    pub fn get_bucket(&self, street: Street, ehs2: f32) -> u16 {
        let boundaries = match street {
            Street::Flop => &self.boundaries.flop,
            Street::Turn => &self.boundaries.turn,
            Street::River => &self.boundaries.river,
        };

        boundaries.partition_point(|&b| b < ehs2) as u16
    }

    pub fn num_buckets(&self, street: Street) -> usize {
        match street {
            Street::Flop => self.boundaries.flop.len() + 1,
            Street::Turn => self.boundaries.turn.len() + 1,
            Street::River => self.boundaries.river.len() + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_boundaries_uniform_distribution() {
        let mut samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let boundaries = compute_percentile_boundaries(&mut samples, 10);

        // Should have 9 boundaries for 10 buckets
        assert_eq!(boundaries.len(), 9);

        // Boundaries should be roughly 0.1, 0.2, ..., 0.9
        for (i, &b) in boundaries.iter().enumerate() {
            let expected = (i + 1) as f32 / 10.0;
            assert!((b - expected).abs() < 0.02, "Boundary {} expected ~{}, got {}", i, expected, b);
        }
    }

    #[test]
    fn bucket_assignment_correct() {
        let mut samples: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let boundaries = BucketBoundaries {
            flop: compute_percentile_boundaries(&mut samples.clone(), 10),
            turn: compute_percentile_boundaries(&mut samples.clone(), 10),
            river: compute_percentile_boundaries(&mut samples, 10),
        };
        let assigner = BucketAssigner::new(boundaries);

        // EHS2 = 0.05 should be bucket 0
        assert_eq!(assigner.get_bucket(Street::Flop, 0.05), 0);

        // EHS2 = 0.15 should be bucket 1
        assert_eq!(assigner.get_bucket(Street::Flop, 0.15), 1);

        // EHS2 = 0.95 should be bucket 9
        assert_eq!(assigner.get_bucket(Street::Flop, 0.95), 9);
    }

    #[test]
    fn extreme_values_boundary_buckets() {
        let boundaries = BucketBoundaries {
            flop: vec![0.2, 0.4, 0.6, 0.8],
            turn: vec![0.2, 0.4, 0.6, 0.8],
            river: vec![0.2, 0.4, 0.6, 0.8],
        };
        let assigner = BucketAssigner::new(boundaries);

        // EHS2 = 0.0 should be bucket 0
        assert_eq!(assigner.get_bucket(Street::River, 0.0), 0);

        // EHS2 = 1.0 should be last bucket (4)
        assert_eq!(assigner.get_bucket(Street::River, 1.0), 4);
    }
}
```

**Step 2: Register module and run tests**

Add to `crates/core/src/abstraction/mod.rs`:

```rust
mod buckets;

pub use buckets::{BucketAssigner, BucketBoundaries};
```

Run: `cargo test -p poker-solver-core buckets -- --nocapture`
Expected: 3 tests pass

**Step 3: Commit**

```bash
git add crates/core/src/abstraction/
git commit -m "feat(abstraction): add bucket boundaries and assignment"
```

---

## Task 9: CardAbstraction Public API

**Files:**
- Modify: `crates/core/src/abstraction/mod.rs`

**Step 1: Write failing test for CardAbstraction**

Add to `crates/core/src/abstraction/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::poker::{Card, Suit, Value};

    #[test]
    fn card_abstraction_isomorphic_hands_same_bucket() {
        // Create simple boundaries for testing
        let boundaries = BucketBoundaries {
            flop: (1..100).map(|i| i as f32 / 100.0).collect(),
            turn: (1..100).map(|i| i as f32 / 100.0).collect(),
            river: (1..100).map(|i| i as f32 / 100.0).collect(),
        };
        let abstraction = CardAbstraction::from_boundaries(boundaries);

        // Two isomorphic situations should get same bucket
        // Board: Ah Kd 7c with holding Qs Qc
        let board1 = vec![
            Card { value: Value::Ace, suit: Suit::Hearts },
            Card { value: Value::King, suit: Suit::Diamonds },
            Card { value: Value::Seven, suit: Suit::Clubs },
        ];
        let holding1 = (
            Card { value: Value::Queen, suit: Suit::Spades },
            Card { value: Value::Queen, suit: Suit::Clubs },
        );

        // Isomorphic: As Kh 7d with holding Qc Qd
        let board2 = vec![
            Card { value: Value::Ace, suit: Suit::Spades },
            Card { value: Value::King, suit: Suit::Hearts },
            Card { value: Value::Seven, suit: Suit::Diamonds },
        ];
        let holding2 = (
            Card { value: Value::Queen, suit: Suit::Clubs },
            Card { value: Value::Queen, suit: Suit::Diamonds },
        );

        let bucket1 = abstraction.get_bucket(&board1, holding1).unwrap();
        let bucket2 = abstraction.get_bucket(&board2, holding2).unwrap();

        // Note: Due to isomorphism, both should get same EHS2 and thus same bucket
        // (This test verifies the pipeline, actual equality depends on calc accuracy)
        assert_eq!(bucket1, bucket2, "Isomorphic hands should get same bucket");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core card_abstraction -- --nocapture`
Expected: FAIL with "cannot find struct `CardAbstraction`"

**Step 3: Implement CardAbstraction**

Add to `crates/core/src/abstraction/mod.rs` (before tests):

```rust
use crate::poker::Card;
use std::path::Path;

/// Configuration for abstraction generation
#[derive(Debug, Clone)]
pub struct AbstractionConfig {
    pub flop_buckets: u16,
    pub turn_buckets: u16,
    pub river_buckets: u16,
    pub samples_per_street: u32,
}

impl Default for AbstractionConfig {
    fn default() -> Self {
        Self {
            flop_buckets: 5_000,
            turn_buckets: 5_000,
            river_buckets: 20_000,
            samples_per_street: 100_000,
        }
    }
}

/// Main entry point for card abstraction
pub struct CardAbstraction {
    assigner: BucketAssigner,
    calculator: HandStrengthCalculator,
}

impl CardAbstraction {
    /// Create from precomputed boundaries
    pub fn from_boundaries(boundaries: BucketBoundaries) -> Self {
        Self {
            assigner: BucketAssigner::new(boundaries),
            calculator: HandStrengthCalculator::new(),
        }
    }

    /// Load boundaries from file
    pub fn load(path: &Path) -> Result<Self, AbstractionError> {
        let data = std::fs::read(path)?;
        let boundaries: BucketBoundaries = rmp_serde::from_slice(&data)
            .map_err(|e| AbstractionError::SerializationError(e.to_string()))?;
        Ok(Self::from_boundaries(boundaries))
    }

    /// Save boundaries to file
    pub fn save(&self, path: &Path) -> Result<(), AbstractionError> {
        let data = rmp_serde::to_vec(&self.assigner.boundaries())
            .map_err(|e| AbstractionError::SerializationError(e.to_string()))?;
        std::fs::write(path, data)?;
        Ok(())
    }

    /// Get bucket for a hand on a board
    pub fn get_bucket(&self, board: &[Card], holding: (Card, Card)) -> Result<u16, AbstractionError> {
        let street = Street::from_board_len(board.len())?;

        // Check for duplicate cards
        let mut seen = std::collections::HashSet::new();
        for &card in board {
            if !seen.insert(card) {
                return Err(AbstractionError::DuplicateCard(card));
            }
        }
        if !seen.insert(holding.0) {
            return Err(AbstractionError::DuplicateCard(holding.0));
        }
        if !seen.insert(holding.1) {
            return Err(AbstractionError::DuplicateCard(holding.1));
        }

        // Canonicalize board and holding
        let canonical_board = CanonicalBoard::from_cards(board)?;
        let canonical_holding = canonical_board.canonicalize_holding(holding.0, holding.1);

        // Calculate EHS2
        let hs = match street {
            Street::Flop => self.calculator.calculate_flop(&canonical_board.cards, canonical_holding),
            Street::Turn => self.calculator.calculate_turn(&canonical_board.cards, canonical_holding),
            Street::River => self.calculator.calculate_river(&canonical_board.cards, canonical_holding),
        };

        Ok(self.assigner.get_bucket(street, hs.ehs2))
    }

    /// Get number of buckets for a street
    pub fn num_buckets(&self, street: Street) -> usize {
        self.assigner.num_buckets(street)
    }
}
```

Also add to `BucketAssigner`:

```rust
    pub fn boundaries(&self) -> &BucketBoundaries {
        &self.boundaries
    }
```

**Step 4: Add rmp-serde dependency**

Run: `cd /Users/ltj/Documents/code/poker_solver_rust && cargo add rmp-serde -p poker-solver-core`

**Step 5: Run test to verify it passes**

Run: `cargo test -p poker-solver-core card_abstraction -- --nocapture --ignored`
(Note: Test is slow due to flop EHS2 calculation)

Expected: PASS

**Step 6: Mark test as ignored**

```rust
    #[test]
    #[ignore] // Slow: involves EHS2 calculation
    fn card_abstraction_isomorphic_hands_same_bucket() {
```

**Step 7: Commit**

```bash
git add crates/core/src/abstraction/ crates/core/Cargo.toml
git commit -m "feat(abstraction): add CardAbstraction public API with load/save"
```

---

## Task 10: Boundary Generation CLI

**Files:**
- Create: `crates/core/src/abstraction/generator.rs`
- Modify: `crates/core/src/abstraction/mod.rs`

**Step 1: Create boundary generator**

Create `crates/core/src/abstraction/generator.rs`:

```rust
use crate::abstraction::{
    AbstractionConfig, BucketBoundaries, HandStrengthCalculator, Street,
};
use crate::poker::{Card, Suit, Value};
use rand::prelude::*;
use rand::SeedableRng;

/// Generates bucket boundaries by sampling hands
pub struct BoundaryGenerator {
    config: AbstractionConfig,
    calculator: HandStrengthCalculator,
}

impl BoundaryGenerator {
    pub fn new(config: AbstractionConfig) -> Self {
        Self {
            config,
            calculator: HandStrengthCalculator::new(),
        }
    }

    /// Generate boundaries by sampling random boards and holdings
    /// This is expensive - meant to be run offline once
    pub fn generate(&self, seed: u64) -> BucketBoundaries {
        let mut rng = StdRng::seed_from_u64(seed);

        println!("Generating river samples...");
        let mut river_samples = self.sample_street(&mut rng, 5, self.config.samples_per_street);

        println!("Generating turn samples...");
        let mut turn_samples = self.sample_street(&mut rng, 4, self.config.samples_per_street);

        println!("Generating flop samples (slow)...");
        // Use fewer samples for flop due to computation cost
        let flop_sample_count = (self.config.samples_per_street / 100).max(100);
        let mut flop_samples = self.sample_street(&mut rng, 3, flop_sample_count);

        BucketBoundaries::from_samples(
            &mut flop_samples,
            &mut turn_samples,
            &mut river_samples,
            self.config.flop_buckets as usize,
            self.config.turn_buckets as usize,
            self.config.river_buckets as usize,
        )
    }

    fn sample_street(&self, rng: &mut StdRng, board_size: usize, num_samples: u32) -> Vec<f32> {
        let mut samples = Vec::with_capacity(num_samples as usize);
        let deck = Self::full_deck();

        for i in 0..num_samples {
            if i % 1000 == 0 && i > 0 {
                println!("  Sampled {}/{}", i, num_samples);
            }

            // Shuffle and deal
            let mut cards = deck.clone();
            cards.shuffle(rng);

            let board: Vec<Card> = cards[0..board_size].to_vec();
            let holding = (cards[board_size], cards[board_size + 1]);

            let hs = match board_size {
                3 => self.calculator.calculate_flop(&board, holding),
                4 => self.calculator.calculate_turn(&board, holding),
                5 => self.calculator.calculate_river(&board, holding),
                _ => panic!("Invalid board size"),
            };

            samples.push(hs.ehs2);
        }

        samples
    }

    fn full_deck() -> Vec<Card> {
        let values = [
            Value::Two, Value::Three, Value::Four, Value::Five, Value::Six,
            Value::Seven, Value::Eight, Value::Nine, Value::Ten,
            Value::Jack, Value::Queen, Value::King, Value::Ace,
        ];
        let suits = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];

        let mut deck = Vec::with_capacity(52);
        for &value in &values {
            for &suit in &suits {
                deck.push(Card { value, suit });
            }
        }
        deck
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generator_produces_valid_boundaries() {
        let config = AbstractionConfig {
            flop_buckets: 10,
            turn_buckets: 10,
            river_buckets: 10,
            samples_per_street: 100, // Small for testing
        };
        let generator = BoundaryGenerator::new(config);
        let boundaries = generator.generate(42);

        assert_eq!(boundaries.flop.len(), 9); // 10 buckets = 9 boundaries
        assert_eq!(boundaries.turn.len(), 9);
        assert_eq!(boundaries.river.len(), 9);

        // Boundaries should be monotonically increasing
        for window in boundaries.river.windows(2) {
            assert!(window[0] <= window[1], "Boundaries not monotonic");
        }
    }
}
```

**Step 2: Add rand dependency**

Run: `cargo add rand -p poker-solver-core`

**Step 3: Register module and export**

Add to `crates/core/src/abstraction/mod.rs`:

```rust
mod generator;

pub use generator::BoundaryGenerator;
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core generator -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/core/src/abstraction/ crates/core/Cargo.toml
git commit -m "feat(abstraction): add boundary generator for offline bucket computation"
```

---

## Task 11: Integration Test and Final Verification

**Files:**
- Create: `crates/core/tests/abstraction_integration.rs`

**Step 1: Create integration test**

Create `crates/core/tests/abstraction_integration.rs`:

```rust
use poker_solver_core::abstraction::{
    AbstractionConfig, BoundaryGenerator, CardAbstraction, Street,
};
use poker_solver_core::poker::{Card, Suit, Value};
use std::path::PathBuf;
use tempfile::tempdir;

#[test]
fn full_pipeline_generate_save_load() {
    // Generate small boundaries
    let config = AbstractionConfig {
        flop_buckets: 10,
        turn_buckets: 10,
        river_buckets: 20,
        samples_per_street: 50,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(12345);

    // Create abstraction and save
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    let dir = tempdir().unwrap();
    let path = dir.path().join("test_boundaries.bin");
    abstraction.save(&path).unwrap();

    // Load and verify
    let loaded = CardAbstraction::load(&path).unwrap();
    assert_eq!(loaded.num_buckets(Street::Flop), 10);
    assert_eq!(loaded.num_buckets(Street::Turn), 10);
    assert_eq!(loaded.num_buckets(Street::River), 20);
}

#[test]
fn bucket_lookup_returns_valid_index() {
    let config = AbstractionConfig {
        flop_buckets: 100,
        turn_buckets: 100,
        river_buckets: 200,
        samples_per_street: 100,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(42);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    // Test river lookup
    let board = vec![
        Card { value: Value::Ace, suit: Suit::Spades },
        Card { value: Value::King, suit: Suit::Spades },
        Card { value: Value::Queen, suit: Suit::Hearts },
        Card { value: Value::Jack, suit: Suit::Diamonds },
        Card { value: Value::Two, suit: Suit::Clubs },
    ];
    let holding = (
        Card { value: Value::Ten, suit: Suit::Spades },
        Card { value: Value::Nine, suit: Suit::Spades },
    );

    let bucket = abstraction.get_bucket(&board, holding).unwrap();
    assert!(bucket < 200, "Bucket {} should be < 200", bucket);
}

#[test]
fn duplicate_card_rejected() {
    let config = AbstractionConfig {
        flop_buckets: 10,
        turn_buckets: 10,
        river_buckets: 10,
        samples_per_street: 10,
    };
    let generator = BoundaryGenerator::new(config);
    let boundaries = generator.generate(1);
    let abstraction = CardAbstraction::from_boundaries(boundaries);

    let board = vec![
        Card { value: Value::Ace, suit: Suit::Spades },
        Card { value: Value::King, suit: Suit::Spades },
        Card { value: Value::Queen, suit: Suit::Hearts },
        Card { value: Value::Jack, suit: Suit::Diamonds },
        Card { value: Value::Two, suit: Suit::Clubs },
    ];
    // Holding has Ace of Spades which is on board
    let holding = (
        Card { value: Value::Ace, suit: Suit::Spades },
        Card { value: Value::Nine, suit: Suit::Hearts },
    );

    let result = abstraction.get_bucket(&board, holding);
    assert!(result.is_err(), "Should reject duplicate card");
}
```

**Step 2: Add tempfile dev-dependency**

Run: `cargo add tempfile --dev -p poker-solver-core`

**Step 3: Run integration tests**

Run: `cargo test -p poker-solver-core --test abstraction_integration -- --nocapture`
Expected: All 3 tests pass

**Step 4: Run all tests and clippy**

Run: `cargo test -p poker-solver-core && cargo clippy -p poker-solver-core -- -D warnings`
Expected: All tests pass, no clippy warnings

**Step 5: Final commit**

```bash
git add crates/core/tests/ crates/core/Cargo.toml
git commit -m "test(abstraction): add integration tests for full pipeline"
```

---

## Summary

After completing all tasks, the card abstraction system provides:

1. **Suit isomorphism** - ~12x reduction in board equivalence classes
2. **EHS2 calculation** - Equity + hand potential via exhaustive enumeration
3. **Percentile-based bucketing** - 30k buckets (5k flop, 5k turn, 20k river)
4. **Serialization** - Save/load bucket boundaries
5. **Clean API** - `CardAbstraction::get_bucket(board, holding)`

The system is ready for integration with the blueprint solver and subgame re-solver.
