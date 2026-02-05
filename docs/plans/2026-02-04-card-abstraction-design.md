# Card Abstraction System Design

## Overview

Card abstraction system for full HUNL poker (preflop → flop → turn → river) supporting Modicum-style blueprint + subgame solving.

## Requirements

- **Scope**: Full HUNL (all streets)
- **Bucketing**: EHS2 (equity + hand potential)
- **Bucket counts**: 30k total (5k flop, 5k turn, 20k river)
- **Isomorphism**: Full suit isomorphism reduction (~12x computation reduction)
- **Hand potential**: Exhaustive enumeration for accuracy

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     CARD ABSTRACTION SYSTEM                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────┐ │
│  │  Isomorphism     │    │  Hand Strength   │    │  Bucket   │ │
│  │  Canonicalizer   │───▶│  Calculator      │───▶│  Assigner │ │
│  └──────────────────┘    └──────────────────┘    └───────────┘ │
│         │                        │                      │       │
│         ▼                        ▼                      ▼       │
│  Canonical boards         EHS + EHS2 values      Bucket index   │
│  (1,755 flop classes)     per holding           (0-4999 flop,   │
│                                                  0-4999 turn,   │
│                                                  0-19999 river) │
└─────────────────────────────────────────────────────────────────┘
```

**Data flow:**
1. **Input**: Raw board + holding
2. **Canonicalize**: Map to canonical suit permutation
3. **Compute EHS2**: Calculate equity + potential via exhaustive enumeration
4. **Assign bucket**: Map EHS2 value to bucket index using precomputed boundaries

## Module Structure

```
crates/core/src/abstraction/
├── mod.rs              // Public API, CardAbstraction struct
├── isomorphism.rs      // CanonicalBoard, suit canonicalization
├── hand_strength.rs    // EHS, EHS2 calculation
└── buckets.rs          // Bucket assignment, boundaries
```

## Component Details

### 1. Isomorphism Canonicalizer

Reduces boards to canonical form by permuting suits. Maps ~22,100 flops to ~1,755 canonical classes.

**Canonicalization rules (applied in order):**
1. Suits ordered by frequency on board (most common first)
2. Ties broken by highest card in that suit
3. Remaining ties broken by second-highest card

**Example:**
```
Original:  Ah Kd 7c  (hearts=1, diamonds=1, clubs=1)
Canonical: As Kh 7d  (spades=hearts, hearts=diamonds, diamonds=clubs)
Mapping:   h→s, d→h, c→d, s→c
```

```rust
pub struct CanonicalBoard {
    cards: Vec<Card>,
    suit_mapping: [Suit; 4],
}

impl CanonicalBoard {
    pub fn canonicalize_holding(&self, card1: Card, card2: Card) -> (Card, Card);
}
```

### 2. Hand Strength Calculator (EHS2)

Computes equity and hand potential via exhaustive enumeration.

**EHS2 formula:**
```
EHS2 = EHS + (1 - EHS) × PPot - EHS × NPot

Where:
- EHS  = current equity vs uniform random opponent
- PPot = positive potential (behind now, ahead after runout)
- NPot = negative potential (ahead now, behind after runout)
```

**Enumeration counts per street:**

| Street | Opponent holdings | Runouts | Evaluations per holding |
|--------|-------------------|---------|-------------------------|
| Flop   | 1,081             | 990     | ~1.07M                  |
| Turn   | 1,035             | 44      | ~45K                    |
| River  | 990               | 1       | ~990                    |

```rust
pub struct HandStrength {
    pub ehs: f32,
    pub ehs2: f32,
    pub ppot: f32,
    pub npot: f32,
}
```

### 3. Bucket Assignment

Maps EHS2 values to discrete bucket indices using percentile-based boundaries.

**Boundary generation (offline):**
1. Sample ~100k random (board, holding) combinations per street
2. Compute EHS2 for each sample
3. Extract percentile boundaries for uniform distribution

**Runtime lookup:** O(log n) binary search

**Info set key format:**
```
Preflop:  "SB:AKs:r500"           (169 canonical hands)
Postflop: "SB:1234:Fxr500"        (bucket index, street prefix)
```

## Public API

```rust
pub struct CardAbstraction {
    assigner: BucketAssigner,
}

impl CardAbstraction {
    pub fn load(path: &Path) -> Result<Self, AbstractionError>;
    pub fn build(config: &AbstractionConfig) -> Result<Self, AbstractionError>;
    pub fn get_bucket(&self, board: &[Card], holding: (Card, Card)) -> u16;
    pub fn save(&self, path: &Path) -> Result<(), AbstractionError>;
}

pub struct AbstractionConfig {
    pub flop_buckets: u16,       // Default: 5,000
    pub turn_buckets: u16,       // Default: 5,000
    pub river_buckets: u16,      // Default: 20,000
    pub samples_per_street: u32, // Default: 100,000
}
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum AbstractionError {
    #[error("Invalid board: expected {expected} cards, got {got}")]
    InvalidBoardSize { expected: usize, got: usize },

    #[error("Duplicate card in board + holding: {card}")]
    DuplicateCard { card: Card },

    #[error("Failed to load boundaries: {0}")]
    LoadError(#[from] std::io::Error),

    #[error("Invalid boundary data: {reason}")]
    InvalidBoundaries { reason: String },
}
```

## Testing Strategy

**Unit tests:**
- Isomorphism: canonical form correctness, suit permutation invariance
- Hand strength: boundary conditions (nuts, air), EHS2 >= EHS for draws
- Buckets: monotonic boundaries, deterministic assignment

**Property-based tests:**
- All suit permutations produce same canonical board
- EHS2 always in [0, 1]
- Isomorphic hands get same bucket

**Integration tests:**
- Full pipeline: board → canonical → EHS2 → bucket
- Isomorphic board+holding pairs produce same bucket

## Future Considerations

- GPU acceleration for EHS2 computation during boundary generation
- EMD-based clustering upgrade for higher accuracy
- Precomputed lookup tables for runtime speed (memory vs CPU tradeoff)
