# Exhaustive Canonical Bucket Files — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace runtime equity computation in MCCFR `get_bucket()` with O(1) precomputed lookups using exhaustive canonical board bucket files.

**Architecture:** Extend the clustering pipeline to enumerate all canonical (suit-isomorphic) boards instead of random sampling, store canonical boards in bucket files for self-contained lookup, and wire `get_bucket()` to use the precomputed data via board canonicalization + combo indexing.

**Tech Stack:** Rust, `rs_poker::core::Card`, existing `CanonicalBoard` from `abstraction/isomorphism.rs`, existing `all_flops()` from `flops.rs`

---

### Task 1: Compact board key type (`PackedBoard`)

**Files:**
- Modify: `crates/core/src/blueprint_v2/bucket_file.rs`

A `u64` representation of a canonical board that is `Hash + Eq + Copy` for HashMap keys and compact serialization.

**Step 1: Add `PackedBoard` type and conversion functions**

At the top of `bucket_file.rs`, after the existing imports, add:

```rust
/// Compact canonical board representation for hashing and serialization.
///
/// Each card is encoded as 8 bits: `(value_rank << 2) | suit_index`.
/// Cards are packed left-to-right into a u64 (MSB-first), with
/// unused high bits zeroed. Supports up to 5 cards (40 bits).
///
/// The encoding is deterministic: given the same canonical card order,
/// the same u64 is always produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PackedBoard(pub u64);

impl PackedBoard {
    /// Pack a slice of canonical cards into a u64.
    ///
    /// Cards must already be in canonical suit form (from `CanonicalBoard`).
    /// Panics if more than 5 cards are provided.
    #[must_use]
    pub fn from_cards(cards: &[Card]) -> Self {
        assert!(cards.len() <= 5, "PackedBoard supports at most 5 cards");
        let mut packed: u64 = 0;
        for (i, card) in cards.iter().enumerate() {
            let byte = encode_card(*card);
            packed |= u64::from(byte) << (56 - i * 8);
        }
        PackedBoard(packed)
    }

    /// Unpack back to a vector of cards.
    ///
    /// `num_cards` specifies how many cards were packed (3, 4, or 5).
    #[must_use]
    pub fn to_cards(self, num_cards: usize) -> Vec<Card> {
        (0..num_cards)
            .map(|i| {
                let byte = (self.0 >> (56 - i * 8)) as u8;
                decode_card(byte)
            })
            .collect()
    }
}

/// Encode a card as a single byte: `(value_rank << 2) | suit_index`.
fn encode_card(card: Card) -> u8 {
    let value_rank = crate::card_utils::value_rank(card.value) as u8;
    let suit_idx = match card.suit {
        Suit::Spade => 0_u8,
        Suit::Heart => 1,
        Suit::Diamond => 2,
        Suit::Club => 3,
    };
    (value_rank << 2) | suit_idx
}

/// Decode a byte back to a card.
fn decode_card(byte: u8) -> Card {
    let rank = byte >> 2;
    let suit_idx = byte & 0x03;
    let value = rank_to_value(rank);
    let suit = match suit_idx {
        0 => Suit::Spade,
        1 => Suit::Heart,
        2 => Suit::Diamond,
        _ => Suit::Club,
    };
    Card::new(value, suit)
}

/// Map numeric rank (2–14) back to Value.
fn rank_to_value(rank: u8) -> Value {
    match rank {
        2 => Value::Two,
        3 => Value::Three,
        4 => Value::Four,
        5 => Value::Five,
        6 => Value::Six,
        7 => Value::Seven,
        8 => Value::Eight,
        9 => Value::Nine,
        10 => Value::Ten,
        11 => Value::Jack,
        12 => Value::Queen,
        13 => Value::King,
        14 => Value::Ace,
        _ => panic!("invalid rank: {rank}"),
    }
}
```

**Step 2: Add tests**

```rust
#[test]
fn packed_board_round_trip_flop() {
    let cards = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
    ];
    let packed = PackedBoard::from_cards(&cards);
    let recovered = packed.to_cards(3);
    assert_eq!(recovered, cards.to_vec());
}

#[test]
fn packed_board_round_trip_river() {
    let cards = [
        Card::new(Value::Two, Suit::Club),
        Card::new(Value::Five, Suit::Diamond),
        Card::new(Value::Ten, Suit::Heart),
        Card::new(Value::Jack, Suit::Spade),
        Card::new(Value::Ace, Suit::Spade),
    ];
    let packed = PackedBoard::from_cards(&cards);
    let recovered = packed.to_cards(5);
    assert_eq!(recovered, cards.to_vec());
}

#[test]
fn packed_board_hash_eq() {
    use std::collections::HashMap;
    let cards = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Heart),
        Card::new(Value::Queen, Suit::Diamond),
    ];
    let p1 = PackedBoard::from_cards(&cards);
    let p2 = PackedBoard::from_cards(&cards);
    assert_eq!(p1, p2);
    let mut map = HashMap::new();
    map.insert(p1, 42_u32);
    assert_eq!(map[&p2], 42);
}

#[test]
fn packed_board_different_boards_different_keys() {
    let a = PackedBoard::from_cards(&[
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Spade),
    ]);
    let b = PackedBoard::from_cards(&[
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Jack, Suit::Spade),
    ]);
    assert_ne!(a, b);
}
```

**Step 3: Run tests**

```bash
cargo test -p poker-solver-core -- bucket_file
```

**Step 4: Commit**

```bash
git add crates/core/src/blueprint_v2/bucket_file.rs
git commit -m "feat: add PackedBoard compact hashable board key"
```

---

### Task 2: Combo index mapping

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

We need a function to map any `(Card, Card)` pair to its index in the `enumerate_combos()` ordering. The deck is built in `VALUES × SUITS` order (13 values × 4 suits = 52 cards). Each card has a deck position, and combo index follows the triangular enumeration `for i in 0..52 { for j in i+1..52 }`.

**Step 1: Add `card_to_deck_index` and `combo_index` functions**

After `enumerate_combos()` (~line 624), add:

```rust
/// Map a card to its position in the canonical deck ordering.
///
/// The deck is ordered by value (Two→Ace) × suit (Spade, Heart, Diamond, Club),
/// matching `build_deck()`.
#[must_use]
pub fn card_to_deck_index(card: Card) -> usize {
    let value_idx = match card.value {
        Value::Two => 0,
        Value::Three => 1,
        Value::Four => 2,
        Value::Five => 3,
        Value::Six => 4,
        Value::Seven => 5,
        Value::Eight => 6,
        Value::Nine => 7,
        Value::Ten => 8,
        Value::Jack => 9,
        Value::Queen => 10,
        Value::King => 11,
        Value::Ace => 12,
    };
    let suit_idx = match card.suit {
        Suit::Spade => 0,
        Suit::Heart => 1,
        Suit::Diamond => 2,
        Suit::Club => 3,
    };
    value_idx * 4 + suit_idx
}

/// Map a two-card hole-card combo to its index in the canonical
/// `enumerate_combos()` ordering (0..1325).
///
/// The two cards can be in any order — they are sorted by deck index
/// internally to match the enumeration `for i in 0..52 { for j in i+1..52 }`.
#[must_use]
pub fn combo_index(c0: Card, c1: Card) -> u16 {
    let mut i = card_to_deck_index(c0);
    let mut j = card_to_deck_index(c1);
    if i > j {
        std::mem::swap(&mut i, &mut j);
    }
    // Triangular number formula: combos before row i = i*(2*52 - i - 1)/2
    // Then offset within row = j - i - 1
    #[allow(clippy::cast_possible_truncation)]
    let idx = (i * (2 * 52 - i - 1) / 2 + (j - i - 1)) as u16;
    idx
}
```

**Step 2: Add tests**

```rust
#[test]
fn combo_index_first_and_last() {
    let deck = build_deck();
    // First combo: (deck[0], deck[1]) → index 0
    assert_eq!(combo_index(deck[0], deck[1]), 0);
    // Last combo: (deck[50], deck[51]) → index 1325
    assert_eq!(combo_index(deck[50], deck[51]), 1325);
}

#[test]
fn combo_index_order_independent() {
    let c0 = Card::new(Value::Ace, Suit::Spade);
    let c1 = Card::new(Value::King, Suit::Heart);
    assert_eq!(combo_index(c0, c1), combo_index(c1, c0));
}

#[test]
fn combo_index_matches_enumeration() {
    let deck = build_deck();
    let combos = enumerate_combos(&deck);
    for (expected_idx, combo) in combos.iter().enumerate() {
        let actual = combo_index(combo[0], combo[1]);
        assert_eq!(
            actual, expected_idx as u16,
            "Mismatch at combo {:?}: expected {expected_idx}, got {actual}",
            combo
        );
    }
}
```

**Step 3: Make `combo_index` and `card_to_deck_index` pub(crate) or pub**

They'll be used from `mccfr.rs`, so make them `pub`.

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- cluster_pipeline::tests::combo_index
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: add combo_index mapping for bucket file lookups"
```

---

### Task 3: Canonical board enumeration for turn and river

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

We need functions to enumerate all canonical 4-card (turn) and 5-card (river) boards, street-relative. The existing `all_flops()` in `flops.rs` handles 3-card canonical flops.

**Step 1: Add `enumerate_canonical_turns` function**

```rust
use crate::abstraction::isomorphism::CanonicalBoard;
use crate::flops::all_flops;
use std::collections::HashMap;

/// A canonical board with its isomorphism weight (occurrence count).
#[derive(Debug, Clone)]
pub struct WeightedBoard<const N: usize> {
    pub cards: [Card; N],
    pub weight: u32,
}

/// Enumerate all canonical 4-card boards (flop + turn) with weights.
///
/// For each of the 1,755 canonical flops, enumerate all 49 possible turn
/// cards, canonicalize the resulting 4-card board, and deduplicate. Each
/// canonical turn board's weight is the sum of (flop_weight × turn_weight)
/// across all raw boards that map to it.
#[must_use]
pub fn enumerate_canonical_turns() -> Vec<WeightedBoard<4>> {
    let deck = build_deck();
    let flops = all_flops();
    let mut board_weights: HashMap<PackedBoard, (Vec<Card>, u32)> = HashMap::new();

    for flop in &flops {
        let flop_cards: Vec<Card> = flop.cards.to_vec();
        for &turn_card in &deck {
            if flop_cards.contains(&turn_card) {
                continue;
            }
            let mut board = flop_cards.clone();
            board.push(turn_card);
            let canonical = CanonicalBoard::from_cards(&board);
            let packed = PackedBoard::from_cards(&canonical.cards);
            let entry = board_weights
                .entry(packed)
                .or_insert_with(|| (canonical.cards.clone(), 0));
            entry.1 += u32::from(flop.weight);
        }
    }

    let mut result: Vec<WeightedBoard<4>> = board_weights
        .into_values()
        .map(|(cards, weight)| WeightedBoard {
            cards: [cards[0], cards[1], cards[2], cards[3]],
            weight,
        })
        .collect();
    result.sort_by_key(|b| PackedBoard::from_cards(&b.cards).0);
    result
}
```

**Step 2: Add `enumerate_canonical_rivers` function**

```rust
/// Enumerate all canonical 5-card boards (flop + turn + river) with weights.
///
/// For each canonical turn board, enumerate all 48 possible river cards,
/// canonicalize the resulting 5-card board, and deduplicate.
#[must_use]
pub fn enumerate_canonical_rivers() -> Vec<WeightedBoard<5>> {
    let deck = build_deck();
    let turns = enumerate_canonical_turns();
    let mut board_weights: HashMap<PackedBoard, (Vec<Card>, u32)> = HashMap::new();

    for turn in &turns {
        let turn_cards: Vec<Card> = turn.cards.to_vec();
        for &river_card in &deck {
            if turn_cards.contains(&river_card) {
                continue;
            }
            let mut board = turn_cards.clone();
            board.push(river_card);
            let canonical = CanonicalBoard::from_cards(&board);
            let packed = PackedBoard::from_cards(&canonical.cards);
            let entry = board_weights
                .entry(packed)
                .or_insert_with(|| (canonical.cards.clone(), 0));
            entry.1 += turn.weight;
        }
    }

    let mut result: Vec<WeightedBoard<5>> = board_weights
        .into_values()
        .map(|(cards, weight)| WeightedBoard {
            cards: [cards[0], cards[1], cards[2], cards[3], cards[4]],
            weight,
        })
        .collect();
    result.sort_by_key(|b| PackedBoard::from_cards(&b.cards).0);
    result
}
```

**Step 3: Add `enumerate_canonical_flops` wrapper**

Wrap existing `all_flops()` to return `Vec<WeightedBoard<3>>`:

```rust
/// Enumerate all 1,755 canonical 3-card flops with weights.
#[must_use]
pub fn enumerate_canonical_flops() -> Vec<WeightedBoard<3>> {
    let flops = all_flops();
    flops
        .into_iter()
        .map(|f| WeightedBoard {
            cards: f.cards,
            weight: u32::from(f.weight),
        })
        .collect()
}
```

**Step 4: Add tests**

```rust
#[test]
fn canonical_flops_count() {
    let flops = enumerate_canonical_flops();
    assert_eq!(flops.len(), 1755);
    let total_weight: u32 = flops.iter().map(|f| f.weight).sum();
    assert_eq!(total_weight, 22100); // C(52,3)
}

#[test]
fn canonical_turns_reasonable_count() {
    let turns = enumerate_canonical_turns();
    // Should be roughly 14,000-16,000 canonical turns
    assert!(turns.len() > 10_000, "too few turns: {}", turns.len());
    assert!(turns.len() < 20_000, "too many turns: {}", turns.len());
    // Total weight should be C(52,3) × 49 = 22100 × 49 = 1,082,900
    let total_weight: u32 = turns.iter().map(|t| t.weight).sum();
    assert_eq!(total_weight, 22100 * 49);
}

// Note: canonical river enumeration is expensive (~30s). Test it with
// #[ignore] and run with `cargo test -- --ignored` when needed.
#[test]
#[ignore]
fn canonical_rivers_reasonable_count() {
    let rivers = enumerate_canonical_rivers();
    assert!(rivers.len() > 500_000, "too few rivers: {}", rivers.len());
    assert!(rivers.len() < 1_000_000, "too many rivers: {}", rivers.len());
    let total_weight: u64 = rivers.iter().map(|r| u64::from(r.weight)).sum();
    assert_eq!(total_weight, 22100 * 49 * 48); // C(52,3) × 49 × 48
}
```

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core -- canonical_flops_count canonical_turns_reasonable
```

**Step 6: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: exhaustive canonical board enumeration with weights"
```

---

### Task 4: Weighted k-means

**Files:**
- Modify: `crates/core/src/blueprint_v2/clustering.rs`

The k-means functions need to accept optional weights so canonical boards with higher isomorphism counts influence centroids proportionally.

**Step 1: Add `kmeans_1d_weighted` function**

After existing `kmeans_1d` (~line 157), add:

```rust
/// Weighted 1-D k-means clustering.
///
/// Each data point has an associated weight. Weighted points contribute
/// proportionally to centroid computation.
#[allow(clippy::cast_possible_truncation)]
pub fn kmeans_1d_weighted(
    data: &[f64],
    weights: &[f64],
    k: usize,
    max_iterations: u32,
) -> Vec<u16> {
    assert_eq!(data.len(), weights.len());
    assert!(!data.is_empty());
    assert!(k > 0);

    let n = data.len();
    if k >= n {
        return (0..n).map(|i| i as u16).collect();
    }

    let mut sorted: Vec<(f64, f64)> = data.iter().copied().zip(weights.iter().copied()).collect();
    sorted.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut centroids: Vec<f64> = (0..k)
        .map(|i| {
            let idx = (i * (n - 1)) / (k.max(2) - 1).max(1);
            sorted[idx.min(n - 1)].0
        })
        .collect();

    let mut assignments = vec![0_u16; n];

    for _ in 0..max_iterations {
        let mut changed = false;
        for (i, &val) in data.iter().enumerate() {
            let nearest = nearest_centroid_1d(val, &centroids);
            if assignments[i] != nearest {
                assignments[i] = nearest;
                changed = true;
            }
        }

        if !changed {
            break;
        }

        let mut sums = vec![0.0_f64; k];
        let mut weight_sums = vec![0.0_f64; k];
        for (i, (&val, &w)) in data.iter().zip(weights.iter()).enumerate() {
            let c = assignments[i] as usize;
            sums[c] += val * w;
            weight_sums[c] += w;
        }
        for c in 0..k {
            if weight_sums[c] > 0.0 {
                centroids[c] = sums[c] / weight_sums[c];
            }
        }
    }

    assignments
}
```

**Step 2: Add `kmeans_emd_weighted` function**

After `kmeans_emd_with_progress`, add a weighted variant:

```rust
/// Weighted EMD k-means clustering.
///
/// Same as [`kmeans_emd_with_progress`] but centroid updates use weighted
/// averages.
#[allow(clippy::cast_possible_truncation)]
pub fn kmeans_emd_weighted(
    data: &[Vec<f64>],
    weights: &[f64],
    k: usize,
    max_iterations: u32,
    seed: u64,
    progress: impl Fn(u32, u32),
) -> Vec<u16> {
    assert_eq!(data.len(), weights.len());
    assert!(!data.is_empty());
    assert!(k > 0);

    let n = data.len();
    if k >= n {
        return (0..n).map(|i| i as u16).collect();
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut centroids = kmeanspp_init(data, k, &mut rng);
    let mut assignments = vec![0_u16; n];
    let dim = data[0].len();

    for iter in 0..max_iterations {
        progress(iter, max_iterations);

        // Assignment step (unchanged — nearest centroid by EMD)
        let changed: bool = data
            .par_iter()
            .zip(assignments.par_iter_mut())
            .map(|(point, assignment)| {
                let nearest = nearest_centroid_emd(point, &centroids);
                if *assignment != nearest {
                    *assignment = nearest;
                    true
                } else {
                    false
                }
            })
            .reduce(|| false, |a, b| a || b);

        if !changed {
            break;
        }

        // Weighted centroid update
        let mut new_centroids = vec![vec![0.0_f64; dim]; k];
        let mut weight_sums = vec![0.0_f64; k];
        for (i, (point, &w)) in data.iter().zip(weights.iter()).enumerate() {
            let c = assignments[i] as usize;
            weight_sums[c] += w;
            for (d, &val) in point.iter().enumerate() {
                new_centroids[c][d] += val * w;
            }
        }
        for c in 0..k {
            if weight_sums[c] > 0.0 {
                for d in 0..dim {
                    new_centroids[c][d] /= weight_sums[c];
                }
            }
        }
        centroids = new_centroids;
    }

    assignments
}
```

Note: This requires `nearest_centroid_emd` and `kmeanspp_init` to be accessible — check that they are not private. If they are, make them `pub(crate)`.

**Step 3: Add tests**

```rust
#[test]
fn weighted_1d_kmeans_respects_weights() {
    // Two clusters: low values with high weight, high values with low weight.
    // Centroids should be pulled toward the heavy cluster.
    let data = vec![0.1, 0.2, 0.9, 1.0];
    let weights = vec![10.0, 10.0, 1.0, 1.0];
    let labels = kmeans_1d_weighted(&data, &weights, 2, 100);
    // The two low values should be in the same cluster
    assert_eq!(labels[0], labels[1]);
    // The two high values should be in the same cluster
    assert_eq!(labels[2], labels[3]);
    // They should be in different clusters
    assert_ne!(labels[0], labels[2]);
}
```

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- clustering
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/clustering.rs
git commit -m "feat: add weighted k-means for canonical board clustering"
```

---

### Task 5: Extend `BucketFile` format with board table

**Files:**
- Modify: `crates/core/src/blueprint_v2/bucket_file.rs`

Extend the binary format to include a board table between the header and bucket data. Bump the version to 2 and maintain backward compatibility for reading v1 files (which have no board table).

**Step 1: Update format constants and header**

```rust
const VERSION: u8 = 2; // bumped from 1

#[derive(Debug, Clone)]
pub struct BucketFileHeader {
    pub street: Street,
    pub bucket_count: u16,
    pub board_count: u32,
    pub combos_per_board: u16,
    /// Version of the file format (1 = no board table, 2 = has board table).
    pub version: u8,
}
```

Add a `boards` field to `BucketFile`:

```rust
pub struct BucketFile {
    pub header: BucketFileHeader,
    /// Canonical boards in order, one per board_idx. Empty for v1 files.
    pub boards: Vec<PackedBoard>,
    /// Flat array: `buckets[board_idx * combos_per_board + combo_idx]`
    pub buckets: Vec<u16>,
}
```

**Step 2: Update `write_to`**

After writing the header, write the board table:

```rust
// Write board table (v2+)
for &board in &self.boards {
    writer.write_all(&board.0.to_le_bytes())?;
}
// Then write bucket data (unchanged)
```

**Step 3: Update `read_from`**

Read version, and if v2, read the board table:

```rust
let version = buf[0]; // after MAGIC
// ... read rest of header ...

let boards = if version >= 2 {
    let mut boards = Vec::with_capacity(board_count as usize);
    for _ in 0..board_count {
        let mut buf = [0u8; 8];
        reader.read_exact(&mut buf)?;
        boards.push(PackedBoard(u64::from_le_bytes(buf)));
    }
    boards
} else {
    Vec::new()
};
```

Ensure v1 files still load correctly (empty `boards` vec).

**Step 4: Update all existing `BucketFile` construction sites**

Search for `BucketFile {` and `BucketFile::` across the codebase. Add `boards: vec![]` where bucket files are constructed without canonical boards (existing code paths). The new canonical clustering functions (Task 6) will populate `boards`.

Also update the `BucketFileHeader` construction sites to include `version: VERSION` (or `version: 2`).

**Step 5: Add tests**

```rust
#[test]
fn bucket_file_v2_round_trip_with_boards() {
    let boards = vec![
        PackedBoard::from_cards(&[
            Card::new(Value::Ace, Suit::Spade),
            Card::new(Value::King, Suit::Spade),
            Card::new(Value::Queen, Suit::Heart),
        ]),
        PackedBoard::from_cards(&[
            Card::new(Value::Two, Suit::Spade),
            Card::new(Value::Three, Suit::Heart),
            Card::new(Value::Four, Suit::Diamond),
        ]),
    ];
    let bf = BucketFile {
        header: BucketFileHeader {
            street: Street::Flop,
            bucket_count: 10,
            board_count: 2,
            combos_per_board: 1326,
            version: 2,
        },
        boards: boards.clone(),
        buckets: vec![0_u16; 2 * 1326],
    };

    let mut buf = Vec::new();
    bf.write_to(&mut buf).unwrap();

    let loaded = BucketFile::read_from(&mut std::io::Cursor::new(buf)).unwrap();
    assert_eq!(loaded.boards.len(), 2);
    assert_eq!(loaded.boards[0], boards[0]);
    assert_eq!(loaded.boards[1], boards[1]);
    assert_eq!(loaded.header.version, 2);
}
```

**Step 6: Run tests**

```bash
cargo test -p poker-solver-core -- bucket_file
```

**Step 7: Commit**

```bash
git add crates/core/src/blueprint_v2/bucket_file.rs
git commit -m "feat: extend BucketFile format v2 with board table"
```

---

### Task 6: Rewrite clustering pipeline with canonical boards

**Files:**
- Modify: `crates/core/src/blueprint_v2/cluster_pipeline.rs`

Replace `sample_boards` / `sample_turn_boards` / `sample_flop_boards` with the exhaustive canonical enumeration from Task 3. Use weighted k-means from Task 4.

**Step 1: Rewrite `cluster_river_with_boards` → `cluster_river_canonical`**

Create a new function that:
1. Calls `enumerate_canonical_rivers()` (or accepts pre-enumerated boards)
2. Computes equity for each (canonical_board, combo) pair
3. Builds weight vector from `WeightedBoard.weight`
4. Calls `kmeans_1d_weighted` instead of `kmeans_1d`
5. Returns `BucketFile` with populated `boards` field

```rust
pub fn cluster_river_canonical(
    bucket_count: u16,
    kmeans_iterations: u32,
    seed: u64,
    progress: impl Fn(f64) + Sync,
) -> BucketFile {
    let combos = enumerate_combos(&build_deck());
    let boards = enumerate_canonical_rivers();
    let num_boards = boards.len();

    // Compute equity for each (board, combo) pair in parallel.
    let board_equities: Vec<Vec<Option<f64>>> = boards
        .par_iter()
        .enumerate()
        .map(|(i, wb)| {
            let eq = compute_board_equities(wb.cards, &combos);
            progress((i + 1) as f64 / num_boards as f64 * 0.8);
            eq
        })
        .collect();

    // Collect valid equities with per-sample weights.
    let mut all_equities: Vec<f64> = Vec::new();
    let mut all_weights: Vec<f64> = Vec::new();
    let mut positions: Vec<(usize, usize)> = Vec::new();

    for (board_idx, (eqs, wb)) in board_equities.iter().zip(boards.iter()).enumerate() {
        for (combo_idx, eq) in eqs.iter().enumerate() {
            if let Some(e) = eq {
                all_equities.push(*e);
                all_weights.push(f64::from(wb.weight));
                positions.push((board_idx, combo_idx));
            }
        }
    }

    let cluster_labels = kmeans_1d_weighted(
        &all_equities, &all_weights, bucket_count as usize, kmeans_iterations,
    );

    let total = num_boards * TOTAL_COMBOS as usize;
    let mut buckets = vec![0_u16; total];
    for (flat_idx, &(board_idx, combo_idx)) in positions.iter().enumerate() {
        buckets[board_idx * TOTAL_COMBOS as usize + combo_idx] = cluster_labels[flat_idx];
    }

    let packed_boards: Vec<PackedBoard> = boards
        .iter()
        .map(|wb| PackedBoard::from_cards(&wb.cards))
        .collect();

    BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count,
            board_count: num_boards as u32,
            combos_per_board: TOTAL_COMBOS,
            version: 2,
        },
        boards: packed_boards,
        buckets,
    }
}
```

**Step 2: Rewrite turn and flop clustering similarly**

Apply the same pattern:
- `cluster_turn_canonical` uses `enumerate_canonical_turns()`, builds histograms using `build_next_street_histogram`, calls `kmeans_emd_weighted` with board weights
- `cluster_flop_canonical` uses `enumerate_canonical_flops()`, same pattern

For the histogram building, note an important subtlety: `build_next_street_histogram` currently computes raw equity and bins it uniformly. With canonical boards, the histogram still represents "distribution over next-street equity bins" and works the same way — no change needed to the histogram building logic.

**Step 3: Update `run_clustering_pipeline`**

Replace calls to `cluster_river` / `cluster_turn` / `cluster_flop` with the canonical variants. Keep the old functions available (don't delete them) — mark them with `#[allow(dead_code)]` for now in case they're useful for testing.

```rust
pub fn run_clustering_pipeline(
    config: &ClusteringConfig,
    output_dir: &Path,
    progress: impl Fn(&str, f64) + Sync,
) -> Result<(), Box<dyn std::error::Error>> {
    // 1. River (canonical)
    progress("river", 0.0);
    let river = cluster_river_canonical(
        config.river.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("river", p),
    );
    river.save(&output_dir.join("river.buckets"))?;

    // 2. Turn (canonical, depends on river bucket_count)
    progress("turn", 0.0);
    let turn = cluster_turn_canonical(
        &river,
        config.turn.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("turn", p),
    );
    turn.save(&output_dir.join("turn.buckets"))?;

    // 3. Flop (canonical, depends on turn bucket_count)
    progress("flop", 0.0);
    let flop = cluster_flop_canonical(
        &turn,
        config.flop.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("flop", p),
    );
    flop.save(&output_dir.join("flop.buckets"))?;

    // 4. Preflop (unchanged — already uses 169 canonical hands)
    progress("preflop", 0.0);
    let preflop = cluster_preflop(
        config.preflop.buckets,
        config.kmeans_iterations,
        config.seed,
        |p| progress("preflop", p),
    );
    preflop.save(&output_dir.join("preflop.buckets"))?;

    Ok(())
}
```

**Step 4: Run tests**

```bash
cargo test -p poker-solver-core -- cluster_pipeline
```

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/cluster_pipeline.rs
git commit -m "feat: canonical board clustering pipeline with weighted k-means"
```

---

### Task 7: Wire `get_bucket` to use bucket file lookups

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs`

This is the payoff — replace the `compute_equity` fallback with O(1) bucket file lookups.

**Step 1: Add board index lookup structure to `AllBuckets`**

```rust
use crate::abstraction::isomorphism::CanonicalBoard;
use super::bucket_file::PackedBoard;
use super::cluster_pipeline::combo_index;
use std::collections::HashMap;

pub struct AllBuckets {
    pub bucket_counts: [u16; 4],
    pub bucket_files: [Option<BucketFile>; 4],
    /// Board index lookup tables, built from bucket file board tables.
    /// `board_maps[street] = HashMap<PackedBoard, u32>` for O(1) board lookup.
    board_maps: [Option<HashMap<PackedBoard, u32>>; 4],
}
```

**Step 2: Build board maps at construction time**

Add a constructor or initialization method:

```rust
impl AllBuckets {
    /// Create AllBuckets and build board index maps from loaded bucket files.
    pub fn new(bucket_counts: [u16; 4], bucket_files: [Option<BucketFile>; 4]) -> Self {
        let board_maps = std::array::from_fn(|street_idx| {
            bucket_files[street_idx].as_ref().and_then(|bf| {
                if bf.boards.is_empty() {
                    None // v1 file, no board table
                } else {
                    let map: HashMap<PackedBoard, u32> = bf
                        .boards
                        .iter()
                        .enumerate()
                        .map(|(idx, &packed)| (packed, idx as u32))
                        .collect();
                    Some(map)
                }
            })
        });
        Self {
            bucket_counts,
            bucket_files,
            board_maps,
        }
    }
}
```

**Step 3: Rewrite `get_bucket` to use board maps**

```rust
pub fn get_bucket(&self, street: Street, hole_cards: [Card; 2], board: &[Card]) -> u16 {
    if street == Street::Preflop {
        let hand = crate::hands::CanonicalHand::from_cards(hole_cards[0], hole_cards[1]);
        let idx = hand.index() as u16;
        return idx.min(self.bucket_counts[0] - 1);
    }

    let street_idx = street as usize;

    // Try bucket file lookup (O(1) when canonical boards are available)
    if let (Some(bf), Some(board_map)) = (
        &self.bucket_files[street_idx],
        &self.board_maps[street_idx],
    ) {
        let canonical = CanonicalBoard::from_cards(board);
        let packed = PackedBoard::from_cards(&canonical.cards);
        if let Some(&board_idx) = board_map.get(&packed) {
            // Apply the same suit permutation to hole cards
            let (c0, c1) = canonical.canonicalize_holding(hole_cards[0], hole_cards[1]);
            let cidx = combo_index(c0, c1) as usize;
            let flat = board_idx as usize * bf.header.combos_per_board as usize + cidx;
            return bf.buckets[flat];
        }
        // Board not in file — fall through to equity fallback
    }

    // Fallback: equity-based bucketing (for v1 files or missing boards)
    let equity = crate::showdown_equity::compute_equity(hole_cards, board);
    let k = self.bucket_counts[street_idx];
    let bucket = (equity * f64::from(k)) as u16;
    bucket.min(k - 1)
}
```

**Step 4: Update `AllBuckets` construction in `trainer.rs`**

Replace direct struct construction with `AllBuckets::new(...)`:

Find where `AllBuckets { bucket_counts, bucket_files }` is constructed in `trainer.rs` (~line 190-194) and replace with:

```rust
let buckets = AllBuckets::new(bucket_counts, bucket_files);
```

**Step 5: Remove the TODO comments**

Delete the two TODO comments in `get_bucket` (lines 96-100 and 108-109).

**Step 6: Add tests**

```rust
#[test]
fn get_bucket_uses_file_when_available() {
    use super::bucket_file::{BucketFile, BucketFileHeader, PackedBoard};
    use crate::abstraction::isomorphism::CanonicalBoard;
    use crate::poker::{Card, Suit, Value};

    // Build a minimal bucket file with one board
    let board_cards = [
        Card::new(Value::Ace, Suit::Spade),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Heart),
        Card::new(Value::Jack, Suit::Diamond),
        Card::new(Value::Ten, Suit::Club),
    ];
    let canonical = CanonicalBoard::from_cards(&board_cards);
    let packed = PackedBoard::from_cards(&canonical.cards);

    let mut buckets_data = vec![0_u16; 1326];
    // Set a known combo to bucket 7
    let (c0, c1) = canonical.canonicalize_holding(
        Card::new(Value::Two, Suit::Spade),
        Card::new(Value::Three, Suit::Spade),
    );
    let cidx = super::super::cluster_pipeline::combo_index(c0, c1) as usize;
    buckets_data[cidx] = 7;

    let bf = BucketFile {
        header: BucketFileHeader {
            street: Street::River,
            bucket_count: 10,
            board_count: 1,
            combos_per_board: 1326,
            version: 2,
        },
        boards: vec![packed],
        buckets: buckets_data,
    };

    let all = AllBuckets::new(
        [169, 10, 10, 10],
        [None, None, None, Some(bf)],
    );

    let result = all.get_bucket(
        Street::River,
        [Card::new(Value::Two, Suit::Spade), Card::new(Value::Three, Suit::Spade)],
        &board_cards,
    );
    assert_eq!(result, 7);
}
```

**Step 7: Run tests**

```bash
cargo test -p poker-solver-core -- mccfr
cargo test  # full suite
```

**Step 8: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: wire get_bucket to use canonical bucket file lookups"
```

---

### Task 8: Resume-safe caching in trainer

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs`

Ensure the clustering pipeline is skipped when bucket files already exist (resume-safe).

**Step 1: Add existence check before clustering**

In the trainer's init or `run_clustering_pipeline` call site, check if all 4 bucket files already exist:

```rust
fn bucket_files_exist(cluster_dir: &Path) -> bool {
    ["river.buckets", "turn.buckets", "flop.buckets", "preflop.buckets"]
        .iter()
        .all(|name| cluster_dir.join(name).exists())
}
```

Before calling `run_clustering_pipeline`, check:

```rust
if bucket_files_exist(cluster_path) {
    log::info!("Bucket files already exist in {}, skipping clustering", cluster_path.display());
} else {
    run_clustering_pipeline(&config.clustering, cluster_path, progress)?;
}
```

**Step 2: Run tests and commit**

```bash
cargo test -p poker-solver-core
git add crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: skip clustering when bucket files already exist (resume-safe)"
```

---

### Task 9: Final validation

**Step 1: Run full test suite**

```bash
cargo test
```

**Step 2: Run clippy**

```bash
cargo clippy
```

**Step 3: Run the canonical rivers enumeration test (slow)**

```bash
cargo test -p poker-solver-core -- canonical_rivers_reasonable_count --ignored
```

Verify the count is in the expected range (~700K) and total weight equals `22100 × 49 × 48`.

**Step 4: Integration smoke test**

Run a toy clustering pipeline with the canonical boards:

```bash
cargo run -p poker-solver-trainer --release -- cluster -c sample_configurations/blueprint_v2_toy.yaml
```

Verify:
- All 4 bucket files are produced
- The files are v2 format (have board tables)
- Rerunning skips clustering ("already exist" message)

**Step 5: Benchmark MCCFR with bucket files**

Run a short training session and verify that `precompute_buckets` no longer calls `compute_equity` (should be dramatically faster):

```bash
cargo run -p poker-solver-trainer --release -- train -c sample_configurations/blueprint_v2_toy.yaml
```

**Step 6: Commit any final fixes**

```bash
git add -A && git commit -m "chore: final validation for canonical bucket files"
```
