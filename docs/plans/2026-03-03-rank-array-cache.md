# Rank Array Cache + Equity Table Restructuring — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Restructure `compute_equity_table` to evaluate each concrete combo once per board (instead of once per opponent matchup), yielding ~20-50x speedup. Cache per-board rank arrays to disk for instant subsequent derivation of equity tables.

**Architecture:** Two-phase approach: Phase 1 computes a u32 rank ordinal for each concrete combo on each (turn, river) board. Phase 2 derives the 169×169 equity table via cheap integer comparison of precomputed ranks. A new `RankArrayCache` module persists the Phase 1 output to disk. Keep `rs_poker` as evaluator — only runs once per flop ever.

**Tech Stack:** Rust, rayon (parallelism), bincode + zstd (cache serialization), rs_poker (hand evaluation), criterion (benchmarks)

## Agent Team & Execution Order

| Agent | Role | Tasks |
|-------|------|-------|
| `rust-developer` A | Core implementation | Tasks 1-3 (rank_to_ordinal, restructured compute, correctness) |
| `rust-developer` B | Cache + integration | Tasks 4-5 (rank cache module, trainer integration) |
| `rust-developer` C | Benchmarking + swap | Task 6 (benchmark, replace old impl) |
| `idiomatic-rust-enforcer` | Review | Post-implementation review |
| `rust-perf-reviewer` | Review | Post-implementation review |

**Execution order:**
1. Task 1 (rank_to_ordinal) — **sequential**, foundation for everything
2. Tasks 2 + 4 — **parallel** (restructured compute + cache module are independent)
3. Task 3 (correctness test) — **sequential**, needs Task 2
4. Task 5 (trainer integration) — **sequential**, needs Tasks 3 + 4
5. Task 6 (benchmark + swap) — **sequential**, needs Task 3
6. Reviews — **parallel** (idiomatic + perf reviewers)

---

### Task 1: Add `rank_to_ordinal` utility

Converts `rs_poker::Rank` enum to a `u32` that preserves comparison ordering. This enables fast integer comparison in the restructured inner loop instead of enum matching.

**Files:**
- Modify: `crates/core/src/showdown_equity.rs`

**Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `showdown_equity.rs`:

```rust
#[timed_test]
fn rank_to_ordinal_preserves_ordering() {
    // Build a set of known hands with known relative ranking
    let board = [
        card(King, Diamond),
        card(Seven, Club),
        card(Three, Spade),
        card(Nine, Heart),
        card(Two, Diamond),
    ];

    // Weakest to strongest
    let hands = [
        [card(Four, Heart), card(Six, Club)],       // high card
        [card(Three, Heart), card(Five, Club)],      // pair of 3s
        [card(Nine, Spade), card(Four, Club)],       // pair of 9s
        [card(King, Heart), card(Four, Club)],       // pair of Ks
        [card(Seven, Heart), card(Nine, Club)],      // two pair 9s and 7s
        [card(King, Spade), card(Seven, Diamond)],   // two pair Ks and 7s
        [card(Three, Club), card(Three, Diamond)],   // trips 3s
        [card(King, Club), card(Nine, Diamond)],     // two pair Ks and 9s
    ];

    let ordinals: Vec<u32> = hands
        .iter()
        .map(|h| rank_to_ordinal(rank_hand(*h, &board)))
        .collect();

    // Verify strictly increasing (stronger hand = higher ordinal)
    for i in 0..ordinals.len() - 1 {
        assert!(
            ordinals[i] < ordinals[i + 1],
            "ordinal[{i}] ({}) should be < ordinal[{}] ({})",
            ordinals[i],
            i + 1,
            ordinals[i + 1],
        );
    }
}

#[timed_test]
fn rank_to_ordinal_category_boundaries() {
    // Any pair beats any high card
    let board = [
        card(King, Diamond),
        card(Seven, Club),
        card(Three, Spade),
        card(Nine, Heart),
        card(Two, Diamond),
    ];
    let high_card = rank_to_ordinal(rank_hand(
        [card(Ace, Heart), card(Jack, Club)],
        &board,
    ));
    let low_pair = rank_to_ordinal(rank_hand(
        [card(Two, Heart), card(Two, Club)],
        &board,
    ));
    assert!(high_card < low_pair, "any pair should beat any high card");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core rank_to_ordinal -- --nocapture`
Expected: compile error — `rank_to_ordinal` not defined

**Step 3: Write minimal implementation**

Add above the `#[cfg(test)]` block in `showdown_equity.rs`:

```rust
/// Convert a [`Rank`] to a `u32` ordinal that preserves comparison ordering.
///
/// The encoding places the hand category in bits 26..28 and the kicker
/// value in bits 0..25. This guarantees that for any two ranks `a` and `b`,
/// `a.cmp(&b) == rank_to_ordinal(a).cmp(&rank_to_ordinal(b))`.
///
/// The maximum kicker value across all `rs_poker` hand categories is <2^26
/// (verified empirically: TwoPair's `(pairs << 13) | low` maxes at ~50M).
pub fn rank_to_ordinal(rank: Rank) -> u32 {
    let (cat, kicker) = match rank {
        Rank::HighCard(k) => (0u32, k),
        Rank::OnePair(k) => (1, k),
        Rank::TwoPair(k) => (2, k),
        Rank::ThreeOfAKind(k) => (3, k),
        Rank::Straight(k) => (4, k),
        Rank::Flush(k) => (5, k),
        Rank::FullHouse(k) => (6, k),
        Rank::FourOfAKind(k) => (7, k),
        Rank::StraightFlush(k) => (8, k),
    };
    debug_assert!(
        kicker < (1 << 26),
        "kicker overflow: {kicker} for category {cat}"
    );
    (cat << 26) | kicker
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p poker-solver-core rank_to_ordinal -- --nocapture`
Expected: all pass

**Step 5: Commit**

```bash
git add crates/core/src/showdown_equity.rs
git commit -m "feat(core): add rank_to_ordinal for fast hand rank comparison"
```

---

### Task 2: Implement restructured `compute_equity_table`

The core algorithmic change. Two phases:
- **Phase 1**: For each (turn, river) board, evaluate every concrete combo once → u32 rank ordinal
- **Phase 2**: For each canonical pair, compare precomputed ranks via u32 comparison

Parallelized via rayon `fold`/`reduce` over boards.

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`
- Modify: `crates/core/src/showdown_equity.rs` (make `rank_to_ordinal` `pub`)

**Step 1: Write the failing correctness test**

Add to `postflop_exhaustive.rs` tests:

```rust
#[timed_test(30)]
#[ignore = "slow: compares old vs new equity table"]
fn restructured_equity_table_matches_original() {
    let flop = test_flop();
    let combo_map = build_combo_map(&flop);

    let original = compute_equity_table(&combo_map, flop);
    let restructured = compute_equity_table_fast(&combo_map, flop);

    assert_eq!(original.len(), restructured.len());
    let n = NUM_CANONICAL_HANDS;
    for h in 0..n {
        for o in 0..n {
            let idx = h * n + o;
            let a = original[idx];
            let b = restructured[idx];
            if a.is_nan() {
                assert!(b.is_nan(), "mismatch at [{h}][{o}]: original=NaN, new={b}");
            } else {
                assert!(
                    (a - b).abs() < 1e-10,
                    "mismatch at [{h}][{o}]: original={a}, new={b}, diff={}",
                    (a - b).abs()
                );
            }
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core restructured_equity_table -- --ignored --nocapture`
Expected: compile error — `compute_equity_table_fast` not defined

**Step 3: Implement `compute_equity_table_fast`**

Add the following function in `postflop_exhaustive.rs`, after the existing `compute_equity_table`:

```rust
use crate::showdown_equity::rank_to_ordinal;

/// Restructured equity table computation: evaluate each combo once per board.
///
/// Mathematically equivalent to `compute_equity_table` but avoids redundant
/// hand evaluations. For each (turn, river) board, ranks all concrete combos
/// once, then derives equity via integer comparison.
///
/// Complexity: O(combos × boards × eval) + O(169² × combos² × boards × cmp)
/// vs original: O(169² × combos² × boards × eval)
#[allow(clippy::cast_precision_loss)]
pub fn compute_equity_table_fast(
    combo_map: &[Vec<(Card, Card)>],
    flop: [Card; 3],
) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;

    // ── Build flat combo index ──────────────────────────────────────────
    // combo_cards[i] = (c1, c2) for flat combo index i
    // combo_canonical[i] = which canonical hand (0..168) this combo belongs to
    // combo_mask[i] = bitmask of the two cards (for conflict detection)
    // canonical_range[h] = start..end range in the flat arrays for canonical hand h
    let total_combos: usize = combo_map.iter().map(|v| v.len()).sum();
    let mut combo_cards: Vec<(Card, Card)> = Vec::with_capacity(total_combos);
    let mut combo_canonical: Vec<usize> = Vec::with_capacity(total_combos);
    let mut combo_mask: Vec<u64> = Vec::with_capacity(total_combos);
    let mut canonical_range: Vec<std::ops::Range<usize>> = Vec::with_capacity(n);

    for (idx, combos) in combo_map.iter().enumerate() {
        let start = combo_cards.len();
        for &(c1, c2) in combos {
            combo_cards.push((c1, c2));
            combo_canonical.push(idx);
            combo_mask.push((1u64 << card_bit(c1)) | (1u64 << card_bit(c2)));
        }
        canonical_range.push(start..combo_cards.len());
    }

    // ── Enumerate all (turn, river) boards ──────────────────────────────
    let deck = all_cards_vec();
    let flop_mask: u64 = (1u64 << card_bit(flop[0]))
        | (1u64 << card_bit(flop[1]))
        | (1u64 << card_bit(flop[2]));

    let mut boards: Vec<(Card, Card, u64)> = Vec::with_capacity(1176);
    for (ti, &turn) in deck.iter().enumerate() {
        let turn_bit = 1u64 << card_bit(turn);
        if flop_mask & turn_bit != 0 {
            continue;
        }
        for &river in &deck[ti + 1..] {
            let river_bit = 1u64 << card_bit(river);
            if flop_mask & river_bit != 0 {
                continue;
            }
            boards.push((turn, river, flop_mask | turn_bit | river_bit));
        }
    }

    // ── Phase 1+2: parallel over boards, accumulate equity ──────────────
    // Each rayon task processes a chunk of boards and accumulates into a
    // local (equity_sum, count) array of size n×n.
    let accum = boards
        .par_iter()
        .fold(
            || vec![(0.0f64, 0u64); n * n],
            |mut accum, &(turn, river, board_mask)| {
                let board = [flop[0], flop[1], flop[2], turn, river];

                // Phase 1: evaluate all non-conflicting combos on this board
                let mut ranks = vec![0u32; total_combos];
                for (ci, &(c1, c2)) in combo_cards.iter().enumerate() {
                    if board_mask & combo_mask[ci] != 0 {
                        continue; // combo conflicts with board
                    }
                    ranks[ci] = rank_to_ordinal(rank_hand([c1, c2], &board));
                }

                // Phase 2: pairwise comparison by canonical hand
                for hero_idx in 0..n {
                    let hero_range = &canonical_range[hero_idx];
                    if hero_range.is_empty() {
                        continue;
                    }
                    for opp_idx in 0..n {
                        let opp_range = &canonical_range[opp_idx];
                        if opp_range.is_empty() {
                            continue;
                        }

                        let cell = &mut accum[hero_idx * n + opp_idx];

                        for hi in hero_range.clone() {
                            let hr = ranks[hi];
                            if hr == 0 && board_mask & combo_mask[hi] != 0 {
                                continue;
                            }
                            let hm = combo_mask[hi];

                            for oi in opp_range.clone() {
                                let or = ranks[oi];
                                if or == 0 && board_mask & combo_mask[oi] != 0 {
                                    continue;
                                }
                                // Skip hero/opp card conflict
                                if hm & combo_mask[oi] != 0 {
                                    continue;
                                }
                                cell.0 += match hr.cmp(&or) {
                                    std::cmp::Ordering::Greater => 1.0,
                                    std::cmp::Ordering::Equal => 0.5,
                                    std::cmp::Ordering::Less => 0.0,
                                };
                                cell.1 += 1;
                            }
                        }
                    }
                }

                accum
            },
        )
        .reduce(
            || vec![(0.0f64, 0u64); n * n],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i].0 += b[i].0;
                    a[i].1 += b[i].1;
                }
                a
            },
        );

    // ── Derive equity table ─────────────────────────────────────────────
    let mut table = vec![f64::NAN; n * n];
    for i in 0..n * n {
        let (eq_sum, count) = accum[i];
        if count > 0 {
            table[i] = eq_sum / count as f64;
        }
    }
    table
}
```

**Important correctness note:** The sentinel `ranks[ci] == 0` check must also verify `board_mask & combo_mask[ci] != 0` because `rank_to_ordinal(Rank::HighCard(0))` could theoretically be 0. In practice, the lowest 7-card high-card hand produces a non-zero kicker, but we check the mask to be safe. An alternative: use `u32::MAX` as sentinel and initialize ranks to `u32::MAX`:

```rust
let mut ranks = vec![u32::MAX; total_combos];
// ... in Phase 2:
if ranks[hi] == u32::MAX { continue; }
```

This avoids the double-check and is cleaner. Use this approach in the implementation.

**Step 4: Run correctness test**

Run: `cargo test -p poker-solver-core restructured_equity_table -- --ignored --nocapture`
Expected: PASS (values match within 1e-10)

**Step 5: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs crates/core/src/showdown_equity.rs
git commit -m "feat(core): add restructured compute_equity_table_fast

Evaluate each concrete combo once per board instead of once per opponent
matchup. Two-phase: rank computation then integer comparison.
Parallel fold/reduce over boards via rayon."
```

---

### Task 3: Swap new for old and verify all tests pass

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs`

**Step 1: Rename functions**

In `postflop_exhaustive.rs`:
- Rename `compute_equity_table` → `compute_equity_table_reference` (keep for test comparison)
- Rename `compute_equity_table_fast` → `compute_equity_table`
- Add `#[cfg(test)]` gate on `compute_equity_table_reference`

**Step 2: Update the correctness test**

Update `restructured_equity_table_matches_original` to call `compute_equity_table_reference` and `compute_equity_table`.

**Step 3: Run full test suite**

Run: `cargo test -p poker-solver-core`
Expected: all pass (no callers changed, just the internal implementation)

Also run: `cargo test -p poker-solver-trainer`
Expected: all pass

**Step 4: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "refactor(core): replace compute_equity_table with restructured version

The old implementation is kept as compute_equity_table_reference behind
#[cfg(test)] for correctness verification."
```

---

### Task 4: Add `RankArrayCache` module

Disk cache for per-board rank arrays. Each canonical flop stores rank ordinals for every concrete combo on every (turn, river) board. Enables deriving equity tables without any hand evaluation.

**Files:**
- Create: `crates/core/src/preflop/rank_array_cache.rs`
- Modify: `crates/core/src/preflop/mod.rs` (add module)

**Step 1: Write the failing test**

Create `crates/core/src/preflop/rank_array_cache.rs`:

```rust
//! Disk cache for per-board rank arrays.
//!
//! For each canonical flop, stores a u32 rank ordinal for every concrete
//! combo on every (turn, river) board. The equity table can then be derived
//! from cached ranks via pure integer comparison — no hand evaluation needed.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preflop::postflop_hands::{build_combo_map, canonical_flops};
    use test_macros::timed_test;

    #[timed_test(10)]
    fn save_load_round_trip() {
        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);
        let data = compute_rank_arrays(&combo_map, flop);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test_ranks.bin");

        let cache = RankArrayCache {
            flops: vec![flop],
            entries: vec![data],
        };
        cache.save(&path).unwrap();

        let loaded = RankArrayCache::load(&path).unwrap();
        assert_eq!(loaded.flops.len(), 1);
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].board_ranks.len(), cache.entries[0].board_ranks.len());
    }

    #[timed_test(10)]
    fn load_returns_none_for_missing_file() {
        assert!(RankArrayCache::load(std::path::Path::new("/tmp/nonexistent_rank_cache.bin")).is_none());
    }

    #[timed_test(10)]
    fn derive_equity_matches_direct_computation() {
        use crate::preflop::postflop_exhaustive::compute_equity_table;
        use crate::preflop::postflop_hands::NUM_CANONICAL_HANDS;

        let flops = canonical_flops();
        let flop = flops[0];
        let combo_map = build_combo_map(&flop);

        let rank_data = compute_rank_arrays(&combo_map, flop);
        let derived = derive_equity_table(&rank_data, &combo_map);
        let direct = compute_equity_table(&combo_map, flop);

        let n = NUM_CANONICAL_HANDS;
        for h in 0..n {
            for o in 0..n {
                let idx = h * n + o;
                let a = direct[idx];
                let b = derived[idx];
                if a.is_nan() {
                    assert!(b.is_nan(), "mismatch at [{h}][{o}]");
                } else {
                    assert!(
                        (a - b).abs() < 1e-10,
                        "mismatch at [{h}][{o}]: direct={a}, derived={b}",
                    );
                }
            }
        }
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core rank_array_cache -- --nocapture`
Expected: compile error — types not defined

**Step 3: Implement the module**

Key types and functions:

```rust
use std::io;
use std::path::Path;

use rayon::prelude::*;
use rs_poker::core::Card;
use serde::{Deserialize, Serialize};

use super::postflop_hands::{all_cards_vec, build_combo_map, NUM_CANONICAL_HANDS};
use crate::showdown_equity::{rank_hand, rank_to_ordinal};

const MAGIC: [u8; 4] = *b"RKAC";
const VERSION: u32 = 1;

/// Rank data for one canonical flop: all boards × all combos.
#[derive(Clone, Serialize, Deserialize)]
pub struct FlopRankData {
    /// (turn, river) pairs as encoded card bytes.
    pub board_cards: Vec<(u8, u8)>,
    /// Flat combo cards: (c1_encoded, c2_encoded) for each combo in combo_map order.
    pub combo_cards: Vec<(u8, u8)>,
    /// Canonical hand index for each combo.
    pub combo_canonical: Vec<u16>,
    /// Rank ordinals: board_ranks[board_idx * num_combos + combo_idx].
    /// u32::MAX = combo conflicts with board.
    pub board_ranks: Vec<u32>,
}

/// In-memory rank array cache.
pub struct RankArrayCache {
    pub flops: Vec<[Card; 3]>,
    pub entries: Vec<FlopRankData>,
}

fn encode_card(c: Card) -> u8 {
    c.value as u8 * 4 + c.suit as u8
}

fn decode_card(b: u8) -> Card {
    use rs_poker::core::{Suit, Value};
    let value = unsafe { std::mem::transmute::<u8, Value>(b / 4) };
    let suit = unsafe { std::mem::transmute::<u8, Suit>(b % 4) };
    Card { value, suit }
}

fn card_bit(card: Card) -> u8 {
    card.value as u8 * 4 + card.suit as u8
}

/// Compute rank arrays for a single flop.
pub fn compute_rank_arrays(
    combo_map: &[Vec<(Card, Card)>],
    flop: [Card; 3],
) -> FlopRankData {
    let deck = all_cards_vec();
    let flop_mask: u64 = (1u64 << card_bit(flop[0]))
        | (1u64 << card_bit(flop[1]))
        | (1u64 << card_bit(flop[2]));

    // Build flat combo list
    let mut combo_cards_raw: Vec<(Card, Card)> = Vec::new();
    let mut combo_canonical: Vec<u16> = Vec::new();
    let mut combo_masks: Vec<u64> = Vec::new();
    for (idx, combos) in combo_map.iter().enumerate() {
        for &(c1, c2) in combos {
            combo_cards_raw.push((c1, c2));
            combo_canonical.push(idx as u16);
            combo_masks.push((1u64 << card_bit(c1)) | (1u64 << card_bit(c2)));
        }
    }
    let num_combos = combo_cards_raw.len();

    // Enumerate boards
    let mut boards: Vec<(Card, Card, u64)> = Vec::new();
    for (ti, &turn) in deck.iter().enumerate() {
        let tb = 1u64 << card_bit(turn);
        if flop_mask & tb != 0 { continue; }
        for &river in &deck[ti + 1..] {
            let rb = 1u64 << card_bit(river);
            if flop_mask & rb != 0 { continue; }
            boards.push((turn, river, flop_mask | tb | rb));
        }
    }

    // Compute ranks: parallel over boards
    let board_ranks: Vec<Vec<u32>> = boards
        .par_iter()
        .map(|&(turn, river, board_mask)| {
            let board = [flop[0], flop[1], flop[2], turn, river];
            let mut ranks = vec![u32::MAX; num_combos];
            for (ci, &(c1, c2)) in combo_cards_raw.iter().enumerate() {
                if board_mask & combo_masks[ci] != 0 { continue; }
                ranks[ci] = rank_to_ordinal(rank_hand([c1, c2], &board));
            }
            ranks
        })
        .collect();

    // Flatten
    let mut flat_ranks = Vec::with_capacity(boards.len() * num_combos);
    for board_rank in &board_ranks {
        flat_ranks.extend_from_slice(board_rank);
    }

    FlopRankData {
        board_cards: boards.iter().map(|&(t, r, _)| (encode_card(t), encode_card(r))).collect(),
        combo_cards: combo_cards_raw.iter().map(|&(c1, c2)| (encode_card(c1), encode_card(c2))).collect(),
        combo_canonical,
        board_ranks: flat_ranks,
    }
}

/// Derive a 169×169 equity table from precomputed rank arrays.
///
/// Pure integer comparison — no hand evaluation.
#[allow(clippy::cast_precision_loss)]
pub fn derive_equity_table(
    data: &FlopRankData,
    combo_map: &[Vec<(Card, Card)>],
) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    let num_combos = data.combo_cards.len();
    let num_boards = data.board_cards.len();

    // Build combo masks for hero/opp conflict detection
    let combo_masks: Vec<u64> = data.combo_cards.iter().map(|&(c1, c2)| {
        (1u64 << c1) | (1u64 << c2)
    }).collect();

    // Build canonical ranges
    let mut canonical_range: Vec<std::ops::Range<usize>> = Vec::with_capacity(n);
    let mut ci = 0;
    for combos in combo_map {
        let start = ci;
        ci += combos.len();
        canonical_range.push(start..ci);
    }

    // Parallel fold/reduce over boards
    let accum = (0..num_boards)
        .into_par_iter()
        .fold(
            || vec![(0.0f64, 0u64); n * n],
            |mut accum, board_idx| {
                let ranks = &data.board_ranks[board_idx * num_combos..(board_idx + 1) * num_combos];

                for hero_idx in 0..n {
                    let hr = &canonical_range[hero_idx];
                    if hr.is_empty() { continue; }
                    for opp_idx in 0..n {
                        let or = &canonical_range[opp_idx];
                        if or.is_empty() { continue; }
                        let cell = &mut accum[hero_idx * n + opp_idx];

                        for hi in hr.clone() {
                            let hero_rank = ranks[hi];
                            if hero_rank == u32::MAX { continue; }
                            let hm = combo_masks[hi];

                            for oi in or.clone() {
                                let opp_rank = ranks[oi];
                                if opp_rank == u32::MAX { continue; }
                                if hm & combo_masks[oi] != 0 { continue; }
                                cell.0 += match hero_rank.cmp(&opp_rank) {
                                    std::cmp::Ordering::Greater => 1.0,
                                    std::cmp::Ordering::Equal => 0.5,
                                    std::cmp::Ordering::Less => 0.0,
                                };
                                cell.1 += 1;
                            }
                        }
                    }
                }
                accum
            },
        )
        .reduce(
            || vec![(0.0f64, 0u64); n * n],
            |mut a, b| {
                for i in 0..a.len() {
                    a[i].0 += b[i].0;
                    a[i].1 += b[i].1;
                }
                a
            },
        );

    let mut table = vec![f64::NAN; n * n];
    for i in 0..n * n {
        let (eq_sum, count) = accum[i];
        if count > 0 {
            table[i] = eq_sum / count as f64;
        }
    }
    table
}

#[derive(Serialize, Deserialize)]
struct CacheData {
    magic: [u8; 4],
    version: u32,
    num_flops: u32,
    flop_bytes: Vec<u8>,
    entries: Vec<FlopRankData>,
}

impl RankArrayCache {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut flop_bytes = Vec::with_capacity(self.flops.len() * 3);
        for flop in &self.flops {
            flop_bytes.push(encode_card(flop[0]));
            flop_bytes.push(encode_card(flop[1]));
            flop_bytes.push(encode_card(flop[2]));
        }
        let data = CacheData {
            magic: MAGIC,
            version: VERSION,
            num_flops: self.flops.len() as u32,
            flop_bytes,
            entries: self.entries.clone(),
        };
        let raw = bincode::serialize(&data)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        // Compress with zstd level 3 (good speed/ratio balance)
        let compressed = zstd::encode_all(std::io::Cursor::new(&raw), 3)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(path, compressed)
    }

    pub fn load(path: &Path) -> Option<Self> {
        let compressed = std::fs::read(path).ok()?;
        let raw = zstd::decode_all(std::io::Cursor::new(&compressed)).ok()?;
        let data: CacheData = bincode::deserialize(&raw).ok()?;
        if data.magic != MAGIC || data.version != VERSION {
            return None;
        }
        let num_flops = data.num_flops as usize;
        if data.flop_bytes.len() != num_flops * 3 || data.entries.len() != num_flops {
            return None;
        }
        let mut flops = Vec::with_capacity(num_flops);
        for i in 0..num_flops {
            let base = i * 3;
            flops.push([
                decode_card(data.flop_bytes[base]),
                decode_card(data.flop_bytes[base + 1]),
                decode_card(data.flop_bytes[base + 2]),
            ]);
        }
        Some(Self {
            flops,
            entries: data.entries,
        })
    }

    pub fn num_flops(&self) -> usize {
        self.flops.len()
    }

    /// Look up rank data for a specific flop.
    pub fn get_flop_data(&self, flop: &[Card; 3]) -> Option<&FlopRankData> {
        let key = [encode_card(flop[0]), encode_card(flop[1]), encode_card(flop[2])];
        self.flops.iter().enumerate().find_map(|(i, f)| {
            let fk = [encode_card(f[0]), encode_card(f[1]), encode_card(f[2])];
            if fk == key { Some(&self.entries[i]) } else { None }
        })
    }
}
```

**Step 4: Add module to `mod.rs`**

In `crates/core/src/preflop/mod.rs`, add:
```rust
pub mod rank_array_cache;
```

**Step 5: Add `zstd` dependency to Cargo.toml**

In `crates/core/Cargo.toml`, add to `[dependencies]`:
```toml
zstd = "0.13"
```

**Step 6: Run tests**

Run: `cargo test -p poker-solver-core rank_array_cache -- --nocapture`
Expected: compile pass, `save_load_round_trip` and `load_returns_none_for_missing_file` pass.
The `derive_equity_matches` test needs the `#[ignore]` tag and runs with:
Run: `cargo test -p poker-solver-core derive_equity_matches -- --ignored --nocapture`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/core/src/preflop/rank_array_cache.rs crates/core/src/preflop/mod.rs crates/core/Cargo.toml
git commit -m "feat(core): add RankArrayCache for per-board rank persistence

Stores u32 rank ordinals per concrete combo per (turn,river) board.
Equity tables can be derived from cached ranks via pure integer
comparison. Uses zstd compression for disk storage."
```

---

### Task 5: Integrate rank cache into trainer

Wire the rank array cache into the `PrecomputeEquity` and `SolvePostflop` commands.

**Files:**
- Modify: `crates/trainer/src/main.rs`

**Step 1: Update `PrecomputeEquity` command**

In the `Commands::PrecomputeEquity` handler (around line 166), change the flow to:
1. Try loading rank cache
2. If not found, compute rank arrays + save
3. Derive equity tables from rank arrays
4. Save equity table cache as before

```rust
Commands::PrecomputeEquity { output } => {
    use poker_solver_core::preflop::rank_array_cache::{
        compute_rank_arrays, derive_equity_table, RankArrayCache,
    };

    let rank_cache_path = Path::new("cache/rank_arrays.bin");
    let total = 1755_u64;

    // Try loading existing rank cache
    let rank_cache = if let Some(cache) = RankArrayCache::load(rank_cache_path) {
        eprintln!("Loaded rank arrays for {} flops from cache", cache.num_flops());
        cache
    } else {
        // Compute rank arrays for all canonical flops
        let pb = ProgressBar::new(total);
        pb.set_style(/* same style as before */);
        pb.enable_steady_tick(Duration::from_millis(200));
        eprintln!("Computing rank arrays for {total} canonical flops...");

        let flops = canonical_flops();
        let start = Instant::now();
        let completed = AtomicU32::new(0);

        let entries: Vec<_> = flops
            .par_iter()
            .map(|flop| {
                let combo_map = build_combo_map(flop);
                let data = compute_rank_arrays(&combo_map, *flop);
                let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                pb.set_position(done as u64);
                data
            })
            .collect();

        pb.finish_with_message("done");
        let elapsed = start.elapsed();
        eprintln!("Computed rank arrays in {:.1}s", elapsed.as_secs_f64());

        let cache = RankArrayCache { flops, entries };
        if let Err(e) = cache.save(rank_cache_path) {
            eprintln!("Warning: failed to save rank cache: {e}");
        } else {
            eprintln!("Rank cache saved to {}", rank_cache_path.display());
        }
        cache
    };

    // Derive equity tables from rank arrays
    eprintln!("Deriving equity tables from rank arrays...");
    let flops = rank_cache.flops.clone();
    let tables: Vec<Vec<f64>> = flops
        .par_iter()
        .enumerate()
        .map(|(i, flop)| {
            let combo_map = build_combo_map(flop);
            derive_equity_table(&rank_cache.entries[i], &combo_map)
        })
        .collect();

    let eq_cache = EquityTableCache { flops, tables_flat: /* flatten */ };
    eq_cache.save(Path::new(&output))?;
    eprintln!("Equity table cache saved to {output}");
}
```

**Step 2: Update `SolvePostflop` to use rank cache for equity table derivation**

In the equity table loading section (around line 407), add a fallback path:
1. Try equity table cache (existing, fast)
2. If miss → try rank array cache → derive equity tables
3. If miss → compute from scratch (restructured, also fast)

**Step 3: Run trainer tests**

Run: `cargo test -p poker-solver-trainer`
Expected: all pass

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat(trainer): integrate rank array cache into precompute and solve commands"
```

---

### Task 6: Benchmark and final cleanup

**Files:**
- Modify: `crates/core/benches/equity_table_bench.rs`
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs` (cleanup)

**Step 1: Update benchmark**

```rust
use poker_solver_core::preflop::rank_array_cache::{compute_rank_arrays, derive_equity_table};

fn bench_equity_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("equity_table");
    group.sample_size(10);

    let flop = flop_akq();
    let combo_map = build_combo_map(&flop);

    group.bench_function("AKQr_restructured", |b| {
        b.iter(|| compute_equity_table(&combo_map, flop));
    });

    // Bench rank array computation separately
    group.bench_function("AKQr_rank_arrays", |b| {
        b.iter(|| compute_rank_arrays(&combo_map, flop));
    });

    // Bench equity derivation from cached ranks
    let rank_data = compute_rank_arrays(&combo_map, flop);
    group.bench_function("AKQr_derive_equity", |b| {
        b.iter(|| derive_equity_table(&rank_data, &combo_map));
    });

    group.finish();
}
```

**Step 2: Run benchmark**

Run: `cargo bench -p poker-solver-core -- equity_table`

Expected output (approximate):
- `AKQr_restructured`: ~1-3 seconds (combined rank + derive)
- `AKQr_rank_arrays`: ~30-50 ms (Phase 1 only — hand evaluation)
- `AKQr_derive_equity`: ~1-3 seconds (Phase 2 only — integer comparison)

Compare against the previous benchmark result for the old `compute_equity_table` (which should be ~15-25 seconds).

**Step 3: Run full test suite**

Run: `cargo test`
Run: `cargo clippy`
Expected: all pass, no warnings

**Step 4: Commit**

```bash
git add crates/core/benches/equity_table_bench.rs
git commit -m "bench: update equity table benchmark for restructured computation"
```

---

## Key Implementation Notes

### Correctness: Accumulation Order

The restructured loop sums equity contributions in (board, hero, opp) order vs the original's (hero, opp, board) order. Both sum the same set of values, so the mathematical result is identical. Floating-point differences may appear at ~1e-15 due to non-associative addition. The correctness test uses 1e-10 tolerance.

### Sentinel for Invalid Combos

Use `u32::MAX` (not 0) as the sentinel for combos that conflict with the board. This avoids ambiguity since `rank_to_ordinal` can theoretically return 0 for `Rank::HighCard(0)`.

### Cache Size

Raw rank array data: ~6.2 MB per flop × 1,755 flops = ~10.9 GB uncompressed. With zstd compression at level 3, expect ~2-4 GB on disk. The existing equity table cache is ~390 MB for comparison.

### Performance Profile

With the restructured loop, the bottleneck shifts from hand evaluation to pairwise comparison:
- Phase 1 (rank computation): ~30 ms per flop (trivially fast)
- Phase 2 (comparison): ~1-3 seconds per flop
- Total per flop: ~1-3 seconds (was ~15-25 seconds)
- All 1,755 flops parallel on 8 cores: ~5-10 minutes (was hours)
