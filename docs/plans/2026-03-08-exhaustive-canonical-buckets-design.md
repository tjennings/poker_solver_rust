# Exhaustive Canonical Bucket Files — Design

## Overview

Replace the random-sample-based clustering pipeline with exhaustive canonical board enumeration. Produce bucket files with deterministic board indices, stored alongside the blueprint. At MCCFR time, canonicalize the deal's board, look up the index, and read the precomputed bucket — eliminating runtime equity computation entirely.

## Canonical Board Enumeration

Street-relative enumeration:
- **Flop**: All 1,755 canonical 3-card boards (existing `all_flops()`)
- **Turn**: For each canonical flop, enumerate 49 possible turn cards, canonicalize the 4-card board, deduplicate → ~15K canonical turns
- **River**: For each canonical turn, enumerate 48 possible river cards, canonicalize the 5-card board, deduplicate → ~700K canonical rivers

Each canonical board carries a **weight** (number of raw boards it represents). Weights are used in k-means so centroids reflect real-play frequency.

Suit isomorphisms are theoretically lossless — the same suit permutation applied to board and hole cards preserves all strategic information. This is universally used by production solvers (Pluribus, PioSOLVER).

## BucketFile Format Extension

Extend `BucketFile` to store the canonical boards in the file:

```
[header]
  street, bucket_count, board_count, combos_per_board (1326)
[board table]           ← NEW
  board_count × packed_board (u64 each, canonical cards encoded)
[bucket data]
  board_count × 1326 × u16
```

The board table makes the file self-contained — at load time, build a `HashMap<u64, u32>` (packed canonical board → board_idx) for O(1) lookup.

## Compact Board Key

Encode canonical cards as a `u64` for hashing:
- Each card = 8 bits (6 bits value + 2 bits suit), up to 5 cards = 40 bits
- Cards sorted in canonical order, packed into a u64
- Implements `Hash + Eq` for HashMap key

## MCCFR Lookup Flow

```
deal.board → CanonicalBoard::from_cards(visible_board)
           → apply suit_mapping to hole_cards
           → pack canonical board → u64 key
           → HashMap lookup → board_idx
           → combo_idx from permuted hole cards
           → bucket = buckets[board_idx * 1326 + combo_idx]
```

## Clustering Pipeline Changes

- Replace `sample_boards()` / `sample_flop_boards()` / `sample_turn_boards()` with exhaustive canonical enumeration
- Weight each (board, combo) sample by the board's isomorphism count in k-means
- Output bucket files with the new format (board table + buckets)
- Cache bucket files in the blueprint output directory — skip if files already exist (resume-safe)

## File Sizes (estimates)

| Street | Boards | Bucket data | Board table | Total |
|-|-|-|-|-|
| Flop | ~1,755 | ~4.5 MB | ~14 KB | ~4.5 MB |
| Turn | ~15K | ~39 MB | ~120 KB | ~39 MB |
| River | ~700K | ~1.8 GB | ~5.6 MB | ~1.8 GB |

River is large but manageable for disk. Can be mmap'd for memory efficiency as a follow-up.

## Out of Scope

- Preflop bucketing (already uses 169 canonical hands)
- Changing the k-means algorithm itself
- Memory-mapping optimization (follow-up)
