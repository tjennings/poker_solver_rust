# Standard Imperfect-Recall Abstraction Design

**Date:** 2026-02-22
**Status:** Approved
**Motivation:** Align our postflop abstraction with the standard approach used by Pluribus/Libratus — independent per-street clustering with imperfect recall, equity histograms with EMD, canonical board enumeration.

## Context

The current abstraction system uses:
- Texture-based board grouping
- Global k-means on raw EHS values (single scalar)
- A hierarchical transition table (`flop_bucket × texture → turn_bucket`) that appears layered but is informationally flat — the global k-means doesn't condition on parent buckets
- L2 distance on single EHS values

The standard in the literature (Johanson 2013, Pluribus 2019, Libratus 2017) uses:
- Independent per-street clustering with imperfect recall
- 10-bin equity histograms as feature vectors
- EMD (≡ L2 on CDFs for 1D distributions) as distance metric
- No cross-street dependencies

This redesign closes two gaps: (1) the clustering feature (histogram vs scalar) and (2) the structure (independent vs pseudo-hierarchical). It also positions us for Pluribus-style real-time subgame solving.

## Design

### Remove

- Texture-based board grouping (`FlopTexture`, `TurnTexture`, `RiverTexture` usage in bucketing)
- Transition tables (`turn_buckets[flop_bucket][texture]`, `river_buckets[turn_bucket][texture]`)
- `HandBucketMapping` hierarchical structure
- `cluster_global` that pools across textures
- `compute_all_flop_features`, `compute_all_turn_features`, `compute_all_river_features` (texture-based)
- `build_transition_table`

### Add

#### 1. Equity Histogram Feature Vectors

For each (hand, board) situation, build a 10-bin histogram of equity over future board runouts:

- **Flop:** Given `(hole_cards, flop_3)`, enumerate all 47 turn cards. For each, compute equity vs uniform random opponent. Bin the 47 equity values into 10 equal-width bins over [0, 1]. Convert to CDF (cumulative sum, normalized).
- **Turn:** Given `(hole_cards, flop_3, turn_1)`, enumerate all 46 river cards. Same process → 10-bin CDF.
- **River:** No future cards. Feature = raw equity scalar. Cluster on equity directly.

The CDF representation means L2 distance between CDFs equals EMD for 1D distributions.

Type: `[f32; 10]` for flop/turn features, `f32` for river.

#### 2. Independent Per-Street Clustering

Each street runs k-means independently:

- **Flop:** Collect feature vectors for all `(hand_idx, canonical_flop)` pairs. Run k-means with L2 on CDF vectors. Produces `F` flop buckets.
- **Turn:** Collect feature vectors for all `(hand_idx, canonical_flop, turn_card)` pairs. Independent k-means. Produces `T` turn buckets.
- **River:** Collect feature vectors for all `(hand_idx, canonical_board_5)` pairs. K-means on scalar equity. Produces `R` river buckets.

No street knows about any other street's bucket assignments. This is **imperfect recall by design** — the player "forgets" their prior-street bucket, spending the full bucket budget on present-state resolution.

#### 3. Canonical Board Enumeration

Replace texture-based board grouping with suit-isomorphic canonical board enumeration:

- Enumerate all C(50,3) = 19,600 flop combos (after removing hole cards), reduce to canonical forms via suit isomorphism
- Similarly for turn (add 1 card) and river (add 2 cards)
- This is more faithful to the true game and eliminates the texture abstraction layer

#### 4. New Data Model

```rust
/// Per-street bucket assignments. Each street is independent (imperfect recall).
pub struct StreetBuckets {
    /// flop_buckets[situation_idx] → bucket_id
    /// situation_idx encodes (hand_idx, canonical_flop)
    pub flop_buckets: Vec<u16>,
    pub num_flop_buckets: u16,

    /// turn_buckets[situation_idx] → bucket_id
    /// situation_idx encodes (hand_idx, canonical_flop, turn_card)
    pub turn_buckets: Vec<u16>,
    pub num_turn_buckets: u16,

    /// river_buckets[situation_idx] → bucket_id
    /// situation_idx encodes (hand_idx, canonical_board_5)
    pub river_buckets: Vec<u16>,
    pub num_river_buckets: u16,
}
```

#### 5. Bucket Equity Tables

Same concept as current `StreetEquity` — `equity[bucket_a][bucket_b]` per street. Dimensioned by the (smaller, default 500) bucket counts.

#### 6. Info Set Key Changes

The hand field in the info key stores the current street's bucket ID. No transition table lookup needed — at each chance node, look up the new bucket directly from the hand situation using the independent per-street mapping.

### Configuration

All bucket counts configurable in YAML, defaulting to Pluribus-like values:

```yaml
num_flop_buckets: 500
num_turn_buckets: 500
num_river_buckets: 500
histogram_bins: 10
kmeans_max_iter: 100
```

### Future: Real-Time Subgame Solving

This design is Pluribus-compatible. The blueprint uses 500-bucket abstractions. When we add real-time solving:

- Blueprint strategy serves as the "trunk" / warm start
- Subgame solver operates losslessly (or with finer abstraction) within a subgame
- Blueprint bucket equity values serve as leaf node estimates at the subgame boundary
- The independent per-street structure means subgame solving can use its own finer abstraction without conflicting with the blueprint's bucket assignments

## Files Affected

| File | Change |
|-|-|
| `crates/core/src/preflop/hand_buckets.rs` | Major rewrite: new data model, histogram features, independent clustering |
| `crates/core/src/preflop/postflop_abstraction.rs` | Remove transition tables, update to use `StreetBuckets` |
| `crates/core/src/preflop/ehs.rs` | Add histogram/CDF computation alongside existing EHS |
| `crates/core/src/abstract_game.rs` | Update bucket lookups to use independent per-street model |
| `crates/core/src/info_key.rs` | Simplify hand field encoding (no transition lookup) |
| `sample_configurations/*.yaml` | Update config schema with new bucket parameters |
| `crates/trainer/src/main.rs` | Update training pipeline for new abstraction |

## Performance Considerations

- Exhaustive enumeration of 47 turn cards × opponent equity per flop situation is feasible (we already do similar work)
- 500 buckets × 500 buckets = 250K entries per equity table (tiny vs current 5K × 5K = 25M)
- K-means on 10-element vectors is fast — dominated by feature computation, not clustering
- Canonical board enumeration may produce more situations to cluster than texture grouping, but k-means scales linearly with data points
