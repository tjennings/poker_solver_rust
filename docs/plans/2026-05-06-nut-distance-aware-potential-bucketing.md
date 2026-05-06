# Nut-Distance-Aware Potential Bucketing

**Status:** Draft implementation spec
**Bean:** `poker_solver_rust-heuc`

## Goal

Improve postflop card abstraction so buckets preserve both:

- **Potential awareness:** where a hand can go across future public cards.
- **Nut dominance:** how dominant or dominated the hand is within its current
  made-hand family on this board.

Preflop remains lossless canonical 169-hand mapping. Lossy strategic bucketing
starts on the flop.

## Core Metric

For two postflop situations `a` and `b` on the same street:

```text
D(a, b) =
  wp * normalized_emd(future_bucket_dist(a), future_bucket_dist(b))
+ wn * normalized_nut_distance(nut_features(a), nut_features(b))
```

Initial defaults:

```text
wp = 1.0
wn sweep = [0.0, 0.1, 0.25, 0.5, 1.0]
default candidate = 0.25
```

`wn = 0.0` is the existing potential-only baseline. The production default
should be selected only after diagnostics show that dominated made hands split
without materially degrading potential-aware cohesion.

## Normalization

Both terms must be comparable before weighting.

`normalized_emd`:

- Use weighted 1-D EMD over ordered child buckets.
- Normalize by observed P95 EMD from the sampled feature set, clamped to `1.0`.
- Fall back to theoretical max EMD when the sample distribution is degenerate.

`normalized_nut_distance`:

- Encode each component into `[0, 1]`.
- Cap rank gaps at a street-specific maximum before scaling.
- Use class-aware distance so irrelevant nut features do not dominate unrelated
  hand families.

## Nut Feature Contract

Nut features are board-aware and computed for `(hole, board)`.

```text
NutFeatures {
  made_class: enum,
  class_nut_rank: u8,
  class_gap: u8,
  global_rank_percentile: f32,
  blocker_to_nuts: bool,
  redraw_to_nuts: bool,
  dominance_margin: f32,
}
```

`made_class` examples:

- HighCard
- Pair
- TwoPair
- Trips
- Straight
- Flush
- FullHouse
- Quads
- StraightFlush

`class_gap` is the distance from the best possible hand in that same made-hand
family on this board. Examples:

- Nut flush: `class_gap = 0`
- Second-nut flush: `class_gap = 1`
- K-high flush on an A-high suited board: worse than K-high flush where the ace
  of that suit is impossible or blocked
- Top set: lower gap than middle or bottom set
- Nut straight: lower gap than lower straight when multiple straights exist

`dominance_margin` measures how many legal opponent combos in the same made
family beat this hand, normalized by legal opponent combos in that family.

## Complementarity Rules

Potential awareness should remain the primary structure.

1. Nut distance is a regularizer, not a replacement.
2. Nut distance is strongest within the same made-hand family.
3. Across unrelated hand families, use made-class distance and global equity
   percentile instead of raw class gaps.
4. River dominance should be encoded first. Turn and flop then inherit richer
   future buckets through the existing bottom-up potential-aware recursion.

This avoids turning the abstraction into a hand-class heuristic while still
preventing high-profit dominated made hands from collapsing together.

## Implementation Plan

1. Add a river-only `NutFeatures` diagnostic module.
   - No clustering behavior change.
   - Produce per-bucket collision reports for flushes, straights, sets, and
     boats.

2. Add optional river centroid/features support.
   - Compare existing river equity buckets against nut-aware river distance.
   - Record `wn` sweep diagnostics.

3. Extend river clustering metric.
   - Keep scalar equity ordering for bucket IDs.
   - Add nut-distance regularization during assignment.
   - Preserve centroid EVs for downstream turn EMD gaps.

4. Rebuild turn/flop via existing potential-aware recursion.
   - Do not add flop/turn side features until river-aware buckets have been
     evaluated.

5. Promote to configurable pipeline option.
   - Example config:

```yaml
clustering:
  nut_distance:
    enabled: true
    weight: 0.25
    normalize: p95
```

## Diagnostics

For each candidate `wn`, report:

- Intra-bucket future EMD.
- Nut collision rate.
- Bucket occupancy and empty bucket count.
- Flush dominance collisions:
  - nut flush grouped with K-high or worse flush
  - second-nut flush grouped with low flush
- Straight dominance collisions:
  - nut straight grouped with lower straight
- Set/boat dominance collisions:
  - top set grouped with bottom set on same texture
  - nut boat grouped with dominated boat

Preferred acceptance threshold:

```text
nut_collision_rate improves materially vs wn=0.0
intra_bucket_future_emd worsens by <= 5%
empty_bucket_count does not increase materially
```

## Open Questions

- Should `dominance_margin` count only same-family hands or all legal opponent
  hands that beat this hand?
- Should blocker-to-nuts be separate from made-hand nut distance for bluffing
  relevance?
- Should turn/flop later add current-street redraw features, or should all
  future potential stay encoded through child buckets?
