# Heuristic V3 Bucketing Design

**Date:** 2026-04-08
**Status:** Approved

## Problem

The existing EHS2/EMD bucketing pipeline produces degenerate strategies — e.g., 45s shoves 99.8% all-in because it gets bucketed with hands that legitimately should shove. The abstraction is too lossy: hands that are qualitatively different in strategic terms get grouped together because they have similar scalar equity values.

## Solution

A new `AbstractionMode::HeuristicV3` that assigns postflop buckets using two axes:

- **Nut distance** (configurable, default 6 bits / 64 bins) — fraction of possible opponent holdings that currently beat us on this board. Captures current-street strength with card-removal awareness.
- **Equity delta** (configurable, default 4 bits / 16 bins) — expected change in equity from current street to future streets. Captures draw potential and vulnerability.

**Default: 1,024 fixed bucket IDs per street** (64 on river where delta = 0). Per-flop mapping from holdings to bucket IDs, precomputed and stored.

## Design Rationale

### Why two axes?

- **Nut distance** = "where do I stand now?" Separates TPTK from bottom pair, nut flush from third-nut flush. Handles the K2o vs AKo problem that the current HandClassV2 `intra_class_strength` fails to distinguish (same pair rank = same strength).
- **Equity delta** = "where am I heading?" Separates draws from dead hands at the same current strength. A hand with high nut distance but positive delta is a semi-bluff candidate; the same nut distance with zero delta is trash.

Together they capture the core strategic axis: "am I ahead and vulnerable (bet for protection), ahead and stable (can trap), behind but drawing (semi-bluff), or dead (give up)?"

### Why not made-hand class?

During design exploration, we considered layering made-hand class (pair, set, flush, etc.) as an additional axis. However, nut distance already encodes relative hand strength within and across classes — a set with low nut distance and an overpair with low nut distance are both "strong" and play similarly. Made class is largely redundant when nut distance has sufficient resolution (64 bins).

### Why not draw class/strength?

Similarly, explicit draw classification (flush draw, OESD, gutshot, etc.) is redundant with equity delta. A nut flush draw produces a large positive delta; a weak backdoor produces a small one. The magnitude of the delta captures what matters strategically without needing a taxonomy.

### Why not raw equity?

Nut distance and raw equity are correlated but not identical. Equity averages over future cards (on flop/turn), while nut distance is a current-board measurement. A hand can have high equity due to draw outs but high nut distance (currently behind many holdings). Splitting these into nut distance (current) + equity delta (future direction) is a cleaner decomposition than raw equity alone.

## Coexistence

HeuristicV3 coexists as an alternative to the existing potential-aware EMD pipeline. Users select it via YAML config:

```yaml
abstraction_mode: heuristic_v3
nut_distance_bits: 6    # 64 bins (configurable 2-8)
equity_delta_bits: 4    # 16 bins (configurable 2-8)
```

The existing `ehs2` and `hand_class_v2` modes remain available.

## Computation

### Nut Distance

```
For holding H on board B:
  our_rank = evaluate(H, B)
  total_opponents = 0
  beaten_by = 0
  for each possible opponent holding O (not conflicting with H or B):
    opp_rank = evaluate(O, B)
    total_opponents += 1
    if opp_rank > our_rank:
      beaten_by += 1
  nut_distance = beaten_by / total_opponents  // 0.0 = nuts, 1.0 = air
  nut_distance_bin = quantize(nut_distance, nut_distance_bits)
```

This is a byproduct of the existing `compute_equity()` enumeration — near-zero marginal cost.

### Equity Delta

```
For holding H on board B (flop or turn):
  current_equity = compute_equity(H, B)
  future_equities = []
  for each possible next card C (not in H or B):
    future_equities.push(compute_equity(H, B + C))
  expected_future_equity = mean(future_equities)
  delta = expected_future_equity - current_equity  // range: ~-0.5 to +0.5
  delta_bin = quantize(delta, equity_delta_bits)    // signed, centered at 0
```

On **river**: no future cards, delta = 0 for all hands. Delta bits unused — all hands get delta_bin = midpoint. Effective bucket count = 64 (nut distance only).

### Cost

Per holding on flop: ~47 turn cards × ~990 opponent enumerations = ~46K evaluations. For ~990 holdings per flop: ~46M evaluations per flop. With the existing evaluator this completes in seconds per flop.

Quantization: linear bins across observed range. Percentile-based bins are worth testing if distribution is skewed.

## Precomputation & Storage

**Bucketing phase** (runs once before training):

For each canonical flop (1,755 flops):
1. Enumerate all non-conflicting holdings (~990 per flop)
2. For each holding: compute nut distance + equity delta, quantize to bins
3. For turn: extend mapping for each possible turn card
4. For river: compute nut distance only (delta = 0)
5. Store mapping as compact lookup table: `holding_index → u16 bucket_id`

**Storage:**
- ~990 holdings × 2 bytes = ~2KB per flop per street
- 1,755 flops × 3 streets × 2KB = ~10MB total

**During MCCFR traversal:**
- Load precomputed mapping for current flop
- `bucket_id = mapping[holding_index]` — O(1) lookup
- Pass `bucket_id` as `hand_or_bucket` to `InfoKey::new()`

Zero equity computation overhead during training.

## Bit Layout

Bucket ID packed into the 28-bit hand field of InfoKey:

```
Bits 27-22: nut_distance_bin   (6 bits, default)
Bits 21-18: equity_delta_bin   (4 bits, default)
Bits 17-0:  spare              (18 bits, zero)
```

Configurable: `(nut_dist_bin << equity_delta_bits) | delta_bin`

Preflop remains 169 canonical hands (lossless, unchanged).

## Integration Points

| Component | Change |
|-----------|--------|
| `game/config.rs` | Add `AbstractionMode::HeuristicV3 { nut_distance_bits, equity_delta_bits }` |
| YAML deserialization | New `heuristic_v3` variant |
| `info_key.rs` | New `encode_hand_v3()` packing function |
| Per-flop pipeline | New precomputation step generating mapping files |
| MCCFR traversal | When HeuristicV3: load precomputed mapping, O(1) lookup |
| Explorer UI | Display bucket as "nut_dist=X, eq_delta=Y" |

**Unchanged:** preflop (169 hands), InfoKey bit layout structure, SPR bucket, action encoding, street encoding.

## Testing & Validation

### Correctness

- Nut distance: AA on dry flop → low bin, 72o on Kh Qh Jh → high bin
- Equity delta: flush draw on flop → positive delta, top set → near-zero delta
- River: all hands map to delta midpoint
- Determinism: same inputs always produce same bucket

### A/B Comparison

Train two blueprints on identical game tree config and iteration count:
- Model A: existing per-flop EMD clustering (200 buckets)
- Model B: HeuristicV3 (6+4 = 1,024 buckets)

Compare:
- Exploitability (if BR computation available)
- Strategy quality on diagnostic hands:
  - 45s on various boards (must not shove 99.8%)
  - AKo vs K2o on K-high boards (must have different strategies)
  - Flush draws vs air at similar equity (must differ)
- Per-flop bucket distribution histograms (check for empty/overpopulated buckets)
