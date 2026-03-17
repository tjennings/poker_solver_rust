# Per-Flop Bucketing Design

**Date:** 2026-03-17
**Goal:** Replace global clustering with per-flop bucketing (Pluribus-style) for better card abstraction quality, validated against exact range-solver solutions.

## Motivation

The current global clustering pipeline produces poor flop abstractions: 89% of flop buckets have intra-bucket equity std > 0.2, and transition consistency audit shows mean EMD of 67.8 (vs 34.8 for turn→river). The root cause is sparse histograms — building 1,000-dimensional features from ~48 turn card samples per combo.

Pluribus solves this by computing buckets **per-flop**: for each canonical flop board, independently cluster turn and river situations. This produces dense, meaningful histograms (200 bins instead of 1,000) over hands on the same board texture.

## Architecture

### Pipeline Per Flop

For each of the 1,755 canonical flops (embarrassingly parallel):

1. **River clustering:** For each (flop, turn) combination, cluster 1,326 combos by equity into 200 river buckets using 1D weighted k-means (same algorithm as current river clustering).
2. **Turn clustering:** Build per-combo histograms over the 200 per-(flop,turn) river buckets across all ~48 turns. Cluster into 200 turn buckets using EMD k-means.

After all 1,755 flops are processed:

3. **Flop clustering:** Build per-combo histograms over per-flop turn buckets. Cluster the 1,755 flops into 200 flop buckets using EMD k-means.

**Preflop:** 169 lossless buckets (unchanged).

### Shared Regret Tables

Per-flop bucketing assigns combos to bucket IDs 0-199. The MCCFR regret/strategy tables are indexed by bucket ID globally — bucket 42 on flop A and bucket 42 on flop B share the same regret table entry. This is imperfect recall, same as the current approach. The per-flop clustering just produces *better assignments* into those 200 shared buckets.

The abstract game size stays the same.

## Storage Format

### Per-Flop Files

One packed file per canonical flop: `flop_0000.buckets` through `flop_1754.buckets`.

```
Header:
  flop_cards: [Card; 3]         — canonical flop
  turn_bucket_count: u16        — 200
  river_bucket_count: u16       — 200
  turn_count: u8                — number of turn cards (~48)

Turn section:
  turn_buckets: [u16; turn_count * 1326]   — per-combo turn bucket assignments

River sections (one per turn card):
  turn_card: Card
  river_buckets: [u16; river_count * 1326] — per-combo river bucket assignments
```

**Size estimates:**
- Per flop: ~5.7MB (127KB turn + 5.6MB river)
- All flops: 1,755 × 5.7MB = ~10GB total
- Plus global `flop.buckets` and `preflop.buckets`

### MCCFR Lookup

Given a flop, load (or cache) that flop's file. Look up turn bucket by turn card index + combo index. Look up river bucket by turn card + river card index + combo index.

## Config

```yaml
clustering:
  flop:
    buckets: 200
  per_flop:
    turn_buckets: 200
    river_buckets: 200
  preflop:
    buckets: 169
```

Bucket counts are configurable but default to 200 (matching Pluribus). The `algorithm` field and `sample_boards` go away — per-flop clustering enumerates exhaustively within each flop.

## Validation Framework

### Method

A fixed set of ~20 canonical spots solved exactly with the range solver, then compared against the blueprint trained on per-flop buckets.

### Spot Selection

- Small trees (1-2 actions deep, single street or street transition)
- Key scenarios: open raise, 3-bet, c-bet, check-raise, turn barrel
- Mix of board textures: dry, wet, paired, monotone
- Mix of SPR situations
- Defined in a YAML/JSON file

### Metrics

1. **Strategy distance** — L2 distance between blueprint and exact action probabilities, averaged across all combos. Per-spot and aggregate.
2. **EV difference** — EV under blueprint strategy minus EV under exact solution, averaged across combos. Measures value lost to abstraction error. Reported in mbb.

### Output

Comparison report: old (global buckets) vs new (per-flop buckets) on the same spot set.

```
Spot: SB open → BB 3-bet → SB facing 3-bet, board [Ks 7d 2c]
  Strategy L2:  global=0.182  per-flop=0.094  (48% improvement)
  EV loss (mbb): global=12.3   per-flop=5.1    (59% improvement)
```

### CLI

New subcommand: `validate-blueprint` that loads a blueprint + spot definitions, runs range-solver on each spot, computes both metrics, prints the report.

## What Changes

| Component | Change |
|-----------|--------|
| `cluster_pipeline.rs` | Replace global pipeline with per-flop pipeline |
| `bucket_file.rs` | New per-flop file format (read/write) |
| `BucketFile` | New lookup methods for per-flop turn/river buckets |
| `mccfr.rs` | Load per-flop bucket files, lookup by flop→turn→river |
| `config.rs` | New `per_flop` config section, remove `algorithm` field |
| `cluster_diagnostics.rs` | Update audit to work with per-flop files |
| `trainer/main.rs` | New `validate-blueprint` subcommand |
| Config YAML | New format (see above) |

## What Does NOT Change

- Preflop bucketing (169 lossless)
- MCCFR algorithm (same traversal, same regret updates)
- Abstract game tree size (200 buckets × action sequences)
- Range solver
- Explorer UI
- Postflop solver
