# Per-Flop Regret Tables Design

**Date:** 2026-03-17
**Goal:** Replace the single global regret table with per-flop regret tables so that each canonical flop has its own independent strategy, eliminating bucket ID alignment issues.

## Motivation

With shared global regret tables, per-flop bucket IDs must have consistent meaning across all flops. Independent k-means runs produce arbitrary bucket IDs — bucket 42 means different things on different flops. This causes the solver to average unrelated strategic situations, producing passive/incoherent strategies.

Per-flop regret tables solve this: each flop's regret table is independent, so bucket 42 only needs to be meaningful within that flop's context.

## Architecture

### Two-Level Storage

- **Preflop:** Global regret table (i16), always in memory. 169 lossless buckets.
- **Per-flop (flop + turn + river):** 1,755 independent regret tables (i16), one per canonical flop. Each table uses 500 per-flop buckets for flop, turn, and river decisions. Managed by a FIFO disk-backed cache.

### Training Loop

Each "epoch" processes all 1,755 canonical flops exactly once, with each flop visited proportionally to its combinatorial weight:

```
Epoch:
  1. Build schedule: all 1755 flops, each repeated `weight` times, shuffled
     Total deals per epoch = C(52,3) = 22,100
  2. AtomicUsize counter = 0
  3. 16 threads, each loops:
     a. idx = counter.fetch_add(1)
     b. if idx >= schedule.len(): epoch done
     c. Load flop's regret table from FIFO cache (or disk)
     d. Sample random hole cards for both players
     e. MCCFR traversal:
        - Preflop: use global regret table
        - Flop/Turn/River: use this flop's regret table
     f. Mark flop's regret table dirty in cache
     g. Go to (a)
  4. Flush all dirty tables to disk
  5. Reshuffle schedule, reset counter, start next epoch
```

### FIFO Cache

- Circular buffer of `flop_cache_capacity` entries (default 1000)
- Each entry: flop index + `Vec<i16>` regret array + dirty flag
- When full, flush the oldest entry to disk before loading a new one
- No random access needed — flops are processed in schedule order and not revisited until next epoch
- On epoch end, flush all remaining dirty entries

### Storage Layout

```
regrets/
  preflop.bin          — i16, always in memory
  flop_0000.bin        — i16, FIFO cached
  flop_0001.bin        — i16, FIFO cached
  ...
  flop_1754.bin        — i16, FIFO cached
```

### MCCFR Traversal Changes

The traversal function currently takes a single `BlueprintStorage`. It changes to take:

- `PreflopStorage` — global i16 regret/strategy arrays for preflop decisions
- `PerFlopStorage` — i16 regret/strategy arrays for this flop's postflop decisions

At the flop chance node, the traversal switches from preflop storage to per-flop storage. The rest of the traversal (flop, turn, river decision nodes) uses the per-flop storage exclusively.

### Per-Flop Regret Table Sizing

Each per-flop table covers all postflop decision nodes in the action tree:

- Flop decisions: 500 buckets × actions at each flop decision node
- Turn decisions: 500 buckets × actions at each turn decision node
- River decisions: 500 buckets × actions at each river decision node

The action tree structure is the same for every flop. Only the bucket assignments differ.

### Quantization

All regrets use i16 (−32,768 to +32,767). This provides sufficient dynamic range for MCCFR regret accumulation. Regrets that would exceed the range are clamped.

### Config

```yaml
clustering:
  per_flop:
    flop_buckets: 500
    turn_buckets: 500
    river_buckets: 500

training:
  per_flop_regrets: true
  regret_quantization: i16
  flop_cache_capacity: 1000
```

## What Changes

| Component | Change |
|-----------|--------|
| `BlueprintStorage` | Split into `PreflopStorage` (global, i16) + `PerFlopStorage` (per-flop, i16) |
| `mccfr.rs` traversal | Takes both storages; switches at flop chance node |
| `trainer.rs` | Epoch-based loop with atomic counter, FIFO cache |
| Snapshot save/load | Save preflop.bin + 1,755 flop_NNNN.bin files |
| Config | New `per_flop_regrets`, `flop_cache_capacity` |

## What Does NOT Change

- Per-flop bucket files (already computed)
- Preflop bucketing (169 lossless)
- Action tree / game tree structure
- Range solver, explorer, postflop solver
- Per-flop clustering pipeline
