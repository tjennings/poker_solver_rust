# Per-Flop Regret Tables Design

**Date:** 2026-03-17
**Goal:** Replace the single global regret table with per-flop regret tables so that each canonical flop has its own independent strategy, eliminating bucket ID alignment issues.

## Motivation

With shared global regret tables, per-flop bucket IDs must have consistent meaning across all flops. Independent k-means runs produce arbitrary bucket IDs — bucket 42 means different things on different flops. This causes the solver to average unrelated strategic situations, producing passive/incoherent strategies.

Per-flop regret tables solve this: each flop's regret table is independent, so bucket 42 only needs to be meaningful within that flop's context.

## Architecture

### Two-Level Storage

- **Preflop:** Global regret table (i16), always in memory. 169 lossless buckets.
- **Per-flop (flop + turn + river):** 1,755 independent regret tables (i16), one per canonical flop. Each table uses 500 per-flop buckets for flop, turn, and river decisions. Loaded/saved to disk, with a preload buffer keeping the next N flops ready in memory.

### i16 Quantized Storage

A new `CompactStorage` struct replaces `BlueprintStorage` for per-flop use:

- **Regrets:** `Vec<AtomicI16>` — 2 bytes per slot (clamped to ±32,767)
- **Strategy sums:** `Vec<AtomicI32>` — 4 bytes per slot (sufficient for per-flop accumulation)
- **Total:** 6 bytes per slot vs 12 bytes in `BlueprintStorage`

With `[1, 500, 500, 500]` bucket counts and the current action tree (~87M postflop slots):
- `BlueprintStorage`: 87M × 12 = **1.04 GB per flop**
- `CompactStorage`: 87M × 6 = **522 MB per flop**

With postflop-only allocation (preflop=0, skip preflop nodes entirely):
- `CompactStorage`: ~86M × 6 = **516 MB per flop**

16 active threads × 516 MB = **8.3 GB active memory**. Feasible.

`CompactStorage` implements the same interface as `BlueprintStorage` (`get_regret`, `add_regret`, `current_strategy`, etc.) so the MCCFR traversal code doesn't change — it just uses a trait or generic.

### Training Loop

Each "epoch" processes all 1,755 canonical flops exactly once, with each flop visited proportionally to its combinatorial weight:

```
Epoch:
  1. Build schedule: 1755 entries, each with (flop_index, flop_cards, weight)
     Shuffle order each epoch
  2. Preload buffer: background thread loads next N flops from disk
  3. AtomicUsize work_idx = 0
  4. 16 worker threads, each loops:
     a. Pop next ready flop from preload buffer (flop_index, weight, storage)
     b. Run `weight` MCCFR deals on this flop:
        - Sample hole cards + turn + river
        - Preflop decisions: use global preflop storage
        - Postflop decisions: use this flop's CompactStorage
     c. Hand dirty storage back for async disk write
     d. Go to (a)
  5. When schedule exhausted, flush remaining writes
  6. Start next epoch
```

### Preload Buffer

- Background **reader thread** walks the schedule ahead of workers, loading `CompactStorage` from disk (or creating empty) into a concurrent buffer
- Buffer capacity: N entries (e.g., 32). When full, reader blocks until workers consume
- Background **writer thread** receives dirty `CompactStorage` from workers and saves to disk asynchronously
- Workers never do disk I/O — they only grab from the ready buffer and hand back dirty storage
- Use `crossbeam::channel::bounded` for multi-consumer support (unlike `mpsc` which is single-consumer)

### Storage Layout

```
regrets/
  preflop.bin              — global, always in memory
  flop_0000.regrets        — i16/i32 CompactStorage, loaded on demand
  flop_0001.regrets
  ...
  flop_1754.regrets
```

### MCCFR Traversal Changes

The traversal function takes two storage references via a trait or generic:

- `preflop_storage` — global storage for preflop decisions
- `postflop_storage` — per-flop storage for flop/turn/river decisions

At each Decision node, select storage based on street:
```rust
let storage = if street == Street::Preflop {
    preflop_storage
} else {
    postflop_storage
};
```

### Config

```yaml
clustering:
  per_flop:
    flop_buckets: 500
    turn_buckets: 500
    river_buckets: 500

training:
  per_flop_regrets: true
  preload_buffer_size: 32
```

## What Changes

| Component | Change |
|-----------|--------|
| New `CompactStorage` | i16 regrets + i32 strategy sums, same API as BlueprintStorage |
| `mccfr.rs` traversal | Takes preflop + postflop storage; selects by street |
| `trainer.rs` | Epoch-based loop with preload buffer and atomic work-stealing |
| New `PerFlopRegretStore` | Preload buffer with reader/writer threads using crossbeam channels |
| Snapshot save/load | Save preflop.bin + flush all per-flop .regrets files |
| Config | New `per_flop_regrets`, `preload_buffer_size` |

## What Does NOT Change

- Per-flop bucket files (already computed)
- Preflop bucketing (169 lossless)
- Action tree / game tree structure
- Range solver, explorer, postflop solver
- Per-flop clustering pipeline
