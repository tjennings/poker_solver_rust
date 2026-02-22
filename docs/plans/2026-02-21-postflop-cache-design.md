# Postflop Cache Design

## Problem

`solve-preflop` recomputes expensive postflop phases on every run, even when only preflop parameters (iterations, stack depth) change. Phases 1-6 take minutes and their outputs are deterministic for a given config.

## Current State

- Phases 2-4 (board abstraction, hand buckets, bucket equity) are cached via `abstraction_cache`
- Phase 1 (equity table) and Phases 5-6 (postflop trees + CFR solve) are always recomputed

## Design: Two New Cache Layers

### Layer 1: Equity Cache

**What**: 169x169 `EquityTable` (equities + weights matrices)

**Key**: `equity_samples: u32`

**Layout**:
```
cache/postflop/equity_<samples>/
  equity.bin      # bincode-serialized EquityTable
```

**Changes**:
- Add `#[derive(Serialize, Deserialize)]` to `EquityTable`
- New module `equity_cache.rs` in `crates/core/src/preflop/`
- Wire into `run_solve_preflop`: check cache before computing, save after

### Layer 2: Postflop Solve Cache

**What**: `PostflopValues` (output of phases 5-7: trees + CFR solve + value extraction)

**Key**: Full `PostflopModelConfig` hash + `has_equity_table` flag. This captures abstraction config, tree config, and solve config.

**Layout**:
```
cache/postflop/solve_<hex_key>/
  key.yaml        # human-readable config
  values.bin      # bincode-serialized PostflopValues
```

**Changes**:
- Add `#[derive(Serialize, Deserialize)]` to `PostflopValues`
- New module `solve_cache.rs` in `crates/core/src/preflop/`
- Refactor `PostflopAbstraction::build` to accept cached values or try cache before solving
- Wire into `run_solve_preflop`

### Cached Flow

```
run_solve_preflop:
  1. equity_cache::load(equity_samples)?        → hit: skip Phase 1
  2. abstraction_cache::load(abstraction_key)?   → hit: skip Phases 2-4 (existing)
  3. solve_cache::load(solve_key)?               → hit: skip Phases 5-7
     ↓ all hit
  4. Rebuild trees (instant) + assemble PostflopAbstraction
  5. Run preflop LCFR (Phase 8)
  6. Save bundle
```

### Cache Invalidation

Content-addressed: each key is a hash of its inputs.
- Change `equity_samples` → equity miss → solve miss (bucket equity changes)
- Change `postflop_solve_iterations` → only solve miss
- Change `num_hand_buckets_flop` → abstraction miss → solve miss
