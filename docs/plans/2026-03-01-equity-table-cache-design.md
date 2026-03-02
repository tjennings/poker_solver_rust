# Design: Flop Equity Table Cache

## Problem

Each flop equity table requires ~5 billion hand evaluations. With 1755 canonical flops, this precomputation dominates startup time before any CFR iteration begins. The tables depend only on flop cards — not SPR, bet sizing, or solver parameters — so they can be computed once and reused across all runs.

## Solution

### New CLI Command: `precompute-equity`

```
cargo run -p poker-solver-trainer --release -- precompute-equity [--output ./cache/equity_tables.bin]
```

- Always computes all 1755 canonical flops (deterministic order from `sample_canonical_flops(0)`)
- `indicatif` progress bar: `[=====] 423/1755 flops (eta: 2m 31s)`
- Parallelized via rayon (`into_par_iter` over flops, same as current solver)
- Default output: `./cache/equity_tables.bin`

### File Format

Bincode-serialized struct:

```rust
struct EquityTableCache {
    magic: [u8; 4],        // b"EQTC"
    version: u32,          // 1
    num_flops: u32,        // 1755
    flops: Vec<[Card; 3]>, // canonical flop order
    tables: Vec<Vec<f64>>, // 1755 × 28,561 floats each
}
```

~382 MB on disk. Plain `fs::read` + `bincode::deserialize` to load (~1-2s).

### Auto-Detection in `solve-postflop`

1. Check if `./cache/equity_tables.bin` exists
2. Deserialize, verify magic + version
3. Build `HashMap<[Card;3], usize>` mapping flop → index in cache
4. For each flop the solver needs, look up its cached table
5. Pass as `pre_equity_tables` to `build_exhaustive` (existing parameter)
6. Log hit/miss: `"Loaded equity tables for 200/200 flops from cache"`

Falls back silently to inline computation if cache is missing or invalid.

### What Changes

| Component | Change |
|-|-|
| `crates/core/src/preflop/equity_table_cache.rs` | New module: `EquityTableCache` struct, `save()`, `load()`, lookup helpers |
| `crates/trainer/src/main.rs` | New `PrecomputeEquity` subcommand with `--output` arg |
| `crates/trainer/src/main.rs` | `solve-postflop` handler: auto-detect cache, extract matching tables |
| `compute_equity_table` | Unchanged |
| `build_exhaustive` | Unchanged (already accepts `pre_equity_tables`) |

### What Stays the Same

- `compute_equity_table` function unchanged
- `build_exhaustive` already accepts `pre_equity_tables: Option<&[Vec<f64>]>`
- Multi-SPR inline precomputation still works (redundant but harmless with cache)
- MCCFR backend unaffected (doesn't use equity tables)

### Decisions

- **No mmap** — plain deserialization is fast enough (~1-2s) and avoids `unsafe`
- **Always all 1755 flops** — any subset is satisfied by the cache
- **Auto-detect at `./cache/equity_tables.bin`** — no CLI flag needed
- **Deterministic flop order** — `sample_canonical_flops(0)` is stable, stored in file for validation
