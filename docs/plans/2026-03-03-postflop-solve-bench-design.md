# Postflop Solve Benchmark Design

## Goal

Criterion micro-benchmark for the exhaustive postflop CFR solver hot path, enabling CPU hot-spot profiling via `samply`.

## What

A single benchmark `solve_1_iter` in `crates/core/benches/postflop_solve_bench.rs` that:

1. Builds a `PostflopTree` and `PostflopLayout` for SPR=3, bet sizes [0.3, 0.5], max 2 raises per street
2. Computes an equity table for flop AhKsQd
3. Allocates `FlopBuffers`
4. Calls `exhaustive_solve_one_flop` with 1 CFR iteration
5. Measures wall time via Criterion (small sample size since each call is expensive)

## Setup details

- Flop: AhKsQd (same as test.yaml)
- SPR: 3, bet sizes: [0.3, 0.5], max_raises: 2
- DCFR params: linear (default)
- Config: `PostflopModelConfig::exhaustive_fast()` or similar minimal config
- No convergence threshold (1 iteration, no early stop)

## Profiling workflow

```bash
# Run benchmark
cargo bench -p poker-solver-core -- postflop_solve

# Profile with samply
cargo build --release -p poker-solver-core --bench postflop_solve_bench
samply record target/release/deps/postflop_solve_bench-* --bench solve_1_iter
```

## File

`crates/core/benches/postflop_solve_bench.rs`
