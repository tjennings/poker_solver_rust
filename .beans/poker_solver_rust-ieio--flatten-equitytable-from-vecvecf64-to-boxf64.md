---
# poker_solver_rust-ieio
title: Flatten EquityTable from Vec<Vec<f64>> to Box<[f64]>
status: todo
type: task
priority: normal
created_at: 2026-02-28T23:58:10Z
updated_at: 2026-02-28T23:58:10Z
---

## Problem
`EquityTable` in `preflop/equity.rs:22-26` stores equities and weights as `Vec<Vec<f64>>` — 169 heap-allocated rows. Every `equity(i,j)` and `weight(i,j)` call double-dereferences (outer Vec ptr → row ptr → element). Millions of double-dereferences per iteration.

## Fix
Replace with flat `Vec<f64>` of size 169×169 (or `Box<[f64; 28561]>`). Index as `equities[i * 169 + j]`. Update construction, serialization, and all accessors.

## Verification
- `cargo test -p poker-solver-core`
- `cargo clippy`
- Check serialization/deserialization still works

## TODO
- [ ] Change EquityTable fields to flat Vec<f64> or Box<[f64]>
- [ ] Update equity() and weight() accessors to use flat indexing
- [ ] Update construction code (compute_equity_table, etc.)
- [ ] Update any serialization/deserialization
- [ ] Run tests and clippy
