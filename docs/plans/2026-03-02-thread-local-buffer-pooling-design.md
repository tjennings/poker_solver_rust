# Thread-Local Buffer Pooling for Exhaustive Flop Solver

## Problem

For SPR=6, each flop allocates 4 buffers of ~9M f64s each (277 MB total).
With 1,755 canonical flops solved via `par_iter`, every flop does fresh
`vec![0.0; 9M]` allocations. Only N_threads flops are active at once, but
each still allocates and deallocates 277 MB independently, causing allocation
churn, page faults, and memory pressure.

## Solution

Use `rayon::ThreadLocal<RefCell<FlopBuffers>>` to lazily create one buffer
set per OS thread. Each thread reuses its buffers across all flops it
processes — zeroing via `fill(0.0)` between flops instead of
allocating/deallocating.

## Design

### New struct

```rust
struct FlopBuffers {
    regret_sum: Vec<f64>,
    strategy_sum: Vec<f64>,
    delta: (Vec<f64>, Vec<f64>),
}
```

With `new(size)` (allocates) and `reset()` (fill-zeros all 4 vecs).

### Changes to `build_exhaustive`

Before the `par_iter`, create the thread-local pool:

```rust
let buffers: ThreadLocal<RefCell<FlopBuffers>> = ThreadLocal::new();
```

Inside the closure, borrow and reset:

```rust
let cell = buffers.get_or(|| RefCell::new(FlopBuffers::new(buf_size)));
let mut bufs = cell.borrow_mut();
bufs.reset();
```

Call `exhaustive_solve_one_flop` with `&mut bufs`, then extract values
while buffers are still populated. The `par_iter` closure returns
`values: Vec<f64>` directly — `FlopSolveResult` no longer needs to own
`strategy_sum`.

### Changes to `exhaustive_solve_one_flop`

- Accept `bufs: &mut FlopBuffers` parameter
- Remove internal `vec![0.0; buf_size]` allocations (lines 625-626, 647)
- Use `bufs.regret_sum`, `bufs.strategy_sum`, `bufs.delta`
- Return only metadata (delta, iterations_used) — not strategy_sum ownership

### Return value handling

Currently `FlopSolveResult` owns `strategy_sum`. With pooling, value
extraction (`exhaustive_extract_values`) happens in the `par_iter` closure
while the pooled buffer is still live, before the buffer is reused for the
next flop. `FlopSolveResult` shrinks to just convergence metadata.

## Files changed

- `crates/core/src/preflop/postflop_exhaustive.rs` only

## Memory impact

Before: 1,755 × 277 MB allocations (sequential, but each is fresh)
After: N_threads × 277 MB allocations (reused across all flops)
