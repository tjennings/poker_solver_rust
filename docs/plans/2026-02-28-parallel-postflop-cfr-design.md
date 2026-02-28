# Parallel Within-Flop Postflop CFR

## Problem

The exhaustive postflop CFR solver is single-threaded within each flop solve. The inner loop traverses 169×169×2 = 57,122 hand-pair/position combinations per iteration, serially. At high SPR values the game tree has thousands of nodes, making each traversal expensive and the full solve painfully slow.

The outer loop parallelizes across flops via rayon, but when solving a single flop (diagnostics, testing) or when per-flop work dominates at high SPR, there is zero parallelism.

## Solution: Shared Snapshot + Delta-Merge Pattern

Port the preflop solver's `parallel_traverse()` pattern to the postflop solver, extracting the shared logic into a trait in `crates/core/src/cfr/parallel.rs`.

### Trait: `ParallelCfr`

```rust
/// Context for one CFR iteration. Implementors provide a frozen-snapshot
/// traversal function that writes regret/strategy deltas to thread-local buffers.
pub trait ParallelCfr: Sync {
    /// Size of the regret/strategy flat buffers.
    fn buffer_size(&self) -> usize;

    /// Traverse one hand pair (both hero positions), accumulating
    /// regret deltas into `dr` and strategy deltas into `ds`.
    fn traverse_pair(&self, dr: &mut [f64], ds: &mut [f64], h1: u16, h2: u16);
}
```

### Generic parallel driver

```rust
/// Parallel fold+reduce over hand pairs. Returns merged (regret_delta, strategy_delta).
pub fn parallel_traverse<T: ParallelCfr>(
    ctx: &T,
    pairs: &[(u16, u16)],
) -> (Vec<f64>, Vec<f64>) {
    let buf_size = ctx.buffer_size();
    pairs.par_iter()
        .fold(
            || (vec![0.0f64; buf_size], vec![0.0f64; buf_size]),
            |(mut dr, mut ds), &(h1, h2)| {
                ctx.traverse_pair(&mut dr, &mut ds, h1, h2);
                (dr, ds)
            },
        )
        .reduce(
            || (vec![0.0; buf_size], vec![0.0; buf_size]),
            |(mut ar, mut a_s), (br, bs)| {
                add_into(&mut ar, &br);
                add_into(&mut a_s, &bs);
                (ar, a_s)
            },
        )
}

/// Element-wise `dst[i] += src[i]`.
pub fn add_into(dst: &mut [f64], src: &[f64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d += s;
    }
}
```

### Preflop implementation

The existing `Ctx` struct implements the trait, replacing the standalone `parallel_traverse()` function in `solver.rs`:

```rust
impl ParallelCfr for Ctx<'_> {
    fn buffer_size(&self) -> usize { self.layout.total_size }
    fn traverse_pair(&self, dr: &mut [f64], ds: &mut [f64], h1: u16, h2: u16) {
        let w = self.equity.weight(h1 as usize, h2 as usize);
        for hero_pos in 0..2u8 {
            let (hh, oh) = if hero_pos == 0 { (h1, h2) } else { (h2, h1) };
            cfr_traverse(self, dr, ds, 0, hh, oh, hero_pos, 1.0, w);
        }
    }
}
```

`train_one_iteration()` changes from `let (mr, ms) = parallel_traverse(&ctx, &self.pairs)` to `let (mr, ms) = cfr::parallel::parallel_traverse(&ctx, &self.pairs)`.

### Postflop implementation

New `PostflopCfrCtx` struct holding frozen snapshot + shared references:

```rust
struct PostflopCfrCtx<'a> {
    tree: &'a PostflopTree,
    layout: &'a PostflopLayout,
    equity_table: &'a [f64],
    snapshot: &'a [f64],   // frozen regret_sum clone
    iteration: u64,
    dcfr: &'a DcfrParams,
}

impl ParallelCfr for PostflopCfrCtx<'_> {
    fn buffer_size(&self) -> usize { self.layout.total_size }
    fn traverse_pair(&self, dr: &mut [f64], ds: &mut [f64], h1: u16, h2: u16) {
        let eq = self.equity_table[h1 as usize * NUM_CANONICAL_HANDS + h2 as usize];
        if eq.is_nan() { return; }  // no valid combos
        for hero_pos in 0..2u8 {
            exhaustive_cfr_traverse(
                self.tree, self.layout, self.equity_table,
                self.snapshot, dr, ds,
                0, h1, h2, hero_pos, 1.0, 1.0,
                self.iteration, self.dcfr,
            );
        }
    }
}
```

### Refactor `exhaustive_cfr_traverse` signature

Split the current single `regret_sum` parameter (used for both reading strategy and writing deltas) into two:

**Before:**
```rust
fn exhaustive_cfr_traverse(
    tree, layout, equity_table,
    regret_sum: &mut [f64],    // read strategy + write regret deltas
    strategy_sum: &mut [f64],  // write strategy deltas
    ...
)
```

**After:**
```rust
fn exhaustive_cfr_traverse(
    tree, layout, equity_table,
    snapshot: &[f64],          // read-only: frozen regrets for strategy computation
    dr: &mut [f64],            // write: regret deltas
    ds: &mut [f64],            // write: strategy deltas
    ...
)
```

Inside the function:
- `regret_matching_into(snapshot, start, ...)` — reads from frozen snapshot
- `dr[start + i] += weight * reach_opp * (val - node_value)` — writes to delta buffer
- `ds[start + i] += weight * reach_hero * s` — writes to delta buffer

### Refactor `exhaustive_solve_one_flop` iteration loop

**Before (serial):**
```rust
for iter in 0..num_iterations {
    for hero_hand in 0..169 {
        for opp_hand in 0..169 {
            exhaustive_cfr_traverse(..., &mut regret_sum, &mut strategy_sum, ...);
        }
    }
    // DCFR discounting
}
```

**After (parallel):**
```rust
let pairs: Vec<(u16, u16)> = ...; // pre-filter valid pairs (non-NaN equity)
let mut snapshot = vec![0.0f64; buf_size];

for iter in 0..num_iterations {
    snapshot.clone_from(&regret_sum);
    let ctx = PostflopCfrCtx { tree, layout, equity_table, snapshot: &snapshot, iteration: iter as u64, dcfr };
    let (dr, ds) = cfr::parallel::parallel_traverse(&ctx, &pairs);
    // DCFR discount existing sums, then merge deltas
    if dcfr.should_discount(iter as u64) {
        dcfr.discount_regrets(&mut regret_sum, iter as u64);
        dcfr.discount_strategy_sums(&mut strategy_sum, iter as u64);
    }
    add_into(&mut regret_sum, &dr);
    add_into(&mut strategy_sum, &ds);
    if dcfr.should_floor_regrets() {
        dcfr.floor_regrets(&mut regret_sum);
    }
    // exploitability check...
}
```

### Parallelize `compute_exploitability`

The best-response computation iterates over 169×169 hand pairs independently, summing EV values. Use `par_iter` with a sum reduction:

```rust
fn compute_exploitability(tree, layout, strategy_sum, equity_table) -> f64 {
    for br_player in 0..2 {
        let (total, count) = (0..169u16)
            .into_par_iter()
            .flat_map(|h| (0..169u16).map(move |o| (h, o)))
            .filter(|&(h, o)| !equity_table[h*169+o].is_nan())
            .map(|(h, o)| best_response_ev(tree, layout, strategy_sum, equity_table, 0, h, o, br_player))
            .fold(|| (0.0, 0u64), |(t, c), v| (t + v, c + 1))
            .reduce(|| (0.0, 0), |(t1, c1), (t2, c2)| (t1 + t2, c1 + c2));
        // ...
    }
}
```

### Nested rayon

When `build_exhaustive()` runs `par_iter` over flops and each flop also runs `par_iter` over hand pairs, rayon handles nesting via work-stealing. No custom thread pool configuration needed initially — rayon degrades gracefully (inner tasks join the global pool, no thread explosion).

## Files Changed

| File | Change |
|-|-|
| `crates/core/src/cfr/parallel.rs` | **New.** `ParallelCfr` trait, `parallel_traverse()`, `add_into()` |
| `crates/core/src/cfr/mod.rs` | Add `pub mod parallel;` |
| `crates/core/src/preflop/solver.rs` | Implement `ParallelCfr` for `Ctx`, remove standalone `parallel_traverse()` and `add_into()`, update `train_one_iteration()` |
| `crates/core/src/preflop/postflop_exhaustive.rs` | Add `PostflopCfrCtx`, split `exhaustive_cfr_traverse` signature (snapshot vs delta), rewrite `exhaustive_solve_one_flop` iteration loop, parallelize `compute_exploitability` |

## Convergence

All convergence guarantees are preserved. The snapshot ensures all threads read the same strategy within an iteration. Merged deltas are mathematically identical to sequential execution (addition is commutative). DCFR discounting remains in the sequential part of the loop, applied once per iteration after the parallel reduce.

## Memory

Per-flop overhead: one snapshot clone (~32 MB at SPR=20) + one delta buffer per rayon worker thread (~32 MB × num_workers). At 8 threads solving one flop: ~288 MB total. Acceptable for the performance gain.
