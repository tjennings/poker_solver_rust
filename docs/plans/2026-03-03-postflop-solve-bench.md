# Postflop Solve Benchmark Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a Criterion micro-benchmark for the exhaustive postflop CFR solver's inner loop (`exhaustive_solve_one_flop`) so it can be profiled with `samply`.

**Architecture:** Promote a few `pub(crate)` items to `pub` so the benchmark (which compiles as an external crate) can access the solver internals. Create a benchmark that sets up a realistic single-flop solve (AhKsQd, SPR=3) and measures 1 CFR iteration. The equity table setup uses cached data from `./cache/` when available (equity_tables.bin → extract, or rank_arrays.bin → derive), falling back to full computation only if no cache exists.

**Tech Stack:** Criterion 0.5 (already a dev-dependency), `rs_poker` for card types.

---

### Task 1: Promote internal visibility for benchmarking

**Why:** Benchmarks in `benches/` compile as external crates. `exhaustive_solve_one_flop`, `FlopBuffers`, `PostflopLayout`, and `annotate_streets` are currently `pub(crate)` or private. We need them `pub`.

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:50` — `FlopBuffers` struct
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:58` — `FlopBuffers::new`
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:805` — `exhaustive_solve_one_flop`
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:94` — `PostflopLayout` struct
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:110` — `PostflopLayout::build`
- Modify: `crates/core/src/preflop/postflop_abstraction.rs:200` — `annotate_streets`

**Step 1: Change visibility**

In `postflop_exhaustive.rs`:
```rust
// line 50: pub(crate) struct FlopBuffers → pub struct FlopBuffers
// line 58: pub(crate) fn new → pub fn new
// line 805: fn exhaustive_solve_one_flop → pub fn exhaustive_solve_one_flop
```

In `postflop_abstraction.rs`:
```rust
// line 94: pub(crate) struct PostflopLayout → pub struct PostflopLayout
// line 110: pub(crate) fn build → pub fn build
// line 200: pub(crate) fn annotate_streets → pub fn annotate_streets
```

**Step 2: Verify it compiles**

Run: `cargo check -p poker-solver-core`
Expected: success with no new errors

**Step 3: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs crates/core/src/preflop/postflop_abstraction.rs
git commit -m "refactor: promote solver internals to pub for benchmarking"
```

---

### Task 2: Create the benchmark file

**Files:**
- Create: `crates/core/benches/postflop_solve_bench.rs`
- Modify: `crates/core/Cargo.toml` — add `[[bench]]` entry

**Step 1: Add bench entry to Cargo.toml**

Append after the existing `[[bench]]` blocks (after line 40):

```toml
[[bench]]
name = "postflop_solve_bench"
harness = false
```

**Step 2: Create the benchmark**

Create `crates/core/benches/postflop_solve_bench.rs`:

```rust
use std::path::Path;

use criterion::{criterion_group, criterion_main, Criterion};

use poker_solver_core::cfr::dcfr::DcfrParams;
use poker_solver_core::poker::{Card, Suit, Value};
use poker_solver_core::preflop::equity_table_cache::EquityTableCache;
use poker_solver_core::preflop::postflop_abstraction::{annotate_streets, PostflopLayout};
use poker_solver_core::preflop::postflop_exhaustive::{
    compute_equity_table, exhaustive_solve_one_flop, FlopBuffers,
};
use poker_solver_core::preflop::postflop_hands::{build_combo_map, NUM_CANONICAL_HANDS};
use poker_solver_core::preflop::rank_array_cache::{derive_equity_table, RankArrayCache};
use poker_solver_core::preflop::{PostflopModelConfig, PostflopTree};

/// AhKsQd — same flop as test.yaml
fn bench_flop() -> [Card; 3] {
    [
        Card::new(Value::Ace, Suit::Heart),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Diamond),
    ]
}

/// Load equity table from cache if available, otherwise compute from scratch.
///
/// Priority:
/// 1. `cache/equity_tables.bin` — extract the AhKsQd table directly
/// 2. `cache/rank_arrays.bin`  — derive via integer comparison (no hand eval)
/// 3. Full computation          — evaluate every (turn, river) x combo pair
fn load_or_compute_equity(flop: [Card; 3], combo_map: &[Vec<(Card, Card)>]) -> Vec<f64> {
    let eq_path = Path::new("cache/equity_tables.bin");
    if let Some(cache) = EquityTableCache::load(eq_path) {
        if let Some(mut tables) = cache.extract_tables(&[flop]) {
            eprintln!("[bench] equity table loaded from cache/equity_tables.bin");
            return tables.remove(0);
        }
    }

    let rank_path = Path::new("cache/rank_arrays.bin");
    if let Some(cache) = RankArrayCache::load(rank_path) {
        if let Some(data) = cache.get_flop_data(&flop) {
            eprintln!("[bench] equity table derived from cache/rank_arrays.bin");
            return derive_equity_table(data, combo_map);
        }
    }

    eprintln!("[bench] no cache found — computing equity table from scratch");
    compute_equity_table(combo_map, flop)
}

fn bench_postflop_solve(c: &mut Criterion) {
    // Setup: build tree, layout, equity table (done once, outside the benchmark loop)
    let config = PostflopModelConfig {
        bet_sizes: vec![0.3, 0.5],
        max_raises_per_street: 2,
        postflop_solve_iterations: 1,
        ..PostflopModelConfig::exhaustive_fast()
    };
    let tree = PostflopTree::build_with_spr(&config, 3.0).unwrap();
    let node_streets = annotate_streets(&tree);
    let n = NUM_CANONICAL_HANDS;
    let layout = PostflopLayout::build(&tree, &node_streets, n, n, n);

    let flop = bench_flop();
    let combo_map = build_combo_map(&flop);
    let equity_table = load_or_compute_equity(flop, &combo_map);

    let dcfr = DcfrParams::linear();

    let mut group = c.benchmark_group("postflop_solve");
    group.sample_size(10);

    group.bench_function("solve_1_iter", |b| {
        b.iter_batched(
            || FlopBuffers::new(layout.total_size),
            |mut bufs| {
                exhaustive_solve_one_flop(
                    &tree,
                    &layout,
                    &equity_table,
                    1,     // num_iterations
                    0.0,   // convergence_threshold (no early stop)
                    "AhKsQd",
                    &dcfr,
                    &config,
                    &|_| {},
                    None,
                    &mut bufs,
                )
            },
            criterion::BatchSize::LargeInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_postflop_solve);
criterion_main!(benches);
```

Key decisions:
- **Cache-first equity loading:** Tries `cache/equity_tables.bin` first (instant extract), then `cache/rank_arrays.bin` (derive via integer comparison, ~seconds), then falls back to full computation (~minutes). The `eprintln!` messages show which path was taken.
- `iter_batched` with `LargeInput` — allocates fresh `FlopBuffers` per iteration (realistic: in production each thread gets its own). The buffer allocation is in setup, not measured.
- `sample_size(10)` — each iteration is expensive; 10 samples is enough for stable measurement.
- Tree/layout/equity built once outside the loop — we're benchmarking the solve, not the setup.

**Step 3: Verify it compiles**

Run: `cargo check -p poker-solver-core --bench postflop_solve_bench`
Expected: success

**Step 4: Run the benchmark**

Run: `cargo bench -p poker-solver-core -- postflop_solve`
Expected: Criterion output showing `solve_1_iter` timing

**Step 5: Commit**

```bash
git add crates/core/benches/postflop_solve_bench.rs crates/core/Cargo.toml
git commit -m "bench: add Criterion benchmark for exhaustive postflop solver"
```

---

### Task 3: Verify profiling workflow with samply

**Not code — just verification.**

**Step 1: Build the benchmark binary**

Run: `cargo build --release -p poker-solver-core --bench postflop_solve_bench`
Expected: binary at `target/release/deps/postflop_solve_bench-*`

**Step 2: Profile with samply**

Run: `samply record target/release/deps/postflop_solve_bench-* --bench solve_1_iter`
Expected: samply opens Firefox Profiler with the profile. Check that the flamegraph shows `exhaustive_solve_one_flop` and its callees (`parallel_traverse_pooled`, `traverse_pair`, `regret_matching_into`, etc.).

**Step 3: Save profile to disk**

samply saves profiles as JSON to `~/.local/share/samply/` (or equivalent on macOS). Verify the profile file exists and note its path.

---

## Profiling cheat sheet

```bash
# Run benchmark (Criterion statistics)
cargo bench -p poker-solver-core -- postflop_solve

# Profile with samply (CPU hot spots)
cargo build --release -p poker-solver-core --bench postflop_solve_bench
samply record target/release/deps/postflop_solve_bench-* --bench solve_1_iter

# Save samply profile JSON for later analysis
# Profiles are auto-saved; check samply output for the path
```
