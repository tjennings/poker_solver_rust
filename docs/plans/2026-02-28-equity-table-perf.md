# Equity Table Performance Optimization

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Optimize `compute_equity_table` which consumes ~98% of postflop solve runtime (flamegraph-confirmed).

**Architecture:** Three incremental optimizations to the inner loop of `compute_equity_table` in `postflop_exhaustive.rs`, plus caching equity tables across SPR solves in the trainer. Each optimization is independently valuable and testable. A criterion benchmark gates correctness and measures impact.

**Tech Stack:** Rust, rayon, criterion, rs_poker

## Agent Team & Execution Order

| Agent | Task | Parallel? |
|-|-|-|
| `rust-developer` | Tasks 1-5 (sequential, each builds on previous) | No |
| `rust-perf-reviewer` | Review after Task 3 and after Task 5 | After milestones |
| `idiomatic-rust-enforcer` | Final review | After Task 5 |

---

### Task 1: Add criterion benchmark for `compute_equity_table`

**Files:**
- Create: `crates/core/benches/equity_table_bench.rs`
- Modify: `crates/core/Cargo.toml:33-35` (add bench entry)

**Step 1: Add bench entry to Cargo.toml**

Append after line 35 of `crates/core/Cargo.toml`:

```toml
[[bench]]
name = "equity_table_bench"
harness = false
```

**Step 2: Write the benchmark**

```rust
//! Benchmark for compute_equity_table to track optimization impact.

use criterion::{criterion_group, criterion_main, Criterion};

use poker_solver_core::poker::{Card, Suit, Value};
use poker_solver_core::preflop::postflop_hands::build_combo_map;
use poker_solver_core::preflop::postflop_exhaustive::compute_equity_table;

fn flop_akq() -> [Card; 3] {
    [
        Card::new(Value::Ace, Suit::Heart),
        Card::new(Value::King, Suit::Spade),
        Card::new(Value::Queen, Suit::Diamond),
    ]
}

fn bench_equity_table(c: &mut Criterion) {
    let mut group = c.benchmark_group("equity_table");
    group.sample_size(10); // expensive function — 10 samples is plenty

    let flop = flop_akq();
    let combo_map = build_combo_map(&flop);

    group.bench_function("AKQr", |b| {
        b.iter(|| compute_equity_table(&combo_map, flop))
    });

    group.finish();
}

criterion_group!(benches, bench_equity_table);
criterion_main!(benches);
```

**Step 3: Make `compute_equity_table` public for benchmarking**

In `crates/core/src/preflop/postflop_exhaustive.rs:39`, change:
```rust
fn compute_equity_table(
```
to:
```rust
pub fn compute_equity_table(
```

Also ensure `compute_equity_table` is re-exported. In `crates/core/src/preflop/mod.rs`, find the `pub(crate) use postflop_exhaustive::` line and add `compute_equity_table` to the public exports, or add:
```rust
pub use postflop_exhaustive::compute_equity_table;
```

(Check actual module structure — may need `pub mod postflop_exhaustive` or a `pub use` in the preflop module.)

**Step 4: Run the benchmark to get baseline**

```bash
cargo bench -p poker-solver-core -- equity_table
```

Record the baseline time. Expected: ~15-25 seconds per iteration.

**Step 5: Commit**

```bash
git add crates/core/benches/equity_table_bench.rs crates/core/Cargo.toml crates/core/src/preflop/postflop_exhaustive.rs crates/core/src/preflop/mod.rs
git commit -m "bench: add criterion benchmark for compute_equity_table"
```

---

### Task 2: Replace `used.contains()` with a 64-bit bitmask

The flamegraph shows `slice_contains` at 3.5% of runtime. The inner loop calls `used.contains(&turn)` and `used.contains(&river)` — a linear scan of 7 elements per runout. Replace with a `u64` bitmask where bit `i` = card index `i` is used.

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:39-107`

**Step 1: Write a test for bitmask correctness**

Add to the `#[cfg(test)] mod tests` block in `postflop_exhaustive.rs`:

```rust
#[timed_test]
fn bitmask_used_cards_matches_contains() {
    use crate::preflop::postflop_hands::{all_cards_vec, build_combo_map};
    use crate::poker::{Card, Suit, Value};

    let flop = [
        Card::new(Value::Two, Suit::Spade),
        Card::new(Value::Seven, Suit::Heart),
        Card::new(Value::Queen, Suit::Diamond),
    ];
    let deck = all_cards_vec();

    // Build bitmask for flop
    let mut mask: u64 = 0;
    for &c in &flop {
        let idx = deck.iter().position(|&d| d == c).unwrap();
        mask |= 1u64 << idx;
    }

    // Verify bitmask matches contains for every card
    for (i, &card) in deck.iter().enumerate() {
        let in_mask = (mask >> i) & 1 == 1;
        let in_slice = flop.contains(&card);
        assert_eq!(in_mask, in_slice, "mismatch for card {card} at index {i}");
    }
}
```

**Step 2: Run test to verify it passes**

```bash
cargo test -p poker-solver-core -- bitmask_used_cards_matches_contains
```

**Step 3: Rewrite `compute_equity_table` inner loop to use bitmask**

Replace the function body (lines 39-107) with:

```rust
fn compute_equity_table(combo_map: &[Vec<(Card, Card)>], flop: [Card; 3]) -> Vec<f64> {
    let n = NUM_CANONICAL_HANDS;
    let deck = all_cards_vec();

    // Pre-compute card-to-index lookup for bitmask construction.
    // deck is always the same 52-card ordering, so index = position in deck.
    let card_index = |c: Card| -> u32 {
        // Cards are ordered by value then suit in all_cards_vec().
        // Use linear scan (52 elements, called outside hot loop).
        deck.iter().position(|&d| d == c).unwrap() as u32
    };

    // Flop bitmask — set once, reused for all hand pairs.
    let flop_mask: u64 = (1u64 << card_index(flop[0]))
        | (1u64 << card_index(flop[1]))
        | (1u64 << card_index(flop[2]));

    // Parallel over hero hands
    let rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|hero_idx| {
            let mut row = vec![f64::NAN; n];
            let hero_combos = &combo_map[hero_idx];
            if hero_combos.is_empty() {
                return row;
            }

            for opp_idx in 0..n {
                let opp_combos = &combo_map[opp_idx];
                if opp_combos.is_empty() {
                    continue;
                }

                let mut total_eq = 0.0f64;
                let mut total_count = 0u64;

                for &(h1, h2) in hero_combos {
                    for &(o1, o2) in opp_combos {
                        if h1 == o1 || h1 == o2 || h2 == o1 || h2 == o2 {
                            continue;
                        }

                        // Bitmask of all 7 used cards.
                        let used: u64 = flop_mask
                            | (1u64 << card_index(h1))
                            | (1u64 << card_index(h2))
                            | (1u64 << card_index(o1))
                            | (1u64 << card_index(o2));

                        for (ti, &turn) in deck.iter().enumerate() {
                            if used & (1u64 << ti) != 0 {
                                continue;
                            }
                            for (ri, &river) in deck[ti + 1..].iter().enumerate() {
                                let river_idx = ti + 1 + ri;
                                if used & (1u64 << river_idx) != 0 {
                                    continue;
                                }
                                let board = [flop[0], flop[1], flop[2], turn, river];
                                let hero_rank = rank_hand([h1, h2], &board);
                                let opp_rank = rank_hand([o1, o2], &board);
                                total_eq += match hero_rank.cmp(&opp_rank) {
                                    std::cmp::Ordering::Greater => 1.0,
                                    std::cmp::Ordering::Equal => 0.5,
                                    std::cmp::Ordering::Less => 0.0,
                                };
                                total_count += 1;
                            }
                        }
                    }
                }

                if total_count > 0 {
                    row[opp_idx] = total_eq / total_count as f64;
                }
            }
            row
        })
        .collect();

    // Flatten
    let mut table = vec![f64::NAN; n * n];
    for (hero_idx, row) in rows.into_iter().enumerate() {
        table[hero_idx * n..hero_idx * n + n].copy_from_slice(&row);
    }
    table
}
```

**Key change:** `used.contains(&turn)` (linear scan of 7-element array) → `used & (1u64 << ti) != 0` (single bitwise op). The `card_index` closure is called per combo-pair (outside the runout loop), not per runout.

**Step 4: Run existing equity table tests**

```bash
cargo test -p poker-solver-core -- equity_table
cargo test -p poker-solver-core -- bitmask_used_cards
```

All should pass. The ignored slow tests (`equity_table_has_correct_size`, `equity_table_symmetric`, etc.) can be run to verify numerical correctness:

```bash
cargo test -p poker-solver-core -- equity_table --ignored
```

**Step 5: Run benchmark to measure improvement**

```bash
cargo bench -p poker-solver-core -- equity_table
```

Expected: ~5-15% improvement from eliminating `slice_contains`.

**Step 6: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "perf: replace slice_contains with bitmask in equity table inner loop"
```

---

### Task 3: Pre-compute card index array to eliminate `From<u8>` conversions

The flamegraph shows 8.3% of time in `From<u8>::from` — card type conversions happening in the hot loop. Pre-compute a `deck_indices: [u32; 52]` array mapping card position to its index, and use raw indices throughout the inner loop instead of `Card` values.

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs:39-107`

**Step 1: Refactor `compute_equity_table` to pre-compute deck indices**

The key insight: the `card_index` closure from Task 2 still does a linear scan per combo-pair. Instead, build a `HashMap<Card, u32>` once before the parallel section:

Replace the `card_index` closure and `flop_mask` setup with:

```rust
    // Pre-compute card → bit-index map (O(52), done once).
    let mut card_to_bit: [u32; 64] = [0; 64]; // indexed by Card's internal representation
    for (i, &c) in deck.iter().enumerate() {
        // Use a unique key from the Card. rs_poker Card can be converted to u8.
        card_to_bit[u8::from(c) as usize] = i as u32;
    }

    let bit_of = |c: Card| -> u32 { card_to_bit[u8::from(c) as usize] };

    let flop_mask: u64 = (1u64 << bit_of(flop[0]))
        | (1u64 << bit_of(flop[1]))
        | (1u64 << bit_of(flop[2]));
```

Then inside the combo pair loop, replace:
```rust
                        let used: u64 = flop_mask
                            | (1u64 << card_index(h1))
                            | (1u64 << card_index(h2))
                            | (1u64 << card_index(o1))
                            | (1u64 << card_index(o2));
```
with:
```rust
                        let used: u64 = flop_mask
                            | (1u64 << bit_of(h1))
                            | (1u64 << bit_of(h2))
                            | (1u64 << bit_of(o1))
                            | (1u64 << bit_of(o2));
```

**Important:** Check that `u8::from(Card)` exists in rs_poker. If not, use a `HashMap<Card, u32>` instead. The `From<u8>` in the flamegraph suggests rs_poker does `Card -> u8` conversions, so this should work. If `Card` doesn't impl `Into<u8>`, look for `Card::as_u8()`, `Card::to_u8()`, or derive the index from `value * 4 + suit`.

**Step 2: Run tests**

```bash
cargo test -p poker-solver-core -- equity_table
cargo test -p poker-solver-core -- bitmask_used_cards
```

**Step 3: Run benchmark**

```bash
cargo bench -p poker-solver-core -- equity_table
```

Expected: additional ~5-10% improvement from eliminating per-card linear scans.

**Step 4: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs
git commit -m "perf: pre-compute card-to-bit index to eliminate hot-loop conversions"
```

---

### Task 4: Cache equity tables across SPR solves

The trainer loops over SPRs sequentially, calling `build_for_spr()` for each. Each call rebuilds the equity table for the same flops. The equity table depends only on flop cards, not SPR. With 2 SPRs × 2 flops, this wastes 50% of equity computation.

**Files:**
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs` (`build_exhaustive` signature)
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` (`build_for_spr`)
- Modify: `crates/trainer/src/main.rs` (`build_postflop_with_progress`)

**Step 1: Add `pre_equity_tables` parameter to `build_exhaustive`**

Change the signature of `build_exhaustive` (line ~759) to accept optional pre-computed equity tables:

```rust
pub(crate) fn build_exhaustive(
    config: &PostflopModelConfig,
    tree: &PostflopTree,
    layout: &PostflopLayout,
    node_streets: &[Street],
    flops: &[[Card; 3]],
    pre_equity_tables: Option<&[Vec<f64>]>,
    on_progress: &(impl Fn(BuildPhase) + Sync),
) -> PostflopValues {
```

Inside the `par_iter` closure, replace lines 786-787:
```rust
            let combo_map = build_combo_map(&flop);
            let equity_table = compute_equity_table(&combo_map, flop);
```
with:
```rust
            let equity_table = if let Some(tables) = pre_equity_tables {
                tables[flop_idx].clone()
            } else {
                let combo_map = build_combo_map(&flop);
                compute_equity_table(&combo_map, flop)
            };
```

**Step 2: Update `build_for_spr` to accept and forward equity tables**

In `postflop_abstraction.rs`, add the parameter to `build_for_spr`:

```rust
    pub fn build_for_spr(
        config: &PostflopModelConfig,
        spr: f64,
        _equity_table: Option<&super::equity::EquityTable>,
        pre_equity_tables: Option<&[Vec<f64>]>,
        on_progress: impl Fn(BuildPhase) + Sync,
    ) -> Result<Self, PostflopAbstractionError> {
```

Forward it to the `build_exhaustive` call (line ~333):
```rust
            PostflopSolveType::Exhaustive => build_exhaustive(config, &tree, &layout, &node_streets, &flops, pre_equity_tables, &on_progress),
```

Also update `build()` (line ~304) which calls `build_for_spr` — pass `None` there.

**Step 3: Compute equity tables once in the trainer's SPR loop**

In `crates/trainer/src/main.rs`, in `build_postflop_with_progress`, before the SPR loop (around line 387):

```rust
    // Pre-compute equity tables once for all SPRs (equity is flop-dependent, not SPR-dependent).
    let flops = if let Some(ref names) = pf_config.fixed_flops {
        poker_solver_core::preflop::postflop_hands::parse_flops(names)
            .map_err(|e| format!("invalid flops: {e}"))?
    } else {
        poker_solver_core::preflop::postflop_hands::sample_canonical_flops(pf_config.max_flop_boards)
    };

    let equity_tables: Vec<Vec<f64>> = if pf_config.solve_type == PostflopSolveType::Exhaustive && sprs.len() > 1 {
        eprintln!("Pre-computing equity tables for {} flops...", flops.len());
        use poker_solver_core::preflop::postflop_hands::build_combo_map;
        use poker_solver_core::preflop::postflop_exhaustive::compute_equity_table;
        flops.par_iter().map(|flop| {
            let combo_map = build_combo_map(flop);
            compute_equity_table(&combo_map, *flop)
        }).collect()
    } else {
        vec![]
    };
    let pre_tables = if equity_tables.is_empty() { None } else { Some(equity_tables.as_slice()) };
```

Then pass `pre_tables` to each `build_for_spr` call:
```rust
        let abstraction = PostflopAbstraction::build_for_spr(
            pf_config,
            spr,
            equity,
            pre_tables,
            |phase| { ... },
        )
```

**Step 4: Run the full solve to verify correctness**

```bash
cargo run -p poker-solver-trainer --release -- solve-postflop -c sample_configurations/perf_test_1flop.yaml -o local_data/akq_vs_234_postflop
```

Output should be identical to previous runs.

**Step 5: Run with 2 SPRs to verify caching**

Create a test config with 2 SPRs and verify the solve runs ~2x faster for the equity phase:

```bash
# Modify perf_test_1flop.yaml to have postflop_sprs: [2, 20]
time cargo run -p poker-solver-trainer --release -- solve-postflop -c sample_configurations/perf_test_1flop.yaml -o local_data/akq_vs_234_postflop
```

Expected: significantly faster than 2× the single-SPR time, since equity is computed once.

**Step 6: Commit**

```bash
git add crates/core/src/preflop/postflop_exhaustive.rs crates/core/src/preflop/postflop_abstraction.rs crates/trainer/src/main.rs
git commit -m "perf: cache equity tables across SPR solves (flop-only dependency)"
```

---

### Task 5: End-to-end timing validation

**Files:**
- No code changes — validation only

**Step 1: Run criterion benchmark**

```bash
cargo bench -p poker-solver-core -- equity_table
```

Compare to Task 1 baseline. Document improvement.

**Step 2: Run the original config**

Restore the original `AKQr_vs_234r_postflop.yaml` config (2 flops, 2 SPRs, 10 iterations) and time it:

```bash
time cargo run -p poker-solver-trainer --release -- solve-postflop -c sample_configurations/AKQr_vs_234r_postflop.yaml -o local_data/akq_vs_234_postflop
```

**Step 3: Run full test suite**

```bash
cargo test -p poker-solver-core
cargo test -p poker-solver-trainer
cargo clippy --all-targets
```

**Step 4: Commit any fixups**

If clippy or tests reveal issues, fix and commit.

---

## Notes for the implementer

- `Card` is `rs_poker::core::Card`. Check how it converts to/from `u8` — the flamegraph shows `From<u8>` is hot, so `u8::from(card)` likely works.
- `all_cards_vec()` returns 52 cards in a fixed order (value × suit). This order is stable and can be relied upon for index mapping.
- The `compute_equity_table` inner loop is `O(169² × ~6² × ~1081)` ≈ 1.1 billion iterations per flop. Even small per-iteration savings compound.
- `build_combo_map` is cheap (~microseconds). No need to cache it.
- The `pub` visibility change for `compute_equity_table` is needed for the criterion bench. If this is undesirable, use `#[doc(hidden)] pub` or a `pub(crate)` with a `#[cfg(test)]` re-export. The simplest path is just `pub`.
