---
# poker_solver_rust-eote
title: Optimize cfvnet datagen pipeline (2h → target <30m for 1M spots)
status: done
type: task
created_at: 2026-03-12T04:19:10Z
updated_at: 2026-03-12T04:19:10Z
---

## Problem

`cargo run -p cfvnet --release -- generate` takes ~2 hours for 1M river spots with 8 threads. Target: <30 min.

## Root Causes (from perf review)

### P0: Memory explosion — all situations held in memory before solve
**File:** `crates/cfvnet/src/datagen/generate.rs:37-39`

All 1M `Situation` structs are collected into a Vec before solving begins. Each situation has two `[f32; 1326]` arrays (~10.5 KB each), so 1M situations = ~10 GB. Then the results Vec adds another ~16 GB. This causes massive memory pressure / page faults that likely dominate the 2h runtime.

**Fix:** Chunk the pipeline — generate N situations, solve in parallel, write, repeat:
```rust
const CHUNK: usize = 10_000;
for chunk_start in (0..num_samples).step_by(CHUNK) {
    let chunk_end = (chunk_start + CHUNK).min(num_samples);
    let situations: Vec<_> = (chunk_start..chunk_end)
        .map(|_| sample_situation(...))
        .collect();
    let results: Vec<_> = situations.par_iter()
        .map(|s| solve_situation(s, &config))
        .collect();
    // write results immediately, then drop both vecs
}
```

### P1: `compute_hand_strengths` called twice per spot (identical board)
**File:** `crates/cfvnet/src/datagen/range_gen.rs:116` (called from `sampler.rs`)

`generate_rsp_range` internally calls `compute_hand_strengths` for each range. For river spots this is 1,326 hand evals × 2 = wasted work. For turn spots it's 1,326 × 48 × 2 = ~127K evals wasted per spot.

**Fix:** Add `generate_rsp_range_with_strengths(strengths: &[u16; 1326], rng)` variant. Compute strengths once in `sample_situation`, pass to both OOP and IP range generation.

### P2: `BetSizeOptions` re-parsed per solve (1M string allocs)
**File:** `crates/cfvnet/src/datagen/solver.rs:64-65`

Every `solve_situation` call does `config.bet_sizes.join(",")` + `BetSizeOptions::try_from(...)`. That's 1M redundant string allocations + parses of identical input.

**Fix:** Parse `BetSizeOptions` once and store in `SolveConfig`:
```rust
pub struct SolveConfig {
    pub bet_sizes: BetSizeOptions,  // pre-parsed, not Vec<String>
    ...
}
```

## Secondary (do after P0-P2)

- **Sort scratch alloc in range_gen** (`range_gen.rs:94-98`): `raw_to_ordinal` allocates a `Vec<(i32, usize)>` per call (~1100 elements, called 2× per spot). Pass a reusable scratch buffer.
- **`Situation.board` is `Vec<u8>`** (`sampler.rs:9`): Heap alloc for 5 bytes. Use `[u8; 5]` or `ArrayVec<u8, 5>`.
- **Scalar `write_all` loop** (`storage.rs:48-57`): 1326×3 individual `write_all` calls per record. Use `bytemuck::cast_slice` to write each `[f32; 1326]` as a single `write_all`.
- **`bool_mask_to_u8` scalar loop** (`generate.rs:143-148`): May not auto-vectorize.

## Verification

- [ ] `cargo test -p cfvnet` passes
- [ ] Measure peak RSS with `/usr/bin/time -l` before and after P0
- [ ] Time 10K spots before/after each fix to measure improvement
- [ ] Final 1M run completes in <30 min

## Also fixed this session

- `sample_situation` now floors `effective_stack` at 5 (was 0, causing solver errors on `compare --num-spots 100`)
- Uncommented `threads: 8` in `sample_configurations/river_cfvnet.yaml`
