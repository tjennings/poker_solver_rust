# Subgame Rollout Perf Fixes + Hands/Sec Telemetry — Design

**Date:** 2026-04-17
**Status:** Approved, pending implementation plan

## Problem

The Tauri bounded subgame solver runs rollouts using the blueprint strategy. Rollouts are forward-simulation only (no regret/strategy-sum updates), so they should be at least as fast as the blueprint MCCFR trainer (~100k hands/sec). In practice they are significantly slower.

## Root Causes (Investigation Summary)

Investigation diffed the two hot paths:

- **MCCFR hot loop:** `crates/core/src/blueprint_v2/trainer.rs:596-691`, traversal `mccfr.rs:644`
- **Rollout hot loop:** `crates/tauri-app/src/postflop.rs:411-484`, inner `crates/core/src/blueprint_v2/continuation.rs:143`

Five concrete causes, in priority order:

1. **Per-combo `Vec<f64>` allocation for opponent weights** at `postflop.rs:432-442`. 1,326 heap allocs per evaluate call in the wide-range case. The weights are the *opponent's* range — they do not vary by hero combo, so this computation is redundant as well as allocating.
2. **Deck reconstruction at every Chance node.** `remaining_deck()` at `continuation.rs:85-106` builds a `Vec<Card>` of length ~45-47 at every flop→turn and turn→river transition in every rollout.
3. **Nested parallelism concern** — on reread, the `std::thread::spawn` at `postflop.rs:1354` is the solve's UI-responsiveness thread, not a parallel domain. The real issue is granularity: combo-level `par_iter` (`postflop.rs:425`) under-saturates cores when ranges are filtered tight on later streets.
4. **Bucket lookups recomputed per street.** `continuation.rs:213-216` caches within a street but invalidates at Chance nodes, hitting the mmap'd bucket file 2-6× per rollout. MCCFR precomputes all 4 streets once per deal.
5. **Per-decision-node `Vec` allocs** for `action_classes` and `bias_strategy`'s output at `continuation.rs:227-228`. 5-10 allocations per rollout.

Clean: no `emit()` in the hot loop, no logging/tracing, no Mutex acquisition, no config reloads.

**There is no existing hands/sec telemetry on the rollout path** — only MCCFR prints `it/s` (`trainer.rs:1136`). We cannot attribute individual fixes without first landing a counter.

## Goals

1. Match or approach MCCFR's ~100k hands/sec throughput on the rollout path.
2. Add per-solve hands/sec telemetry to the existing progress bar so the user can see throughput in real time.
3. Enable per-commit attribution of each optimization.

## Non-Goals

- Touching MCCFR internals.
- Touching the `range-solver` crate.
- New frontend components (reuse the existing progress display).
- New benchmarks except where a commit's speedup is non-obvious.

## Design

### Ordering

Measure first, then optimize. A single PR with 6 atomic commits so per-commit speedup is visible via `git bisect` / revert and the PR description can carry an attribution table.

### Commit sequence

1. **`feat(postflop): add rollout hands/sec telemetry`**
   Global `AtomicU64` `rollout_hands` on `PostflopState`. Every rollout worker increments once per terminal reached. `PostflopProgress` gains `rollout_hands_per_sec: f32` computed as `rollout_hands / elapsed_secs`. Frontend polls `postflop_get_progress` (already in place); adds the new number alongside existing iteration progress. Mirrors MCCFR's `{:.0} it/s` format. This commit lands the baseline measurement.

2. **`perf(postflop): hoist opponent weights out of per-combo loop`**
   Compute the 169-entry opponent weight table once per `evaluate()` call. Pass as `&[f32]` into combo workers. Eliminates 1,326× (or N× for filtered ranges) `Vec<f64>` allocations. Addresses suspect #1.

3. **`perf(continuation): precompute 4-street bucket table per rollout`**
   Mirror MCCFR's `DealWithBuckets`: when the rollout enters from a boundary state, compute `[[u16; 4]; 2]` (bucket per street per player) once at deal-generation. Thread through `rollout_inner`. Eliminates per-street `get_bucket()` mmap hits. Addresses suspect #4.

4. **`perf(continuation): replace remaining_deck Vec with u64 mask`**
   `RolloutContext` carries a `u64` dead-card mask (low 52 bits). Sampling a random remaining card is a weighted pick over set bits (or trailing-zero-count from a random rotation). Eliminates per-Chance-node `Vec<Card>` allocation. Addresses suspect #2.

5. **`perf(continuation): pass RolloutScratch through rollout_inner`**
   New `RolloutScratch { action_classes: SmallVec<[ActionClass; 16]>, bias_probs: SmallVec<[f32; 16]> }`, constructed once per par-unit and reused across all rollouts in that unit. `bias_strategy` is rewritten to write into the scratch buffer instead of returning `Vec<f32>`. Addresses suspect #5.

6. **`perf(postflop): flatten to combo × rollout parallelism`**
   Replace nested `par_iter` over combos with `par_iter` over flat `(combo_idx, rollout_idx)` pairs built from `iproduct!` or equivalent. Saturates cores on tight-range (river) spots where combo count drops to 30-100. Per-combo opponent-weight slice from commit 2 is shared via `Arc<[f32]>` or `&'env [f32]`. Addresses suspect #3 reshaped.

### Scratch API shape

- **One `RolloutScratch` per par-unit** (the `(combo, rollout)` pair under commit 6's scheme).
- **Passed as `&mut RolloutScratch`** through `rollout_inner` — no thread-locals, no pools.
- **Backed by `SmallVec`** sized 16 (matches MCCFR's action-buffer sizes at `mccfr.rs:914-921`).
- **Opponent weights are separate** — shared `&[f32]` slice, not part of the scratch.

### Parallelism

- Flat `par_iter` over `combo × rollout` pairs.
- Rationale: ranges filter as the game progresses; late-street combo counts can drop to 30-100, which under-saturates combo-level parallelism on 12+ core machines. Flat parallelism always saturates; cache-locality cost (hero cards no longer pinned to a thread) is minor — hero is 2 cards.
- Opponent weights computed once, lifetime-bound to the `par_iter` closure via `&'env [f32]`.

### Correctness

Each commit must leave the full test suite green:
- `cargo test` (all crates)
- `cargo test -p range-solver-compare --release` (identity harness)

No rollout-output change expected from any commit — these are pure perf optimizations with identical numerical behavior. The `RolloutScratch` and bucket-table changes must be verified equivalent to the existing computation by retaining at least one test that compares rollout chip values end-to-end.

### Measurement

The commit-1 counter records total hands processed. PR description will carry:

| commit | hands/sec | delta |
|--------|-----------|-------|
| baseline (commit 1) | ? | — |
| opponent-weight hoist (2) | ? | +? |
| 4-street bucket precompute (3) | ? | +? |
| deck mask (4) | ? | +? |
| scratch struct (5) | ? | +? |
| flat par_iter (6) | ? | +? |

Target: approach ~100k hands/sec.

## Risks

- **Lifetime plumbing for the shared opponent weights** under flat `par_iter` may require `rayon::scope` or `Arc<[f32]>`. Either works; pick the one that lets the type signatures stay readable.
- **Deck mask sampling correctness** — a bitmask-based random pick with correct uniform weighting needs a tested primitive (the current `remaining_deck()` + `choose()` is implicitly correct). Unit-test the new sampler against a known distribution before rolling in.
- **SmallVec inline size** — if any action context has >16 actions (shouldn't in the current tree-building), the SmallVec spills to heap and we lose the win. Assert or `debug_assert!` on the upper bound in the scratch constructor.
