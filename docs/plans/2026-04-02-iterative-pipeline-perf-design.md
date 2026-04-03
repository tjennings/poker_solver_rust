# Iterative Pipeline Performance Optimization Design

**Date**: 2026-04-02
**Status**: Approved
**Branch**: `feat/iterative-pipeline-perf`
**Baseline**: Lockstep at commit `a1fc4d51` — 19.6 mbb/h, 0.9/s

## Problem

The working iterative pipeline (per-iteration boundary re-eval via GPU) achieves excellent exploitability (~20 mbb/h) but only 0.9 samples/sec because the GPU evaluates boundaries every single iteration in lockstep — CPU sits idle during GPU, GPU sits idle during CPU.

## Constraint

A previous async rewrite broke correctness (29k mbb/h instead of 20 mbb/h). Multiple changes were made simultaneously, so the root cause is unclear. This design uses **incremental phases with validation gates** — exploitability is verified after each phase before proceeding.

## Solution

Three phases, each validated independently:

### Phase 1: Add eval_interval to lockstep

No architectural changes. Skip GPU eval on most iterations.

```
loop:
    if any game's iter % eval_interval == 0:
        evaluate_pool_boundaries(games needing eval)
    solve_step all games (1 iteration)
    for games where next iter needs eval:
        flush_boundary_caches
    graduate/inject
```

Between GPU evals, `solve_step` reuses the last-set boundary CFVs. The flush only happens when the next iteration will trigger a GPU eval — this way the solver keeps using valid CFVs during the interval.

**Validation**: eval_interval=1 must match baseline (~20 mbb/h). Then sweep 5, 10, 30 — measure exploitability at each.

**Expected throughput gain**: eval_interval=30 → ~30x fewer GPU calls → estimated 5-25/s.

### Phase 2: Queues with GPU-side flush

Replace lockstep with two queues. Persistent solver threads. GPU and CPU overlap.

```
[Deal Buffer] → [Ready Queue] → [N Solver Threads] → [Eval Queue] → [GPU Thread] → [Ready Queue]
                                                    ↘ [Write Queue] (if done)
```

**Solver thread loop:**
1. Pop from ready queue
2. Run eval_interval iterations of `solve_step`
3. If done → finalize, extract, send to write queue
4. Else → send to eval queue (reaches intact, NO flush)

**GPU thread loop (main thread, owns model/device):**
1. Drain eval queue into batch
2. Read `boundary_reach` from each game → `build_iterative_game_inputs`
3. `flush_boundary_caches()` on each game (AFTER reading reaches)
4. `model.forward()` → `set_boundary_cfvs()`
5. Push to ready queue
6. Inject new deals when games graduate

**Key insight**: Flush happens on the GPU thread AFTER reading reaches, not on the solver thread before sending. This preserves reaches for GPU input and gives clean state for fresh CFVs.

**Validation**: eval_interval=1 must match baseline (~20 mbb/h). If exploitability breaks, channels are the problem — stop and investigate. If it holds, proceed.

### Phase 3: Independent sizing (config tuning)

No code changes beyond Phase 2. The queue architecture naturally decouples GPU and CPU sizing:

- `active_pool_size`: Total games in flight. Should be > threads.
- `gpu_batch_size`: Max games per GPU forward pass (defaults to pool size).
- `threads`: Number of persistent solver threads.

**Validation**: Sweep active_pool_size (64, 128, 256) × eval_interval (1, 5, 10, 30). Verify exploitability holds across all combos.

## Config Options

```yaml
datagen:
  mode: "iterative"
  active_pool_size: 256    # total games in flight
  leaf_eval_interval: 10   # GPU eval every N iterations
  solver_iterations: 300   # iterations per game
  threads: 18              # solver threads
```

## Files Changed

- `crates/cfvnet/src/datagen/turn_generate.rs` — all phases modify `generate_turn_training_data_iterative`
- `crates/cfvnet/src/config.rs` — no changes (active_pool_size and leaf_eval_interval already exist)
