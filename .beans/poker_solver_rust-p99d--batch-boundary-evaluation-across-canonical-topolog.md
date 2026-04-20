---
# poker_solver_rust-p99d
title: Batch boundary evaluation across canonical-topology batch (turn datagen)
status: completed
type: task
priority: normal
created_at: 2026-04-20T06:51:09Z
updated_at: 2026-04-20T22:47:52Z
blocked_by:
    - poker_solver_rust-oox2
---

Follow-up to bean `vb8r` (OOM fix). Surfaced while validating batch=256 in Task 8 of plan 2026-04-20-skip-zero-showdown-buffer.md.

## Problem

At `gpu_batch_size=256`, 2+ minutes into the run the main thread is pegged at 100% CPU, 7 worker threads at ~48%, GPU compute util 0% despite 40 GB of GPU memory allocated, and no output file has been produced. Memory is fine (peak RSS 20 GB, stable ~14 GB) — this is pure throughput pathology, not OOM.

## Root cause (from code-explorer agent analysis, 2026-04-20)

The batched turn datagen work in commit `22971ba` batches the DCFR kernel across 256 games but does NOT batch the boundary (river) evaluation. Inside the DCFR iteration loop, `evaluate_boundaries_batched` is called serially once per game with a batch-of-1 request:

- `crates/cfvnet/src/datagen/domain/pipeline.rs:772-823` — per-sit eval loop calling `evaluate_boundaries_batched(&[single_request])` in a serial for-loop over the 256 specs.
- `crates/cfvnet/src/datagen/domain/pipeline.rs:652-665` — initial pre-DCFR eval that must fully complete before any DCFR kernel launches.
- `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:207-269` — f64 CPU reduction of size `num_boundaries × 1326 × 48`, main thread.
- `crates/cfvnet/src/datagen/gpu_boundary_eval.rs:40-48, 147-173` — `zero_conflicting_hands` called ~96× per boundary (52² each).

With `solver_iterations: 300, leaf_eval_interval: 50` (~6 eval rounds × 256 games per batch = 1536 batch-of-1 ORT calls) before first batch completes.

## Work

- [ ] Make `evaluate_boundaries_batched` actually batch across the canonical topology's 256 specs in a single ORT `session.run`.
- [ ] Parallelize the f64 CPU reduction in `gpu_boundary_eval.rs:207-269` with rayon over specs.
- [ ] Parallelize `zero_conflicting_hands` preprocessing the same way.
- [ ] Re-test at `gpu_batch_size=256` and confirm first file completes in reasonable time.
- [ ] Measure throughput (samples/s) and compare against the 50M-in-1h target.

## Workaround until fixed

`gpu_batch_size=32` (the handoff's recommendation) avoids the worst of this and still gets ~8× DCFR-launch amortization over batch=4.

## Priority

High — blocks production turn datagen at intended batch size. Without this, we can't hit the 50M-sample target at batch=256, and at batch=32 we may still be bottlenecked on boundary eval rather than the DCFR kernel (validates after `vb8r` soak test at batch=32 finishes).



## Summary of Changes (Layer A only)

Hoisted evaluate_boundaries_batched out of the per-sit loop in batch_boundary_leaf_cfvs at \`crates/cfvnet/src/datagen/domain/pipeline.rs:746-826\`. The ORT call now runs once per DCFR eval round with all sits' requests concatenated, instead of once per sit.

**Plan:** \`docs/plans/2026-04-20-boundary-eval-batching.md\`
**Design:** \`docs/plans/2026-04-20-boundary-eval-batching-design.md\`

**Commits on main:**
- \`7287ff6\` test(cfvnet): regression test for batched boundary eval
- \`1939ef0\` feat(cfvnet): batch boundary eval across canonical topology

**Validation:**

1. Regression test \`layer_a_batched_matches_per_sit_within_tolerance\` passes: batched path matches per-sit path within 1e-5 abs / 1e-4 rel tolerance. No TensorRT kernel-selection drift observed.
2. Full workspace tests: same 8 pre-existing failures as oox2, no new regressions.
3. Manual validation: production datagen (\`gpu_batch_size=4, solver_iterations=300\`) went from ~15 min/file (oox2 baseline) to ~11 min/file (~30% improvement) for 2 of 4 files before killed.

**Reality check:** the 30% improvement at batch=4 is far short of the hoped-for speedup. Observed: main thread 100%, 7 ORT threads at ~30%, GPU 0% util for most wall time. Root cause: the CPU-side f64 reduction in \`evaluate_boundaries_batched:207-269\` is serial per request and dominates main-thread time. Layer A only amortized the ORT call; the reduction was untouched.

**Filed as follow-up:** bean \`6c5k\` — Rayon-parallelize the f64 reduction (Layer C). Expected to be the actual throughput unlock.

**Unblocks:**
- Oox2's end-to-end full validation: the numerically-equivalent batched path is in place.
- \`6c5k\` (Layer C): the batched request layout makes per-request rayon parallelism straightforward.
