---
# poker_solver_rust-krhd
title: Measure turn datagen throughput at gpu_batch_size=32 and 256
status: todo
type: task
priority: high
created_at: 2026-04-21T01:00:58Z
updated_at: 2026-04-21T01:01:18Z
---

Validation follow-up to bean \`6c5k\` (Layers B+C rayon parallelization).

## Motivation

6c5k landed Layer A (batched ORT call), Layer B (rayon per-sit prep), Layer C (rayon f64 reduction). At \`gpu_batch_size=4\` the wall clock was unchanged vs Layer A alone (~45 min for 32 samples / 4 files). Observed under the hood:
- CPU utilization went from 313% peak (pre-rayon) → 400% peak (post-rayon).
- GPU utilization went from ~0% persistent → 100% peak (was stuck waiting for CPU).
- RSS modestly lower.

The interpretation is that at batch=4, wall clock is dominated by the DCFR GPU kernel iterations — the CPU work rayon parallelizes is below the critical path. The predicted throughput gain only materializes when boundary-eval CPU work becomes a large share of per-batch time, which happens at larger batches.

## What to measure

1. Run \`cargo run -p cfvnet --release --features gpu-turn-datagen -- generate ... --num-samples 32 --per-file 32\` at:
   - \`gpu_batch_size: 32\` (1 batch per file)
   - \`gpu_batch_size: 256\` (1 batch for all 32 samples)
2. Record wall clock, peak RSS, peak GPU utilization, peak total CPU %.
3. Compare against the batch=4 baseline (~45 min for 32 samples).
4. Compute samples/sec; compare against the 50M-in-1h target (13,889 samples/sec).

## Expected outcome

- If Layer B+C actually helps at larger batch, wall-clock-per-sample should drop significantly at batch=32 vs batch=4. Possibly near-linear in batch size until another bottleneck appears.
- If wall clock stays proportional (no scaling benefit from rayon), something else is the bottleneck — likely the DCFR kernel time scaling with num_edges.

## Follow-ups this bean may surface

- If DCFR kernel is the new dominant cost, look at whether the kernel itself can be optimized (less iteration, different algorithm, batched kernel launches).
- If ORT inference itself is the dominant cost, look at TensorRT engine size / kernel fusion.
- If memory (48 GB GPU) limits batch size, look at halving DCFR state buffers.

## Priority

High — this determines whether Layer B+C is enough for production datagen, or whether we need further architectural changes. Should be quick: just two datagen runs (~30-60 min each max) plus analysis.
