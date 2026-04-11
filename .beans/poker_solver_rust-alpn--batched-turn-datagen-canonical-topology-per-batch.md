---
# poker_solver_rust-alpn
title: 'Batched turn datagen: canonical topology + per-batch fold payoffs'
status: completed
type: task
priority: high
created_at: 2026-04-11T19:22:01Z
updated_at: 2026-04-11T20:18:14Z
parent: poker_solver_rust-qg6w
---

Plan: docs/plans/2026-04-11-batched-turn-datagen.md

Batch 256 turn games per GPU kernel launch using one canonical topology
(highest-SPR tree, no bet-size collapsing) and per-batch fold payoffs.
Eliminates per-game kernel compilation and fully utilizes GPU SMs.

Expected speedup: 8x (25s/game -> ~3.1s/game); boundary eval becomes new
bottleneck after DCFR batching.

## Tasks
- [x] Task 1: Per-batch fold payoffs in kernel (SubgameSpec, prepare_batch, kernel indexing, tests) — commit e0f811a
- [x] Task 2a: Per-batch leaf CFVs in gpu-range-solver (kernel + update_leaf_cfvs + tests) — commit e99fc38
- [x] Task 2b: Canonical topology + batched orchestrator — commit 22971ba

Recovered from crashed session (2026-04-11). Task 1 code changes appear
largely in place in working tree; needs verification + tests + commit.

## Summary of Changes

Task 2b (commit 22971ba): Rewrote cfvnet turn datagen orchestrator to use a single canonical turn tree built once at startup (pot=100, stack=10000 → SPR=100, no bet-size collapsing, 1326-hand universal layout). A single `GpuBatchSolver` instance serves all batched games; per-batch `SubgameSpec` supplies per-game initial weights (zeroing board-conflicting hands), per-game fold payoffs scaled from sit.pot, and per-game leaf CFVs from BoundaryNet. Removed diagnostic timing eprintln; added aggregate throughput logging every 1000 samples. New test `gpu_turn_batched_pipeline_produces_records` exercises batch_size=4 with 6 samples.

Plan lines 202-457 (Task 2) fully implemented. All 221 cfvnet tests pass (including the new batched test and the existing single-game turn test, which now uses the batched path with batch_size=1). All 94 gpu-range-solver tests still pass.
