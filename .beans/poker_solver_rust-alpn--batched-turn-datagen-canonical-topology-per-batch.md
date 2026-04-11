---
# poker_solver_rust-alpn
title: 'Batched turn datagen: canonical topology + per-batch fold payoffs'
status: in-progress
type: task
priority: high
created_at: 2026-04-11T19:22:01Z
updated_at: 2026-04-11T19:50:12Z
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
- [ ] Task 2b: Canonical topology + batched orchestrator

Recovered from crashed session (2026-04-11). Task 1 code changes appear
largely in place in working tree; needs verification + tests + commit.
