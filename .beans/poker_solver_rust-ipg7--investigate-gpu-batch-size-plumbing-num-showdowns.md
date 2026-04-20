---
# poker_solver_rust-ipg7
title: Investigate gpu_batch_size plumbing + num_showdowns discrepancy
status: draft
type: task
created_at: 2026-04-20T05:21:19Z
updated_at: 2026-04-20T05:21:19Z
blocked_by:
    - poker_solver_rust-vb8r
---

Follow-up to bean `vb8r` (OOM fix via skipping all-zero showdown buffer).

## Why

Expected host RAM at `gpu_batch_size=32, num_showdowns=24` was ~28 GB per code-explorer's analysis of `batch.rs:322-325` and `pipeline.rs:1015-1021`. Actual OOM was 127 GB RSS. 4× discrepancy, unexplained.

Possible causes:

1. Config not reaching orchestrator — `DatagenConfig.gpu_batch_size` read from YAML but not plumbed into `run_gpu_turn` / `GpuBatchSolver::new` / `prepare_batch`. Maybe a hardcoded default is used.
2. `num_showdowns` is significantly larger than 24 for the canonical turn tree at SPR=100 with bet sizes `[25%, 50%, 100%, a]` × `[25%, 75%, a]`.
3. Another large host allocator not yet identified.

## Work

- Confirm `gpu_batch_size: 32` from YAML is actually used in the orchestrator (print it in a debug log or unit test).
- Confirm actual `num_showdowns` for the canonical turn tree (log it at `build_canonical_turn_tree`).
- If both look correct, profile host RSS growth during `prepare_batch` (e.g. tcmalloc heap profile or simple `procfs` polling) and find the missing allocator.

## Priority / blocking

Non-blocking if bean `vb8r` fixes OOM at `batch=256`. If `vb8r` succeeds only at `batch<64`, this bean is promoted to high priority.
