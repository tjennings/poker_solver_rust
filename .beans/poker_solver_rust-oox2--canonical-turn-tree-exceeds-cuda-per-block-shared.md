---
# poker_solver_rust-oox2
title: Canonical turn tree exceeds CUDA per-block shared memory limit (blocks GPU turn datagen)
status: completed
type: bug
priority: critical
created_at: 2026-04-20T17:55:56Z
updated_at: 2026-04-20T19:42:03Z
---

Blocker discovered 2026-04-20 while validating bean `vb8r` (OOM fix) end-to-end. Turn datagen now fails 100% with `CUDA_ERROR_INVALID_VALUE` at the first `run_iterations` launch, for any `gpu_batch_size`. **Not caused by vb8r** — this is a pre-existing bug introduced by commit `22971ba` (batched turn datagen with canonical topology).

## Repro

```
cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \
  -c sample_configurations/turn_gpu_datagen.yaml \
  -o local_data/cfvnet/turn/gpu_v1_batch4_validation \
  --num-samples 32 --per-file 8
```

```
[tree] memory per game: 57.9 MB
data generation failed: run_iterations failed: DriverError(CUDA_ERROR_INVALID_VALUE, "invalid argument")
```

## Root cause (from code-explorer, 2026-04-20)

The `cfr_solve` kernel requests ~103 KB of dynamic shared memory for the canonical turn tree (`num_edges=6590`, `num_nodes=6591`, `max_depth=16`). CUDA's default per-block limit is 48 KB and Ada's opt-in max is ~99 KB. Both are exceeded → the driver rejects the launch config.

Breakdown of the 103 KB request (`gpu.rs:355-360`):
- `edge_parent/edge_child/edge_player` — 3 × 6590 × 4 = **79 KB** (dominant)
- `level_starts/level_counts` — 2 × 17 × 4 = ~0.1 KB
- `actions_per_node` — 6591 × 4 = 26 KB

River test topology (`make_river_game`) needs <1 KB of shared memory, which is why all unit tests pass and this went undetected.

## Fix approach (TBD — open for design)

1. Opt in to max dynamic shared memory on the `cfr_solve` kernel via `cudaFuncSetAttribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, N)` (cudarc equivalent) — necessary but not sufficient.
2. Move `edge_parent/child/player` (79 KB) out of dynamic shared memory into global memory reads. These arrays are already uploaded as globals (`d_edge_parent/child/player`), so this is a kernel-source edit to change smem references to global reads. With that change, remaining smem = 26 KB — comfortably under 48 KB with no opt-in needed.
3. Add launch-time assertion on `shared_mem_bytes` vs `cudaDevAttrMaxSharedMemoryPerBlockOptin`.
4. Add a turn-sized topology test so this regression cannot recur silently.

## Key file:line references

- `crates/gpu-range-solver/src/gpu.rs:349-360` — `compute_hand_parallel_shared_mem` formula
- `crates/gpu-range-solver/src/batch.rs:203-204` — shared_mem_bytes computed at `GpuBatchSolver::new`
- `crates/gpu-range-solver/src/batch.rs:441-446` — `LaunchConfig` with oversized `shared_mem_bytes`
- `crates/gpu-range-solver/src/batch.rs:155` — `HandParallelKernel::compile` site where opt-in would go
- `crates/cfvnet/src/datagen/domain/game_tree.rs:207-230` — `build_canonical_turn_tree` (SPR=100)
- `crates/cfvnet/src/datagen/domain/pipeline.rs:486-499` — `canonical_num_hands=1326`, solver construction site

## Priority

Critical — blocks all GPU turn datagen. Also blocks beans `vb8r` (can't complete end-to-end validation) and `p99d` (boundary-eval throughput — can't measure until kernel launches work).



## Summary of Changes

Moved `edge_parent/edge_child/edge_player` out of the `cfr_solve` kernel's dynamic shared memory and into direct global-memory reads. The three arrays were already uploaded as GPU globals — the smem copy was redundant. Dynamic smem budget for the canonical turn tree drops from **105,580 bytes (103 KB)** to **26,428 bytes (26 KB)** — comfortably under CUDA's 48 KB default per-block limit.

**Plan:** \`docs/plans/2026-04-20-cfr-kernel-smem-fix.md\`

**Commits (all on main, oldest first):**

| SHA | Title |
|-|-|
| d6dc095 | test(gpu-range-solver): failing test for canonical turn tree smem budget |
| efc0aac | fix(gpu-range-solver): remove edge-array smem copies from cfr_solve kernel |
| e013d8b | feat(gpu-range-solver): opt in to Ada's 99 KB per-block dynamic smem |
| e4c0092 | test(gpu-range-solver): debug-assert cfr_solve smem budget at launch |
| 265f544 | test(cfvnet): smoke test for canonical turn tree GPU solve |

**Validation:**

1. Unit test \`canonical_turn_tree_smem_fits_under_cuda_default_limit\` passes (no CUDA needed).
2. Integration test \`canonical_turn_tree_runs_one_iteration_without_smem_overflow\` passes on RTX 6000 Ada — 117s in debug build, launches successfully with all \`strategy_sum\` finite.
3. Production datagen command exercised end-to-end:
   \`\`\`
   cargo run -p cfvnet --release --features gpu-turn-datagen -- generate \\
     -c sample_configurations/turn_gpu_datagen.yaml \\
     -o local_data/cfvnet/turn/smem_fix_verify --num-samples 32 --per-file 8
   \`\`\`
   Exit 0, 4/4 files written (276 KB each, 64 GPU turn records total). Log closed with "Done."
4. Clippy: no new warnings introduced. Pre-existing warnings in untouched files only.
5. Workspace tests: 201 passed. 8 pre-existing failures in \`poker-solver-trainer::mp_tui_scenarios\` — 10s timer too tight for 13M-node MP tree build (even isolated, single-threaded, the test takes 10.065s). Unrelated to this fix.

**Note for reviewers:** The debug assertion in \`batch.rs\` is set to 48 KB (the CUDA default), intentionally more conservative than the Ada 99 KB opt-in that \`set_attribute\` now enables. A future topology that pushes smem between 48 KB and 99 KB would debug-panic with a clear message instead of launching — callers can either shrink smem or raise the assert threshold.

**Unblocks:**
- \`vb8r\` — OOM fix can now be validated end-to-end (partially done already via this bean's Task 7).
- \`p99d\` — boundary-evaluation batching throughput work can proceed; the kernel now launches.
