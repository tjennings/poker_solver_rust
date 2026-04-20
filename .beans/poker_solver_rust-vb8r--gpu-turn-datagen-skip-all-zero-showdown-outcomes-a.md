---
# poker_solver_rust-vb8r
title: 'GPU turn datagen: skip all-zero showdown outcomes allocation (OOM fix)'
status: in-progress
type: bug
priority: high
created_at: 2026-04-19T20:51:42Z
updated_at: 2026-04-20T17:56:05Z
blocked_by:
    - poker_solver_rust-oox2
---

## Problem

`cargo run -p cfvnet --release --features gpu-turn-datagen -- generate -c sample_configurations/turn_gpu_datagen.yaml` OOMs even at `gpu_batch_size: 32`. Confirmed via dmesg:

- `oom-kill: global_oom, task=cfvnet`
- `total-vm: 407 GB, anon-rss: 127.6 GB` at kill time
- Host RAM OOM, not GPU

## Root cause

The showdown outcomes buffer `[B × num_showdowns × 1326² × 4B]` per player is allocated host-side per batch per thread. For turn datagen these values are always zero — leaf injection from BoundaryNet supplies the real CFVs.

Math reconciles: `12 threads × 32 batch × 2 players × num_showdowns × 7 MB ≈ 127 GB` → `num_showdowns ≈ 24`. Plausible for the turn tree with `[25%, 50%, 100%, a]` × `[25%, 75%, a]` bet sizes.

## Goal

Eliminate the wasteful allocation for the turn-datagen leaf-injection path so even `gpu_batch_size: 256` fits in host+GPU memory, without breaking the river-datagen path which legitimately uses the buffer.

## Workflow stage

Brainstorming — design doc target: `docs/plans/2026-04-19-skip-zero-showdown-buffer-design.md`

## Out of scope

- Throughput optimization beyond the OOM fix
- Refactoring unrelated GPU solver code
- Touching the river datagen allocation path
