---
# poker_solver_rust-v55b
title: Fix test-suite runtime and gpu-range-solver failures on non-CUDA hosts
status: todo
type: task
priority: high
created_at: 2026-04-17T16:17:32Z
updated_at: 2026-04-17T16:20:19Z
---

Two issues blocking CLAUDE.md's <60s test-suite rule:

1. **gpu-range-solver has 34 failing tests on macOS** (kernel compile, CUDA state alloc, mega-kernel, batch solver). These should be feature-gated or cfg-gated so they only run where CUDA is available.

2. **cargo test --all takes ~4:10** — needs to drop under 60s per CLAUDE.md. Start by profiling the suite to find the slowest tests, then decide between parallelization, #[ignore]-ing slow integration tests, or moving them to --release benches.

**Why now:** blocking the perf/subgame-rollout PR workflow from using the full test suite as its regression gate. That PR is scoped to non-GPU crates and gates on `cargo test --workspace --exclude gpu-range-solver` as a workaround.

## Update 2026-04-17

Narrower gate `cargo test -p poker-solver-core -p poker-solver-tauri` runs in **22s** but still has one pre-existing failure: `blueprint_mp::mccfr::tests::traverse_updates_strategy_sums` at crates/core/src/blueprint_mp/mccfr.rs:539. Failure likely fallout from 3a8f168e (REGRET_SCALE 20→1 rescale) — strategy sums not updated after single traversal. Unrelated to subgame rollout path this PR modifies.

Split suggestion: a separate beans/PR fixes the `traverse_updates_strategy_sums` expectation or the underlying rescale bug. Blocks fully-green workspace but does not block perf/subgame-rollout work.
