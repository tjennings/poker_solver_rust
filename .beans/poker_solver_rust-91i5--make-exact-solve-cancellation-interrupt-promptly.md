---
# poker_solver_rust-91i5
title: Make exact solve cancellation interrupt promptly
status: completed
type: bug
priority: high
created_at: 2026-07-30T19:08:47Z
updated_at: 2026-07-30T19:59:12Z
---

Exact postflop solves do not visibly stop promptly when the user presses cancel. The Tauri state has a cancel flag, but the exact range-solver traversal may not observe it until an entire CFR iteration completes, and the UI awaits cancellation/state refresh before settling.

- [x] Trace exact solve and cancel execution paths, including command concurrency
- [x] Add an immediate interrupt token checked by the exact solver at safe traversal boundaries
- [x] Ensure the TUI cancel action returns promptly and reports the stopped solve state
- [x] Add focused regression tests for exact cancellation
- [x] Verify focused tests and document behavior if the command contract changes


## Review Findings

- [x] Bind cancellation requests to the active solve generation so stale requests cannot cancel a newer solve
- [x] Keep the TUI cancellation state/poller aligned with backend acknowledgement
- [x] Reduce or eliminate the remaining exact tree-construction cancellation gap

Review also identified that terminal evaluator work remains cooperative only at traversal boundaries; this is acceptable for this patch if the remaining bounded latency is documented.



## Current Review Work

- [x] Baseline research: trace generation, cancel, and exact construction paths
- [x] Return solve generation and reject stale generation cancellation
- [x] Add cancellation-aware action-tree construction and exact worker checks
- [x] Add focused stale-generation and tree-construction tests
- [x] Run formatting and focused range-solver/tauri/devserver verification
- [x] Document remaining uninterruptible construction latency and final test limitations

## Summary of Changes

Implemented generation-safe cooperative cancellation for exact and subgame solves. The solve API returns its backend generation, cancellation validates and atomically stores against that generation, exact game construction runs in the worker with recursive action-tree cancellation checks, and CFR traversal exits safely without publishing partial buffers or caches. The TUI sends the generation, remains in cancelling state until acknowledgement, and prevents overlapping restarts.

Focused verification passed: range-solver cancellation tests, Tauri cancellation tests, devserver parameter tests, 60 Vitest tests, frontend production build, and cargo formatting. Remaining bounded latency is inside PostFlopGame arena allocation and terminal/evaluator internals; these are checked before and after and never publish partial solve state, but are not forcibly preempted mid-operation.
