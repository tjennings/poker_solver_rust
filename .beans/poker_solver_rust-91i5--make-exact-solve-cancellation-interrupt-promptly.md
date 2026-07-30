---
# poker_solver_rust-91i5
title: Make exact solve cancellation interrupt promptly
status: in-progress
type: bug
priority: high
created_at: 2026-07-30T19:08:47Z
updated_at: 2026-07-30T19:35:24Z
---

Exact postflop solves do not visibly stop promptly when the user presses cancel. The Tauri state has a cancel flag, but the exact range-solver traversal may not observe it until an entire CFR iteration completes, and the UI awaits cancellation/state refresh before settling.

- [x] Trace exact solve and cancel execution paths, including command concurrency
- [x] Add an immediate interrupt token checked by the exact solver at safe traversal boundaries
- [ ] Ensure the TUI cancel action returns promptly and reports the stopped solve state
- [ ] Add focused regression tests for exact cancellation
- [ ] Verify focused tests and document behavior if the command contract changes


## Review Findings

- [ ] Bind cancellation requests to the active solve generation so stale requests cannot cancel a newer solve
- [ ] Keep the TUI cancellation state/poller aligned with backend acknowledgement
- [ ] Reduce or eliminate the remaining exact tree-construction cancellation gap

Review also identified that terminal evaluator work remains cooperative only at traversal boundaries; this is acceptable for this patch if the remaining bounded latency is documented.



## Current Review Work

- [x] Baseline research: trace generation, cancel, and exact construction paths
- [ ] Return solve generation and reject stale generation cancellation
- [ ] Add cancellation-aware action-tree construction and exact worker checks
- [ ] Add focused stale-generation and tree-construction tests
- [ ] Run formatting and focused range-solver/tauri/devserver verification
- [ ] Document remaining uninterruptible construction latency and final test limitations
