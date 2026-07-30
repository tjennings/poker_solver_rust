---
# poker_solver_rust-91i5
title: Make exact solve cancellation interrupt promptly
status: in-progress
type: bug
priority: high
created_at: 2026-07-30T19:08:47Z
updated_at: 2026-07-30T19:08:47Z
---

Exact postflop solves do not visibly stop promptly when the user presses cancel. The Tauri state has a cancel flag, but the exact range-solver traversal may not observe it until an entire CFR iteration completes, and the UI awaits cancellation/state refresh before settling.

- [ ] Trace exact solve and cancel execution paths, including command concurrency
- [ ] Add an immediate interrupt token checked by the exact solver at safe traversal boundaries
- [ ] Ensure the TUI cancel action returns promptly and reports the stopped solve state
- [ ] Add focused regression tests for exact cancellation
- [ ] Verify focused tests and document behavior if the command contract changes
