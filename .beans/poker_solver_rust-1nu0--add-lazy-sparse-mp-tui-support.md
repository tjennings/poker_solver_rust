---
# poker_solver_rust-1nu0
title: Add lazy sparse MP TUI support
status: in-progress
type: task
priority: high
created_at: 2026-05-08T02:04:22Z
updated_at: 2026-05-08T02:04:22Z
parent: poker_solver_rust-5kvv
---

Restore train-blueprint-mp TUI support when training.backend is lazy_sparse.

Tasks:
- [ ] Remove the lazy_sparse --no-tui hard block and add a lazy TUI run path.
- [ ] Bridge lazy sparse iterations, quit, snapshot trigger, and telemetry into MP TUI metrics.
- [ ] Provide useful live strategy/regret updates from sparse storage without dense tree scans.
- [ ] Update docs for lazy_sparse TUI behavior and limitations.
- [ ] Run focused trainer/core tests.
