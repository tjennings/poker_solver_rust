---
# poker_solver_rust-eh4s
title: Compare eager and lazy Blueprint MP action generation
status: todo
type: bug
priority: high
created_at: 2026-05-20T13:27:39Z
updated_at: 2026-05-20T13:27:39Z
parent: poker_solver_rust-kiqt
---

Ensure the eager public tree and lazy_sparse public-state generator produce equivalent action sets for matched preflop paths.

## Subtasks

- [ ] Build a small shared list of canonical preflop paths to resolve in both backends
- [ ] Compare action vector length, variant order, labels, and chip amounts
- [ ] Include unopened paths, single-open response paths, call chains, and capped raise paths
- [ ] Identify whether divergence is in eager game_tree.rs, lazy_mccfr.rs, or path resolution
- [ ] Add parity tests that fail on future divergence
- [ ] Decide whether eager mirror should remain authoritative or be refactored to shared logic
