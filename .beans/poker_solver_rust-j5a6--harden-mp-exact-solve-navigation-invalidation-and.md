---
# poker_solver_rust-j5a6
title: Harden MP exact solve navigation invalidation and cache rewind
status: in-progress
type: bug
priority: high
created_at: 2026-07-28T20:37:17Z
updated_at: 2026-07-28T20:37:17Z
parent: poker_solver_rust-ja8p
---

Fix the final review findings for Universal MP lazy turn/river navigation.

- [ ] Preserve completed exact-solve caches when back-navigation stays within the same street/state family.
- [ ] Capture solve generation atomically with solve inputs and guard all worker completion/error state updates against stale generations.
- [ ] Add deterministic regressions for same-street rewind and navigation during solve setup where feasible.
- [ ] Update docs/architecture.md to remove stale turn/river unsupported wording.
- [ ] Run focused Tauri/core tests and diff checks.
