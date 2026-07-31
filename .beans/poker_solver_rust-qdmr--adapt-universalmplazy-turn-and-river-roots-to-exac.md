---
# poker_solver_rust-qdmr
title: Adapt UniversalMpLazy turn and river roots to exact solver
status: in-progress
type: task
priority: high
created_at: 2026-07-29T13:00:12Z
updated_at: 2026-07-29T13:22:04Z
parent: poker_solver_rust-g7yj
---

Implement the street-aware UniversalMpLazy exact solve adapter on top of the existing range-solver/full-depth backend. Preserve root action semantics and cache/generation behavior for two-player turn and river decisions.


## Implementation Notes

The research phase confirmed the range-solver already supports non-terminal turn and river roots. This task now owns the street-aware UniversalMpLazy adapter, while preserving the explicit two-player boundary and existing solve-generation invalidation.


## Work Checklist

- [x] Research current UniversalMpLazy exact-solve and range-solver contracts
  - Range solver supports Flop/Turn/River roots but uses integer chip units and per-depth rows; the MP adapter currently rounds values and reuses flop sizes.
- [ ] Brainstorm street-aware adapter and fractional-value handling
- [ ] Plan implementation and focused regression coverage
- [ ] Dispatch implementation in an isolated worktree
- [ ] Review implementation and tests
- [ ] Integrate, run focused verification, and update docs if needed
- [ ] Commit code, tests, and owned bean changes atomically
