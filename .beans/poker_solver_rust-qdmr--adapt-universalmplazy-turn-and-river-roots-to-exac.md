---
# poker_solver_rust-qdmr
title: Adapt UniversalMpLazy turn and river roots to exact solver
status: in-progress
type: task
priority: high
created_at: 2026-07-29T13:00:12Z
updated_at: 2026-07-29T13:17:11Z
parent: poker_solver_rust-g7yj
---

Implement the street-aware UniversalMpLazy exact solve adapter on top of the existing range-solver/full-depth backend. Preserve root action semantics and cache/generation behavior for two-player turn and river decisions.


## Implementation Notes

The research phase confirmed the range-solver already supports non-terminal turn and river roots. This task now owns the street-aware UniversalMpLazy adapter, while preserving the explicit two-player boundary and existing solve-generation invalidation.
