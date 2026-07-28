---
# poker_solver_rust-azcg
title: Invalidate exact solves across MP street navigation
status: in-progress
type: bug
priority: high
created_at: 2026-07-28T20:10:27Z
updated_at: 2026-07-28T20:10:27Z
parent: poker_solver_rust-ja8p
---

When a UniversalMpLazy session deals/backtracks across flop, turn, and river, any running exact solve must be cancelled or generation-invalidated before the new street state is published. A stale background worker must not repopulate solve matrices/cache after SolveState::reset or allow a second solve to race it. Add a regression for solve-then-street-navigation and preserve HU behavior.
