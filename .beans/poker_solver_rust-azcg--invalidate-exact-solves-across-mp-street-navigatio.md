---
# poker_solver_rust-azcg
title: Invalidate exact solves across MP street navigation
status: completed
type: bug
priority: high
created_at: 2026-07-28T20:10:27Z
updated_at: 2026-07-28T21:25:56Z
parent: poker_solver_rust-ja8p
---

When a UniversalMpLazy session deals/backtracks across flop, turn, and river, any running exact solve must be cancelled or generation-invalidated before the new street state is published. A stale background worker must not repopulate solve matrices/cache after SolveState::reset or allow a second solve to race it. Add a regression for solve-then-street-navigation and preserve HU behavior.


## Summary of Changes

Added generation-safe invalidation for exact solves across Universal MP flop, turn, river, and back navigation. Stale worker success/error/completion cannot repopulate current overlays, while same-street cache rewind remains available.

## Verification

The dedicated Universal MP explorer integration target passed 27 tests on the final code; focused core lazy-MP tests passed 27 tests. Rustfmt and git diff --check passed. Exact MP solving remains two-player and flop-only.
