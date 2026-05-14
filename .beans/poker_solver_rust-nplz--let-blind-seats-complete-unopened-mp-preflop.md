---
# poker_solver_rust-nplz
title: Let blind seats complete unopened MP preflop
status: completed
type: bug
priority: high
created_at: 2026-05-14T20:12:03Z
updated_at: 2026-05-14T20:15:33Z
---

The MP allow_preflop_limp=false gate currently removes every unopened preflop Call, including SB completion. It should only remove cold limps from seats with no posted blind; SB can call/complete and BB can check when action reaches it.

## Summary of Changes

- Changed the MP no-limp gate to block only cold unopened preflop calls from seats with no blind posted.
- Preserved SB completion/call when allow_preflop_limp is false.
- Added eager and lazy tests that walk UTG/HJ/CO/BTN folds, assert those seats cannot cold-limp, assert SB can complete, and assert BB can check after SB completion.
- Updated training docs to describe the blind-seat exception.

## Validation

- cargo fmt --check
- cargo test -p poker-solver-core no_limp -- --nocapture
- git diff --check
