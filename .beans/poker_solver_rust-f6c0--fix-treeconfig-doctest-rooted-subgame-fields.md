---
# poker_solver_rust-f6c0
title: Fix TreeConfig doctest rooted-subgame fields
status: completed
type: bug
priority: normal
created_at: 2026-05-05T13:44:46Z
updated_at: 2026-05-05T13:45:27Z
---

Update the TreeConfig documentation example in crates/range-solver/src/action_tree.rs so range-solver doctests compile after rooted-subgame fields were added.

- [x] Inspect existing action_tree.rs implementation and doc example
- [x] Apply minimal documentation fix
- [x] Run cargo test -p range-solver --doc

## Summary of Changes

Updated the TreeConfig documentation example to include the rooted-subgame initialization fields with fresh-street defaults. Verified with cargo test -p range-solver --doc.
