---
# poker_solver_rust-f6c0
title: Fix TreeConfig doctest rooted-subgame fields
status: in-progress
type: bug
created_at: 2026-05-05T13:44:46Z
updated_at: 2026-05-05T13:44:46Z
---

Update the TreeConfig documentation example in crates/range-solver/src/action_tree.rs so range-solver doctests compile after rooted-subgame fields were added.\n\n- [ ] Inspect existing action_tree.rs implementation and doc example\n- [ ] Apply minimal documentation fix\n- [ ] Run cargo test -p range-solver --doc
