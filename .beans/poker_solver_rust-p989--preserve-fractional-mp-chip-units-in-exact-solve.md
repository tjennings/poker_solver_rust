---
# poker_solver_rust-p989
title: Preserve fractional MP chip units in exact solve
status: todo
type: bug
priority: high
created_at: 2026-07-28T18:50:49Z
updated_at: 2026-07-28T18:50:49Z
parent: poker_solver_rust-mk2k
---

UniversalMpLazy exact root construction currently rounds raw fractional stacks, pots, street bets, and prior action amounts into integer range-solver inputs, and semantic matching can collide on fractional BB amounts. Define a lossless supported conversion or reject unsupported fractional states explicitly, then add fractional pot/stack/action regression coverage.
