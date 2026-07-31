---
# poker_solver_rust-iu44
title: Preserve MP flop raise-depth actions in exact solve
status: todo
type: bug
priority: high
created_at: 2026-07-28T18:50:49Z
updated_at: 2026-07-28T18:50:49Z
parent: poker_solver_rust-mk2k
---

UniversalMpLazy exact flop solve currently converts only the first configured flop raise-size row into range-solver bet sizes, while lazy traversal supports depth-specific raise rows. Preserve the full configured depth vector and max raise depth so exact cached actions match live MP navigation. Add a multi-depth regression.
