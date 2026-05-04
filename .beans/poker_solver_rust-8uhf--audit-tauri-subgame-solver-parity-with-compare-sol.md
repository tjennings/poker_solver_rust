---
# poker_solver_rust-8uhf
title: Audit Tauri subgame solver parity with compare-solve
status: in-progress
type: task
priority: high
created_at: 2026-05-04T06:25:29Z
updated_at: 2026-05-04T06:25:29Z
parent: poker_solver_rust-e90m
---

Audit the Tauri subgame solve path against trainer compare-solve to identify any behavioral mismatches in game construction, boundary selection, evaluator wiring, gadget handling, seeding, iterations, and reported strategy output.\n\n## Scope\n\n- Compare Tauri solve entrypoints against crates/trainer/src/compare_solve.rs.\n- Focus on exact_subtree and cfvnet boundary behavior for subgame solves.\n- Report findings with file/line references and recommend follow-up fixes if needed.
