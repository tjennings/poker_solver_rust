---
# poker_solver_rust-lr0y
title: Restore passing complete test-suite baseline
status: in-progress
type: bug
priority: high
created_at: 2026-08-04T12:59:50Z
updated_at: 2026-08-04T13:04:53Z
---

The required baseline cargo test run fails to compile poker-solver-trainer: nine tests in crates/trainer/src/inspect_spot.rs construct poker_solver_tauri::GameAction with struct literals, but exact_amount_bb is private and cannot be supplied. Repair the tests or public construction API through the mandated research, design, implementation, and review pipeline. The user explicitly waived the one-minute runtime requirement for this task.

- [ ] Research the GameAction construction API and regression origin
- [ ] Brainstorm the minimal compatible repair
- [ ] Plan and dispatch implementation in an isolated worktree
- [ ] Review the repair
- [ ] Confirm the complete suite passes
- [ ] Summarize the outcome
