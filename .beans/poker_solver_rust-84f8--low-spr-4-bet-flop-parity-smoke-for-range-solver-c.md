---
# poker_solver_rust-84f8
title: Low-SPR 4-bet flop parity smoke for range-solver-compare
status: in-progress
type: task
priority: high
tags:
    - range-solver
    - compare
created_at: 2026-05-19T15:46:05Z
updated_at: 2026-05-19T15:54:52Z
---

Add deterministic low-SPR 4-bet flop parity smoke coverage to the range-solver-compare harness.

Checklist:
- [x] Commit this bean tracking state before implementation.
- [x] Run the full test suite baseline and confirm it passes in under one minute. Skipped for this session by user request because training is running; initial baseline failed in 27.41s due existing `blueprint_mp::trainer` wall-clock timed-test overruns under load, not compare-harness failures.
- [ ] Dispatch research/brainstorming on low-SPR 4-bet flop parity dimensions and scope.
- [ ] Dispatch Rust implementation in a separate worktree.
- [ ] Dispatch review before integration.
- [ ] Integrate accepted changes into the feature branch.
- [ ] Run flop smoke compare tests and existing turn/river compare smoke tests. Full-suite performance gate skipped for this session by user request.
- [ ] Complete the bean and commit the final tracking update.
