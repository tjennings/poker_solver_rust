---
# poker_solver_rust-9uee
title: Turn parity smoke for range-solver-compare
status: in-progress
type: task
priority: high
tags:
    - range-solver
    - compare
created_at: 2026-05-19T14:48:33Z
updated_at: 2026-05-19T14:58:01Z
---

Add deterministic turn-street parity smoke coverage to the range-solver-compare harness.

Checklist:
- [x] Commit this bean tracking state before implementation.
- [x] Run the full test suite baseline and confirm it passes in under one minute. Warm baseline: `cargo test` passed in 48.95s after an initial compile-including run took 77.76s.
- [x] Dispatch research/brainstorming on the correct turn parity dimensions and scope. Result: reuse existing tolerant smoke comparators, add two deterministic turn spots at 50-100 iterations, avoid exact equality/random soak for normal smoke.
- [ ] Dispatch Rust implementation in a separate worktree.
- [ ] Dispatch review before integration.
- [ ] Integrate accepted changes into the feature branch.
- [ ] Run turn smoke compare tests, existing river compare smoke tests, and the full suite under one minute.
- [ ] Complete the bean and commit the final tracking update.
