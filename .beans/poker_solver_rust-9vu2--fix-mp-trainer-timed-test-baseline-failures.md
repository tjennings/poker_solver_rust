---
# poker_solver_rust-9vu2
title: Fix MP trainer timed-test baseline failures
status: in-progress
type: bug
priority: high
tags:
    - tests
    - blueprint-mp
created_at: 2026-05-19T15:47:59Z
updated_at: 2026-05-19T15:47:59Z
---

Baseline full cargo test failed before Part 3 implementation due blueprint_mp::trainer 1-second timed-test overruns. Fix the test-suite blocker before continuing low-SPR flop parity smoke work.

Checklist:
- [x] Commit this blocker bean tracking state before implementation.
- [ ] Dispatch research/brainstorming on why these timed tests are flaking and the least risky fix.
- [ ] Dispatch Rust implementation in a separate worktree.
- [ ] Dispatch review before integration.
- [ ] Integrate accepted changes into the feature branch.
- [ ] Run targeted MP trainer timed tests.
- [ ] Run full cargo test under one minute.
- [ ] Complete the blocker bean and commit the final tracking update.
