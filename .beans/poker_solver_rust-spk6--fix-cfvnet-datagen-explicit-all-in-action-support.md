---
# poker_solver_rust-spk6
title: Fix CFVNet datagen explicit all-in action support
status: in-progress
type: bug
priority: critical
created_at: 2026-05-04T15:16:21Z
updated_at: 2026-05-04T15:24:05Z
---

River CFVNet datagen currently parses "a" out of bet_sizes and disables all-in thresholds, so all-in is not guaranteed as an explicit action despite configs and Supremus/DeepStack abstractions including all-in. Fix the datagen action-tree construction, validate with tests/eval, update config/docs as needed, and prepare a new data generation command.

## Plan

- [x] Baseline test suite passes under one minute on warm run (50.24s).
- [x] Research/brainstorm confirmed the minimal fix: preserve explicit all-in in typed datagen bet sizes and fuzz only pot-relative sizes.
- [ ] Implement explicit all-in preservation in CFVNet domain datagen tree construction.
  - [x] Local research confirmed the domain path can use range-solver BetSize directly.
- [ ] Update regression tests and any stale comments/tests that assumed all-in was skipped.
- [ ] Run focused and full tests.
- [ ] Generate a tiny fixed-seed smoke dataset and run datagen-eval.
- [ ] Prepare the full regenerated-data command and QA manifest plan.
