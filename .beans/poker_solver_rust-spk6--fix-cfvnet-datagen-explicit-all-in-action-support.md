---
# poker_solver_rust-spk6
title: Fix CFVNet datagen explicit all-in action support
status: completed
type: bug
priority: critical
created_at: 2026-05-04T15:16:21Z
updated_at: 2026-05-04T15:45:48Z
---

River CFVNet datagen currently parses "a" out of bet_sizes and disables all-in thresholds, so all-in is not guaranteed as an explicit action despite configs and Supremus/DeepStack abstractions including all-in. Fix the datagen action-tree construction, validate with tests/eval, update config/docs as needed, and prepare a new data generation command.

## Plan

- [x] Baseline test suite passes under one minute on warm run (50.24s).
- [x] Research/brainstorm confirmed the minimal fix: preserve explicit all-in in typed datagen bet sizes and fuzz only pot-relative sizes.
- [x] Implement explicit all-in preservation in CFVNet domain datagen tree construction.
  - [x] Local research confirmed the domain path can use range-solver BetSize directly.
- [x] Update regression tests and any stale comments/tests that assumed all-in was skipped.
- [x] Run focused and full tests.
- [x] Generate a tiny fixed-seed smoke dataset and run datagen-eval.
- [x] Prepare the full regenerated-data command and QA manifest plan.

## Summary of Changes

Preserved explicit all-in in CFVNet domain datagen by parsing bet sizes into typed range-solver BetSize values instead of dropping "a" and converting everything to pot-relative floats. Bet-size fuzzing now only mutates PotRelative entries and leaves AllIn unchanged. GameBuilder and domain pipeline test helpers were updated to pass typed sizes through to range-solver BetSizeOptions for both opening bets and raises.

Added regression tests for lowercase/uppercase all-in parsing, fuzz preservation, root explicit all-in exposure, and raise-node explicit all-in exposure with all-in thresholds disabled.

Validation completed: cargo test -p cfvnet datagen::domain passed (68 passed, 1 ignored); git diff --check passed; tiny smoke generation wrote 8 records to /tmp/cfvnet_allin_smoke_tRyTu; datagen-eval loaded those records successfully; full cargo test passed in a true warm run at 50.31s.

Full production-scale data regeneration is tracked separately by poker_solver_rust-5e71 because it is a long-running data job.
