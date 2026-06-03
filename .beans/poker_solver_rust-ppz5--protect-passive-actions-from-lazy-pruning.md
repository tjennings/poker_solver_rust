---
# poker_solver_rust-ppz5
title: Protect passive actions from lazy pruning
status: completed
type: bug
priority: high
created_at: 2026-05-14T18:56:08Z
updated_at: 2026-05-14T18:59:28Z
---

Lazy sparse traversal pruning currently skips any action whose regret is below prune_threshold, unlike eager MCCFR which protects terminal fold children. In MP, passive actions route to high-leverage terminal/showdown/later-player states, and pruning them may make open strategies degenerate after pruning starts. Restrict lazy traversal pruning to aggressive actions and add focused tests.

## Summary of Changes

- Changed lazy sparse traversal pruning so `prune_threshold` only skips aggressive actions.
- Passive structural actions (`Fold`, `Check`, `Call`, and all-in calls that do not increase the current max bet) remain traversable even when their regret is below the pruning threshold.
- Added focused tests for passive-action pruning protection and continued aggressive-action pruning.
- Updated training and architecture docs to document lazy aggressive-only pruning semantics.

## Validation

- cargo fmt --check
- cargo test -p poker-solver-core lazy_pruning -- --nocapture
- cargo test -p poker-solver-core negative_action -- --nocapture
- git diff --check
