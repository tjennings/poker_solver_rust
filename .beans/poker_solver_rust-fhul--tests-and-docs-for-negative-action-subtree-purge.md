---
# poker_solver_rust-fhul
title: Tests and docs for negative-action subtree purge
status: completed
type: task
priority: high
created_at: 2026-05-14T14:32:13Z
updated_at: 2026-05-14T16:07:43Z
parent: poker_solver_rust-xl3h
blocked_by:
    - poker_solver_rust-85ui
---

Add focused tests and documentation for the experimental negative-action subtree purge. Tests should cover config defaults/parsing, purge prefix correctness, sibling preservation, traversal allocation blocking, DCFR-driven reactivation, and default behavior unchanged when disabled. Update docs/training.md and docs/architecture.md with the training keys, recommended 6-max experiment settings, and the caveat that `prune_explore_pct` should be 0.0 for this purge strategy. Acceptance: full test suite passes in under one minute and docs describe how to run the experiment.

## Implementation Notes

Starting after telemetry landed in commit adf978c8. Most focused code coverage now exists in the config, sparse storage, lazy traversal, and telemetry slices; this task should fill any obvious remaining test gaps and update docs/training.md plus docs/architecture.md with the experimental purge semantics, config keys, telemetry fields, and recommended `prune_explore_pct: 0.0` setting.

## Summary of Changes

Added the missing disabled-default behavior test for the negative-action gate and updated docs/training.md plus docs/architecture.md to describe the experimental lazy sparse purge policy. The docs now cover config keys, prune/reactivation semantics, first-visit descendant behavior after reactivation, the `prune_explore_pct: 0.0` recommendation, and the `neg_action[...]` no-TUI telemetry fields. Focused `negative_action` tests passed, and independent review found no blocking docs/test issues.
