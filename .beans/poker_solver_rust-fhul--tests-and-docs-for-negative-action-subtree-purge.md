---
# poker_solver_rust-fhul
title: Tests and docs for negative-action subtree purge
status: todo
type: task
priority: high
created_at: 2026-05-14T14:32:13Z
updated_at: 2026-05-14T14:32:13Z
parent: poker_solver_rust-xl3h
blocked_by:
    - poker_solver_rust-85ui
---

Add focused tests and documentation for the experimental negative-action subtree purge. Tests should cover config defaults/parsing, purge prefix correctness, sibling preservation, traversal allocation blocking, DCFR-driven reactivation, and default behavior unchanged when disabled. Update docs/training.md and docs/architecture.md with the training keys, recommended 6-max experiment settings, and the caveat that `prune_explore_pct` should be 0.0 for this purge strategy. Acceptance: full test suite passes in under one minute and docs describe how to run the experiment.
