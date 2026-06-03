---
# poker_solver_rust-l6r9
title: 'Phase 2: validate lazy tree trainer against known-good HU game'
status: draft
type: task
priority: high
created_at: 2026-06-03T18:09:40Z
updated_at: 2026-06-03T18:10:49Z
parent: poker_solver_rust-34kn
blocked_by:
    - poker_solver_rust-kqpn
    - poker_solver_rust-6y86
---

Phase 2 of the blueprint trainer tree roadmap.

Scope:
- Ingest the small known-good heads-up game variant and expected output data supplied by the user.
- Build a validation harness/runbook comparing the lazy tree trainer against the known-good baseline.
- Validate strategy outputs, regret/strategy-sum evolution, legal action expansion, terminal utilities, and deterministic replay where applicable.
- Capture tolerances explicitly: exact equality where deterministic/integer-like, numeric epsilon where floating-point CFR accumulation requires it.
- Produce a short validation report suitable for keeping with docs or test fixtures.

Acceptance criteria:
- The Phase 1 lazy tree implementation matches the supplied HU baseline within documented tolerances.
- The validation fixture is reproducible by command line.
- Any mismatch produces actionable diagnostics: node path/history, infoset/bucket, legal actions, strategy, regret, utility delta.
- No pruning or disk eviction is enabled during validation.

Blocked until Phase 1 exists and the user supplies the known-good HU data.
