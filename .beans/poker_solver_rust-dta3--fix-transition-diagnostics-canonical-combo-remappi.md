---
# poker_solver_rust-dta3
title: Fix transition diagnostics canonical combo remapping
status: in-progress
type: bug
priority: normal
created_at: 2026-05-06T16:39:16Z
updated_at: 2026-05-06T20:30:29Z
parent: poker_solver_rust-hfnv
---

diag-clusters transition consistency canonicalizes boards but reads bucket files using raw combo_idx. The diagnostic must apply the same board suit mapping to the holding before bucket lookup, otherwise it can measure a different transition distribution than training/runtime.

Acceptance:
- Share or mirror the canonical board+holding lookup helper used by clustering/runtime
- Add a diagnostic regression test with a non-identity suit mapping
- Ensure flop→turn and turn→river transition audits report on canonicalized combo IDs
- Mention the corrected semantics in docs/training.md diagnostics section if useful

## Work Start

Started after confirming a clean worktree on `codex/audit-blueprint-clustering`. Plan: run the pre-change full test suite, inspect `audit_transition_consistency`, update diagnostic child-bucket lookups to canonicalize holdings under the child board suit mapping, add a non-identity suit-mapping regression test, and rerun focused plus full tests.
