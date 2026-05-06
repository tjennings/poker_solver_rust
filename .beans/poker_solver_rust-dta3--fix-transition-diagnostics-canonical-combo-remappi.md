---
# poker_solver_rust-dta3
title: Fix transition diagnostics canonical combo remapping
status: completed
type: bug
priority: normal
created_at: 2026-05-06T16:39:16Z
updated_at: 2026-05-06T20:37:28Z
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

## Summary of Changes

Fixed transition consistency diagnostics to canonicalize board and holding together before bucket lookup, matching runtime and clustering semantics. The diagnostic now uses the board suit mapping for both the current-street bucket and each next-street bucket lookup, so flop→turn and turn→river audits no longer mix canonical board indices with raw combo indices. Added a non-identity suit-mapping regression test and documented `diag-clusters --transition-audit` semantics in `docs/training.md`.

Verification:
- `cargo test -p poker-solver-core blueprint_v2::cluster_diagnostics` passed.
- `time cargo test` passed after implementation; first post-change run included compilation and took 1:24.81.
- Warm-cache `time cargo test` passed in 57.805s.
- `cargo fmt -p poker-solver-core --check` still reports pre-existing formatting drift in unrelated files (`blueprint_mp/game_tree.rs`, `blueprint_mp/trainer.rs`, `nut_features.rs`); the changed import was manually adjusted to match rustfmt.
