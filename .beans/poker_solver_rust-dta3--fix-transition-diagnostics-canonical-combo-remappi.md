---
# poker_solver_rust-dta3
title: Fix transition diagnostics canonical combo remapping
status: todo
type: bug
priority: normal
created_at: 2026-05-06T16:39:16Z
updated_at: 2026-05-06T16:39:16Z
parent: poker_solver_rust-hfnv
---

diag-clusters transition consistency canonicalizes boards but reads bucket files using raw combo_idx. The diagnostic must apply the same board suit mapping to the holding before bucket lookup, otherwise it can measure a different transition distribution than training/runtime.

Acceptance:
- Share or mirror the canonical board+holding lookup helper used by clustering/runtime
- Add a diagnostic regression test with a non-identity suit mapping
- Ensure flop→turn and turn→river transition audits report on canonicalized combo IDs
- Mention the corrected semantics in docs/training.md diagnostics section if useful
