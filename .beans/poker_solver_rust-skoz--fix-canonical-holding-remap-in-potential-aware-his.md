---
# poker_solver_rust-skoz
title: Fix canonical holding remap in potential-aware histograms
status: todo
type: bug
priority: high
created_at: 2026-05-06T16:39:00Z
updated_at: 2026-05-06T16:39:00Z
parent: poker_solver_rust-hfnv
---

build_bucket_histogram_u8 canonicalizes/sorts the child board key but indexes the previous-street bucket file with the raw combo index. Training histograms must canonicalize the extended board, apply the same suit mapping to the hole cards, and then call combo_index on the mapped holding.

Acceptance:
- Add focused tests with suit-isomorphic boards where raw and canonical combo indices differ
- Fix global turn/flop clustering histogram construction
- Audit per-flop clustering paths for the same failure mode
- Confirm MCCFR runtime lookup and clustering feature construction agree on board and holding canonicalization
