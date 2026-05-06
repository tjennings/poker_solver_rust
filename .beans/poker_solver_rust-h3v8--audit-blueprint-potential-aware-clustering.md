---
# poker_solver_rust-h3v8
title: Audit blueprint potential-aware clustering
status: completed
type: task
priority: normal
created_at: 2026-05-06T15:51:16Z
updated_at: 2026-05-06T16:35:58Z
---

Audit the blueprint clustering implementation for correctness issues and deviations from potential-aware bucketing research.

- [x] Locate the blueprint clustering implementation and call sites
- [x] Compare implemented features/distance metric/training flow against potential-aware bucketing expectations
- [x] Identify concrete issues with file/line references
- [x] Summarize risks and recommended fixes

## Summary of Changes

Completed a read-only audit of blueprint potential-aware clustering. Key findings: histogram construction canonicalizes next-street boards without remapping hole-card combo indices; preflop clustering is canonical-hand mapping despite docs/design saying EMD over flop-bucket distributions; CFVNet river clustering discards centroid EVs, disabling ordered/weighted child-bucket distances; training and diagnostics paths can silently use stale or misleading bucket data. No Rust code was changed.

## Correction

Retracted the preflop finding as an algorithm defect. Lossless canonical preflop hand mapping is acceptable because strategic lossy bucketing begins on the flop with hole cards plus board cards. The remaining concern is documentation/config wording that describes preflop as EMD over flop-bucket distributions, which should be clarified separately from clustering correctness.
