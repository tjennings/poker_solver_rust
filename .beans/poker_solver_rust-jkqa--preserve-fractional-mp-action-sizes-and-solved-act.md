---
# poker_solver_rust-jkqa
title: Preserve fractional MP action sizes and solved-action identity
status: in-progress
type: bug
priority: high
created_at: 2026-07-29T13:53:54Z
updated_at: 2026-07-29T14:20:48Z
parent: poker_solver_rust-g7yj
---

The MP exact adapter preserves root chip state with scaling but still rounds fractional action percentages and later normalizes live/cached actions by rounded BB values. Use exact scaled action descriptors or an explicit representability policy so fractional sizes cannot map to the wrong action; add collision and non-integer-size regressions.


## Review Findings

The first adapter patch still needs to reconcile range-solver integer action quantization with raw fractional MP actions and the existing HU semantic matcher. Exact action descriptors must remain identity-safe at scaled values instead of comparing rounded BB labels.
