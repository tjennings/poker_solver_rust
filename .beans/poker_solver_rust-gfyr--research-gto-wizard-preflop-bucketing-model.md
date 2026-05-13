---
# poker_solver_rust-gfyr
title: Research GTO Wizard preflop bucketing model
status: completed
type: task
priority: normal
created_at: 2026-05-13T13:56:50Z
updated_at: 2026-05-13T13:57:57Z
---

Find public writings, if any, that identify the bucketing or abstraction model used by GTO Wizard for legacy/pre-solved preflop solutions.

## Summary of Changes

Searched GTO Wizard blog/help/glossary and related linked materials. Finding: GTO Wizard publicly states legacy/pre-solved preflop uses abstraction/bucketing and has artifacts consistent with lossy abstraction, but does not publish the exact bucketing model, bucket counts, feature vector, distance metric, or clustering pipeline for its own preflop solves. Their public pointers are generic: flop subsets credited to PioSolver, and bucketing references to HRC/Johanson-style distribution-aware/imperfect-recall abstraction literature.
