---
# poker_solver_rust-midh
title: Research GTO Wizard non-AI solver architecture
status: completed
type: task
priority: normal
created_at: 2026-05-13T13:47:19Z
updated_at: 2026-05-13T13:48:48Z
---

Investigate public evidence for how GTO Wizard's non-AI solver works, especially whether it uses postflop bucketing/abstraction like Pluribus or exact/custom tree solving.

## Summary of Changes

Researched public GTO Wizard and Pluribus sources. Main finding: GTO Wizard pre-solved postflop solutions are described as standard CFR solves with betting-tree abstraction, not Pluribus-style postflop hand bucketing; preflop/multiway legacy solving uses or historically relied on abstraction/bucketing, while GTO Wizard AI uses depth-limited neural value approximation.
