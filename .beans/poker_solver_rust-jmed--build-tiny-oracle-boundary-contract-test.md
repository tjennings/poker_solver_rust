---
# poker_solver_rust-jmed
title: Build tiny oracle-boundary contract test
status: in-progress
type: task
priority: high
created_at: 2026-05-04T01:09:18Z
updated_at: 2026-05-04T01:17:46Z
parent: poker_solver_rust-e90m
---

Create a minimal one-boundary game where full exact and depth-limited exact_oracle should match within numerical noise. Use it to prove whether the boundary handoff fails before poker complexity enters.

## Work Notes

Starting step 2. Goal: add a tiny one-boundary contract test that compares a depth-limited exact-oracle handoff against the full exact solve, before investigating orientation/unit/reach variants.
