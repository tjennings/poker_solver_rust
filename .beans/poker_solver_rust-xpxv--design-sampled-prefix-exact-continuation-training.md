---
# poker_solver_rust-xpxv
title: Design sampled-prefix exact-continuation training mode
status: completed
type: task
priority: normal
created_at: 2026-05-19T15:51:31Z
updated_at: 2026-05-19T15:53:38Z
---

Write a design note for sampling a deal prefix and fully traversing the continuation in the MP lazy trainer, including estimator semantics, implementation plan, risks, and validation.

## Summary of Changes

- Added docs/plans/2026-05-19-sampled-prefix-exact-continuation.md.
- Captured sampled full deal, sampled-turn exact-river, and sampled-flop exact-turn/river modes.
- Separated exact chance continuation from exact opponent/action continuation.
- Documented estimator weighting, implementation phases, validation, and risks.
