---
# poker_solver_rust-7ei7
title: Investigate sampled river convergence versus iteration cap
status: completed
type: task
priority: normal
created_at: 2026-05-14T06:11:08Z
updated_at: 2026-05-14T06:17:48Z
---

Controlled diagnostic for sampled river datagen: hold spots fixed, disable target early stop, and verify exploitability changes as max solver iterations increase.

- [x] Add or run a controlled same-spot iteration sweep
- [x] Compare results against current datagen target-stop behavior
- [x] Document the interpretation for sampled river training runs

## Summary of Changes

Added cfvnet sampled-river-convergence diagnostic to hold sampled river spots fixed across iteration caps. Verified seed 123 one-spot sweep: with target disabled, 50/100/250 iterations improved avg exploitability from 251.2 to 89.3 to 32.8 mbb/h; with target enabled, 250 and 5000 both stopped at iteration 120 with 57.4 mbb/h, proving the current target can mask iteration cap increases. Documented the command and interpretation in docs/training.md.
