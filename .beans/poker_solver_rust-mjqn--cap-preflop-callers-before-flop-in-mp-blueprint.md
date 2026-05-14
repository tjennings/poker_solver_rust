---
# poker_solver_rust-mjqn
title: Cap preflop callers before flop in MP blueprint
status: in-progress
type: feature
priority: normal
created_at: 2026-05-14T00:39:57Z
updated_at: 2026-05-14T00:45:53Z
---

Add configurable maximum number of players allowed to continue from preflop to the flop during train-blueprint-mp. Enforce by removing call/check-call after the first call unless the actor is closing action, so non-closing overcalls can be pruned while action-closing calls remain legal.\n\n- [x] Research current MP preflop action generation and config path\n- [x] Design configurable cap semantics and defaults\n- [ ] Implement cap without changing postflop action generation\n- [ ] Add focused tests for call removal and action-closing exception\n- [ ] Update training docs/config docs if user-facing config changes\n- [ ] Run full test suite under 1 minute and target trainer command sanity check
