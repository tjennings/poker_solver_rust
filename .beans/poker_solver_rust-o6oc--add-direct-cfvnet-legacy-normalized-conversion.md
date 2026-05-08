---
# poker_solver_rust-o6oc
title: Add Direct CFVNet legacy normalized conversion
status: in-progress
type: bug
priority: high
created_at: 2026-05-08T20:20:35Z
updated_at: 2026-05-08T20:20:35Z
---

Current Direct CFVNet checkpoint outputs normalized chip_ev/(pot+stack) values. Add the temporary conversion back to solver-native bcfv at inference so the current local model can be used while preserving clear documentation of the compatibility tradeoff.\n\n- [ ] Verify baseline full test suite is clean and under one minute\n- [ ] Patch Direct evaluator conversion for current normalized checkpoint\n- [ ] Update tests and docs for temporary Direct compatibility\n- [ ] Run focused and full verification\n- [ ] Commit code and bean
