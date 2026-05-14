---
# poker_solver_rust-o9bc
title: Fix CFVNet neural boundary raw-CFV contract
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T04:17:18Z
updated_at: 2026-05-14T04:17:18Z
parent: poker_solver_rust-8e9f
---

Long-term contract fix for CFVNet boundary magnitude mismatch: neural evaluators currently return conditional chip EVs through compute_raw_cfvs_both, but range-solver raw boundaries require opponent-reach-integrated chip CFVs. Disable or repair that raw path and align tests/docs so CFVNet boundaries use one explicit conditional-value contract.\n\n- [ ] Add failing coverage for neural raw path contract decision\n- [ ] Change neural evaluator to avoid incorrect raw-CFV handoff\n- [ ] Update docs to describe conditional model output vs solver reach integration\n- [ ] Run focused boundary evaluator and compare-solve tests\n- [ ] Summarize result and follow-up target-unit cleanup
