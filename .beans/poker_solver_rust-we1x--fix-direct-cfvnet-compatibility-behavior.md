---
# poker_solver_rust-we1x
title: Fix Direct CFVNet compatibility behavior
status: in-progress
type: bug
priority: critical
created_at: 2026-05-08T20:47:13Z
updated_at: 2026-05-08T20:57:35Z
---

Direct CFVNet compatibility conversion makes subgame behavior worse: subgame bets/shoves 100% while exact checks 100%. Audit model target units, bcfv conversion, sign/player orientation, and boundary evaluator handoff; patch only once the solver convention is verified.

- [x] Verify working tree clean and test baseline
- [x] Trace solver bcfv convention and current BoundaryNet target
- [x] Identify why legacy conversion causes all-bet/all-shove policy
- [x] Patch compatibility mode or roll back unsafe mapping
- [x] Run focused and full verification
- [x] Commit corrected unit-conversion patch
- [ ] Compare exact vs Direct boundary CFVs for reported 50/50 root split

## Notes

The current Python BoundaryNet trainer encodes targets as `bcfv * pot / (pot + effective_stack)`. The previous compatibility patch treated outputs as `chip_ev / (pot + effective_stack)` and applied `2 * output * (pot + effective_stack) / pot - 1`, which injects a false `-1` offset. The correct inverse for this checkpoint is `output * (pot + effective_stack) / pot`.

Verification: focused `cfvnet` conversion test passed, focused Tauri solver-log test passed, `git diff --check` passed, `beans check` passed, and warm full `cargo test --quiet` passed in 51.96s.

Residual: after the corrected conversion, the reported behavior regressed from 100% bet/shove back to the earlier 50/50 check/bet split while exact remains 100% check. That remaining mismatch is a boundary-value audit/model-quality problem, not explained by the removed affine offset.
