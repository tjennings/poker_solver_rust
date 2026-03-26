---
# poker_solver_rust-nv3f
title: Boundary CFV magnitude calibration — rollout values too large
status: todo
type: bug
created_at: 2026-03-26T13:35:42Z
updated_at: 2026-03-26T13:35:42Z
---

The RolloutLeafEvaluator produces boundary CFVs with magnitudes far exceeding the physically achievable range (e.g. ±16 when max should be ±1.3). The rollout walks the abstract V2 tree whose bet sizes don't match the solve tree's, producing oversized terminal payoffs. The relative ordering is correct so the solver converges directionally, but exploitability plateaus around 18-32 instead of <5.

## TODO
- [ ] Investigate: unit-game terminal payoffs may use abstract tree pot/invested instead of boundary pot/invested
- [ ] Option A: Clamp rollout chip values to ±(remaining_stack / unit_pot) before converting to bcfv
- [ ] Option B: Run rollout with boundary-specific bet sizes instead of abstract tree sizes  
- [ ] Option C: Use equity-based boundary values as a simpler alternative to rollouts
- [ ] Validate: compare boundary CFVs against exact range-solver solution for a small test case
