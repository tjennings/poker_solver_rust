---
# poker_solver_rust-we1x
title: Fix Direct CFVNet compatibility behavior
status: in-progress
type: bug
priority: critical
created_at: 2026-05-08T20:47:13Z
updated_at: 2026-05-08T20:47:13Z
---

Direct CFVNet compatibility conversion makes subgame behavior worse: subgame bets/shoves 100% while exact checks 100%. Audit model target units, bcfv conversion, sign/player orientation, and boundary evaluator handoff; patch only once the solver convention is verified.\n\n- [ ] Verify working tree clean and test baseline\n- [ ] Trace solver bcfv convention and current BoundaryNet target\n- [ ] Identify why legacy conversion causes all-bet/all-shove policy\n- [ ] Patch compatibility mode or roll back unsafe mapping\n- [ ] Run focused and full verification\n- [ ] Commit code and bean
