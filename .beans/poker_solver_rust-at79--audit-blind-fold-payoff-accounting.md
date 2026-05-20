---
# poker_solver_rust-at79
title: Audit blind fold payoff accounting
status: completed
type: task
priority: normal
created_at: 2026-05-20T18:29:47Z
updated_at: 2026-05-20T18:32:22Z
---

Verify that multiplayer terminal payoff and related strategy reporting include posted blind contributions, so SB fold is -0.5bb and BB fold is -1bb in a 1/2 blind game.



Audit complete: MP terminal fold payoff already subtracts each seat's total contribution before awarding the pot. Tree forced bets populate contributions and terminal nodes preserve them. Added regression coverage proving SB fold at 1/2 blinds is -1 chip (-0.5bb) and BB fold is -2 chips (-1bb). Focused terminal/tree/MCCFR tests pass.
