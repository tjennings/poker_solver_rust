---
# poker_solver_rust-31v7
title: Honor per-bet-round CFVNet datagen bet rows
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T20:32:31Z
updated_at: 2026-05-04T20:32:31Z
---

Fix CFVNet domain datagen so game.bet_sizes rows are interpreted by betting round/depth instead of flattening rows after the first into one shared raise pool. The target Supremus-style abstraction has configured rows for the first bet and following bet/raise rounds, with four rounds after the first before forcing all-in-only continuation.
