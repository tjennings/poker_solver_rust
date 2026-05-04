---
# poker_solver_rust-31v7
title: Honor per-bet-round CFVNet datagen bet rows
status: completed
type: bug
priority: high
created_at: 2026-05-04T20:32:31Z
updated_at: 2026-05-04T20:42:21Z
---

Fix CFVNet domain datagen so game.bet_sizes rows are honored by betting round instead of flattening every row after the first into one raise pool. Added optional per_num_bets support in range-solver BetSizeOptions; legacy callers keep using the existing raise list when that table is empty. CFVNet datagen now maps row 0 to the first bet, row 1 to the second bet / first raise, row 2 to the third bet, etc., and range-solver forces all-in-only raises beyond the configured rows. Added tests for all-in-only third-bet rows and for a five-row setup that allows four rounds after the first before forcing all-in. Updated docs and sample config comments to describe the new semantics. Validation: focused cfvnet per-round row tests passed, range-solver bet-size parser test passed, full cargo test passed warm in 53.72s.
