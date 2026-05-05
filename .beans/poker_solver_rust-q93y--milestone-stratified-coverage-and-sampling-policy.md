---
# poker_solver_rust-q93y
title: 'Milestone: stratified coverage and sampling policy'
status: todo
type: feature
priority: high
created_at: 2026-05-05T02:55:44Z
updated_at: 2026-05-05T02:55:44Z
parent: poker_solver_rust-fp06
---

Design and implement sampling so the turn-boundary dataset covers normal play plus difficult regions: deep raises, tiny-pot/high-SPR, low-SPR/all-in, blueprint-derived ranges, and RSP ranges.\n\n## Acceptance\n\n- Coverage report tracks samples by pot, stack, SPR, raise depth, boundary ordinal, all-in proximity, board texture, and source.\n- Policy can oversample sparse 3-bet/4-bet/5-bet+ states.\n- Production configs can target a balanced validation split by strata.
