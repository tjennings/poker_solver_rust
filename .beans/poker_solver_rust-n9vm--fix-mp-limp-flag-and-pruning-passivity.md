---
# poker_solver_rust-n9vm
title: Fix MP limp flag and pruning passivity
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T19:55:40Z
updated_at: 2026-05-14T19:55:40Z
---

MP lazy training is improving but strategies look overly passive after protecting passive actions from pruning. Also game.allow_preflop_limp is present in the 6-max config but ignored by MP config/action generation. Implement MP allow_preflop_limp for eager/lazy action generation and adjust lazy pruning so it does not structurally bias toward passive actions while avoiding the earlier subtree wipeout.
