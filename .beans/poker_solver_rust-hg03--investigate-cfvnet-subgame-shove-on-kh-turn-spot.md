---
# poker_solver_rust-hg03
title: Investigate CFVNet subgame shove on Kh turn spot
status: in-progress
type: bug
priority: high
created_at: 2026-05-04T13:38:41Z
updated_at: 2026-05-04T13:38:41Z
parent: poker_solver_rust-e90m
---

Tauri subgame with cfvnet boundary reports an always-shove strategy while exact reports always-call on spot: sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Kh.\n\n## Tasks\n\n- [ ] Reproduce the spot in compare-solve with the local cfvnet model.\n- [ ] Compare against exact and exact_oracle/exact_subtree controls where possible.\n- [ ] Determine whether this is Tauri wiring, compare-solve parity, CFVNet value calibration, or model/domain mismatch.\n- [ ] Fix or document the next concrete repair step.
