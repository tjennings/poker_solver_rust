---
# poker_solver_rust-ruma
title: Run CFVNet paired-turn root-attribution sweep
status: in-progress
type: task
priority: high
created_at: 2026-05-04T14:40:42Z
updated_at: 2026-05-04T14:40:42Z
parent: poker_solver_rust-e90m
---

Use compare-solve --dump-boundary-cfvs to measure whether CFVNet root bet-pressure errors repeat across paired-turn Kx spots.\n\n## Tasks\n\n- [ ] Define a small paired-turn validation grid around the known Kh failure.\n- [ ] Run compare-solve CFVNet diagnostics on each spot.\n- [ ] Summarize root regret-input deltas and exact-vs-subgame strategy divergence.\n- [ ] Decide whether the next repair is model/data calibration, runtime guard/fallback, or feature/input audit.
