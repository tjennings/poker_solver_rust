---
# poker_solver_rust-iaem
title: Investigate turn_boundary_cfvnet_v5 normalization mismatch
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T04:08:04Z
updated_at: 2026-05-14T04:08:04Z
parent: poker_solver_rust-8e9f
---

Investigate why local_data/models/turn_boundary_cfvnet_v5, trained after CFVNet boundary normalization work, produces boundary CFV magnitudes incompatible with runtime contracts.\n\n- [ ] Pin v5 best.pt checkpoint metadata and config\n- [ ] Identify training data path and dataset manifest/target units\n- [ ] Inspect Rust dataset writer target units for turn_boundary river_net source\n- [ ] Inspect Python loader target conversion and loss target units\n- [ ] Compare raw model outputs against expected normalized ranges on records or diagnostic spot\n- [ ] Report likely root cause and next fix
