---
# poker_solver_rust-sp4z
title: Update BoundaryNet river datagen to Supremus bet sizes
status: completed
type: task
priority: normal
created_at: 2026-05-04T20:23:54Z
updated_at: 2026-05-04T20:27:48Z
---

Change sample_configurations/boundary_net_river_datagen.yaml to encode the Supremus-style river datagen betting rows: first bet 25/50/100/all-in, second bet 25/75/all-in, and third bet all-in only with fold/check/call implicit. Also updated the sample output path comments to local_data/cfvnet/river_supremus. Validation: cargo test -p cfvnet parse_full_config passed; cargo test -p cfvnet game_tree::tests passed; full cargo test passed in 50.81s.
