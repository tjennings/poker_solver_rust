---
# poker_solver_rust-h3z5
title: Support exact-river and river-net oracle target sources
status: in-progress
type: task
priority: high
created_at: 2026-05-05T02:56:57Z
updated_at: 2026-05-05T03:12:03Z
parent: poker_solver_rust-85k4
---

Allow the dataset generator to emit targets from exact river solving where available and from the current river CFVNet oracle path for scale, recording the source for every shard.



Started by defining the RiverRunoutOracle trait boundary so exact-river and river-net adapters can share the same turn-boundary averaging path.
