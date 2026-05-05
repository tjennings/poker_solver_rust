---
# poker_solver_rust-h3z5
title: Support exact-river and river-net oracle target sources
status: completed
type: task
priority: high
created_at: 2026-05-05T02:56:57Z
updated_at: 2026-05-05T03:42:46Z
parent: poker_solver_rust-85k4
---

Allow the dataset generator to emit targets from exact river solving where available and from the current river CFVNet oracle path for scale, recording the source for every shard.



Started by defining the RiverRunoutOracle trait boundary so exact-river and river-net adapters can share the same turn-boundary averaging path.



Added oracle adapters in crates/cfvnet/src/datagen/turn_boundary_oracle.rs: BoundaryNetRiverRunoutOracle wraps the existing river BoundaryNet inference contract and converts normalized chip/(pot+stack) outputs back to pot-relative CFVs, while ExactRiverSolverOracle wraps solve_situation for exact river targets. Both zero river-board blockers before target evaluation.



Completed source selection for the first generator path with datagen.turn_boundary_target_source = river_net | exact_river. River-net uses the GPU BoundaryNet adapter when built with gpu-turn-datagen; exact_river uses solve_situation.
