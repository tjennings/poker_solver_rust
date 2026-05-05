---
# poker_solver_rust-z4pj
title: Implement slow-oracle turn-boundary record generator
status: completed
type: task
priority: high
created_at: 2026-05-05T02:56:54Z
updated_at: 2026-05-05T03:42:45Z
parent: poker_solver_rust-85k4
---

Generate turn-boundary training records by running the existing river-enumeration evaluator as the oracle target source for sampled public turn states, ranges, pot, and stack metadata.



Started with crates/cfvnet/src/datagen/turn_boundary_oracle.rs: a pure builder that averages legal river runout oracle CFVs into a turn-boundary TrainingRecord.



The pure averaging builder now has concrete oracle sources available underneath it: a river-net adapter for scaled data and an exact-river adapter for validation/spot-check data. Next step is wiring selection into the dataset generation entrypoint.



Completed by wiring street=turn_boundary generation through generate_turn_boundary_data. The generator samples turn states, emits OOP/IP turn-boundary records through the shared river-runout averaging builder, and writes manifest-backed shards.
