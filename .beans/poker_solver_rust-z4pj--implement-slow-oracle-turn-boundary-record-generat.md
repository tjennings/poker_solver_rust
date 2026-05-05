---
# poker_solver_rust-z4pj
title: Implement slow-oracle turn-boundary record generator
status: in-progress
type: task
priority: high
created_at: 2026-05-05T02:56:54Z
updated_at: 2026-05-05T03:12:03Z
parent: poker_solver_rust-85k4
---

Generate turn-boundary training records by running the existing river-enumeration evaluator as the oracle target source for sampled public turn states, ranges, pot, and stack metadata.



Started with crates/cfvnet/src/datagen/turn_boundary_oracle.rs: a pure builder that averages legal river runout oracle CFVs into a turn-boundary TrainingRecord.
