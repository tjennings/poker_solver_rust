---
# poker_solver_rust-85k4
title: 'Milestone: turn-boundary oracle dataset generator'
status: in-progress
type: feature
priority: high
created_at: 2026-05-05T02:55:44Z
updated_at: 2026-05-06T00:27:34Z
parent: poker_solver_rust-fp06
---

Build the data generator that creates turn-boundary training examples by enumerating river runouts with the current slow oracle path and storing one averaged turn-boundary target per boundary/player request.\n\n## Acceptance\n\n- Generator emits turn-boundary records with schema metadata.\n- Oracle can use exact river solves or a chosen river CFVNet.\n- Generator supports resumable sharded output and deterministic seeds.\n- Output includes enough metadata for stratified validation and coverage reports.



Progress: turn-boundary generation now has a CLI/config route (street=turn_boundary), oracle source selection, exact/river-net adapters, and sharded manifest output. Remaining milestone gap is richer coverage/validation bucket summaries in the manifest.



Coverage gap reduced: turn-boundary manifest now records stratified counters for generated data. Validation split generation remains under poker_solver_rust-q93y.

Progress: exact-river production config now favors faster sample throughput with solver_iterations=350 and target_exploitability=0.5, while retaining bounded river-runout parallelism via threads=4. Use tighter settings for trusted validation shards.
