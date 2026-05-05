---
# poker_solver_rust-ewjj
title: 'Milestone: turn-boundary model contract and dataset schema'
status: completed
type: feature
priority: high
created_at: 2026-05-05T02:54:34Z
updated_at: 2026-05-05T03:08:12Z
parent: poker_solver_rust-fp06
---

Define the exact input/output contract, target semantics, units, metadata, and storage schema for the turn-boundary CFVNet.\n\n## Acceptance\n\n- Contract specifies 4-card board input, reaches, pot/stack normalization, player indicator, and 1326 bcfv output.\n- Target semantics match slow river enumeration exactly.\n- Dataset metadata supports stratified validation by SPR, pot, stack, raise depth, boundary ordinal, all-in proximity, and source distribution.\n- Contract docs identify raw leaf conversion used by GPU turn datagen.



Completed initial contract/schema milestone: model I/O contract doc is committed, manifest schema is implemented for Rust/Python, and compatibility tests pass.
