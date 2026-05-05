---
# poker_solver_rust-fp06
title: Turn-boundary CFVNet for fast GPU turn datagen
status: todo
type: epic
priority: high
created_at: 2026-05-05T02:54:12Z
updated_at: 2026-05-05T02:54:12Z
---

Build a first-class turn-boundary CFVNet that predicts river-averaged boundary CFVs directly from 4-card turn states, replacing online 48-river enumeration in GPU turn datagen while preserving exploitability validation quality.\n\n## Goal\n\nReduce GPU turn datagen boundary evaluation from batch * boundaries * players * rivers rows to batch * boundaries * players rows. Keep the current correctness level by using the existing slow river-enumeration path as an oracle for training targets, parity tests, and sampled production validation.\n\n## Acceptance\n\n- Turn-boundary model contract is documented and implemented.\n- Dataset generator can produce stratified oracle targets with metadata for raise depth, SPR, pot, stack, and source distribution.\n- Training and evaluation pipelines produce a model that passes stratified offline validation.\n- GPU turn datagen can use the turn-boundary model instead of online river enumeration.\n- Production validation compares sampled batches against the slow oracle and preserves the exploitability gate.
