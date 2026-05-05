---
# poker_solver_rust-oh3o
title: Define turn-boundary dataset metadata schema
status: completed
type: task
priority: high
created_at: 2026-05-05T02:54:42Z
updated_at: 2026-05-05T03:08:01Z
parent: poker_solver_rust-ewjj
---

Define metadata fields for every turn-boundary training row: pot, stack, SPR, boundary ordinal, canonical node id, raise depth, all-in proximity, board, source distribution, oracle source, and validation stratum labels.



Completed with Rust and Python turn-boundary manifest schemas. Rust owns DatasetManifest validation/read/write in crates/cfvnet/src/datagen/manifest.rs; Python can read/write the same YAML schema in crates/cfvnet/python/cfvnet/manifest.py.
