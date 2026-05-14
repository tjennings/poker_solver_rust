---
# poker_solver_rust-kou7
title: Implement actual sampled river-spot datagen
status: completed
type: feature
priority: high
created_at: 2026-05-14T05:34:45Z
updated_at: 2026-05-14T05:43:31Z
---

Generate CFVNet river training records from actual sampled blueprint/postflop river spots rather than random boards with preflop-only ranges. Include a reasonable river action set with explicit all-in coverage and tests/docs for the new datagen mode.\n\nChecklist:\n- [x] Trace current blueprint/range datagen APIs and choose integration point\n- [x] Add river spot source that samples concrete reached river spots with line-conditioned ranges\n- [x] Ensure generated river subgames use a reasonable action set including all-in\n- [x] Add config/sample docs for the new mode\n- [x] Add focused tests or smoke coverage\n- [x] Run targeted verification

## Summary of Changes

Implemented sampled river-spot datagen for CFVNet. Added `datagen.sampled_river_spots` and `datagen.blueprint_bundle_path`; when enabled for river datagen, the situation generator loads a full blueprint bundle, samples a concrete river board, walks the blueprint strategy through preflop/flop/turn, tracks reached pot/effective stack, normalizes board-filtered line-conditioned OOP/IP ranges, and feeds the existing river solver/writer. Added a sampled-river config with explicit all-in rows, updated training/architecture docs, and verified with focused tests plus a one-sample real blueprint smoke generation.
