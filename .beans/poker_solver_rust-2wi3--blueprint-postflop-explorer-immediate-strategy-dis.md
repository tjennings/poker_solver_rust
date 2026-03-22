---
# poker_solver_rust-2wi3
title: Blueprint postflop explorer — immediate strategy display
status: completed
type: feature
priority: normal
created_at: 2026-03-22T18:07:33Z
updated_at: 2026-03-22T19:20:54Z
---

Blueprint postflop explorer — immediate strategy display after flop deal. Backend: real postflop bucket lookups in get_strategy_matrix_v2. Frontend: blueprint mode with navigation, SOLVE transition, green Solved indicator, street progression.

## Summary of Changes

### Backend (crates/tauri-app/src/exploration.rs)
- Threaded CbvContext (with AllBuckets) into get_strategy_matrix_v2
- Replaced uniform distribution postflop stub with real bucket lookup + strategy query
- Extended reaching probability computation for postflop nodes
- Added to_act field to StrategyMatrix
- Created BucketLookup struct and bucket_probs_for_hand helper to deduplicate combo enumeration
- Fixed board slice mismatch bug in reaching probability replay

### Frontend (frontend/src/PostflopExplorer.tsx)
- Added blueprint mode: fetches and displays blueprint strategy immediately after flop deal
- Blueprint tree navigation (forward/back) using get_strategy_matrix
- SOLVE transition: switches from blueprint to solver with live updates
- Green SOLVED button indicator when solve completes
- Street progression with blueprint range propagation
- Extracted helpers: makeBlueprintPosition, fetchBlueprintMatrix, pushActionCard
- Removed dead state variables (cache, loading)

### Branch: feat/blueprint-postflop-explorer (8 commits)
