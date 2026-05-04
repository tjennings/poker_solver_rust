---
# poker_solver_rust-i5rc
title: Check Tauri wiring for oracle boundary compare-solve changes
status: completed
type: task
priority: normal
created_at: 2026-05-04T04:50:36Z
updated_at: 2026-05-04T04:56:12Z
parent: poker_solver_rust-e90m
---

Verify whether the recent compare-solve oracle-boundary diagnostic and finalization changes need any Tauri/devserver/frontend wiring, and record any missing hooks.

Checklist:

[x] Inspect Tauri command surface and devserver API for compare-solve or related solver diagnostics.
[x] Inspect frontend invoke/API usage for compare-solve, oracle-boundary, exact_subtree, or root trace controls.
[x] Determine whether the recent changes are CLI-only/internal or require Explorer UI updates.
[x] Run the narrowest relevant build/type checks if frontend or Tauri wiring is touched.

## Summary of Changes

Verified compare-solve itself is trainer CLI-only, while the Tauri Explorer uses `game_solve` with `streetBoundaryConfig` for `exact`, `cfvnet`, and `exact_subtree` modes. The devserver mirrors `game_solve` and forwards the same boundary, trace, and gadget parameters to `game_solve_core`.

Found one missing behavioral hook: Tauri Explorer finalization with per-boundary evaluators did not clear evaluator-backed boundary CFV caches before `finalize`, so it could retain the same stale-finalization behavior fixed in compare-solve. Added the same `clear_boundary_cfvs()` guard before finalization in `game_solve_core`, which covers native Tauri and devserver/browser mode.

Frontend controls were already sending `exact_subtree` boundary mode through `buildSolveParams`; no UI control change was needed. While validating the frontend build, fixed the stale TypeScript `ComboDetail` shape to include the backend-provided optional `bucket` field.

Validation: `npm --prefix frontend run build` passed; focused Tauri and devserver tests passed; full warm `cargo test` passed in 52.70s.
