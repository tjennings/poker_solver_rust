---
# poker_solver_rust-ja8p
title: Support Universal MP lazy turn and river navigation
status: completed
type: feature
priority: high
created_at: 2026-07-28T19:32:03Z
updated_at: 2026-07-28T21:25:56Z
parent: poker_solver_rust-mk2k
---

Extend the UniversalMpLazy GameExplorer beyond the flop boundary.

- Support selecting a legal turn card, loading the turn bucket source/rows from configured training.cluster_path or bundle-local sources, and rendering the turn strategy state.
- Support river card selection and river strategy navigation using the same sparse arena/tree model.
- Preserve card removal, chance transitions, action history, exact active-root state, and stale solve/cache anchors across streets.
- Keep missing bucket/row errors precise and state immutable on failure.
- Add focused Tauri/core regression coverage for flop-to-turn-to-river and unsupported/missing-source cases.
- Update Explorer docs and retain HU/eager/legacy behavior.


## Summary of Changes

UniversalMpLazy GameExplorer now supports legal flop-to-turn-to-river navigation and backtracking with configured street bucket lookup, full-board blocker filtering, root-reach-weighted strategy matrices, precise missing-source/row errors, and state-preserving failure handling. Exact-solve generation invalidation now covers action/street navigation, same-street cache rewind, successful saved-spot loads, and malformed/partial saved-spot replay. Explorer and architecture documentation describe the actual two-player browser and flop-only exact-solve boundaries.

## Verification

`cargo test -p poker-solver-tauri --test universal_explorer_integration -- --test-threads=1`: 27 passed. Focused core lazy-MP tests: 27 passed. Rustfmt and git diff --check passed. The full workspace suite was not run because it exceeds the project time target and contains unrelated long-running filtered binaries. The user's sample YAML remains untouched and unstaged.
