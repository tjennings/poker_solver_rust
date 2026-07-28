---
# poker_solver_rust-ja8p
title: Support Universal MP lazy turn and river navigation
status: in-progress
type: feature
priority: high
created_at: 2026-07-28T19:32:03Z
updated_at: 2026-07-28T19:32:03Z
parent: poker_solver_rust-mk2k
---

Extend the UniversalMpLazy GameExplorer beyond the flop boundary.

- Support selecting a legal turn card, loading the turn bucket source/rows from configured training.cluster_path or bundle-local sources, and rendering the turn strategy state.
- Support river card selection and river strategy navigation using the same sparse arena/tree model.
- Preserve card removal, chance transitions, action history, exact active-root state, and stale solve/cache anchors across streets.
- Keep missing bucket/row errors precise and state immutable on failure.
- Add focused Tauri/core regression coverage for flop-to-turn-to-river and unsupported/missing-source cases.
- Update Explorer docs and retain HU/eager/legacy behavior.
