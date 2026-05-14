---
# poker_solver_rust-n9vm
title: Fix MP limp flag and pruning passivity
status: completed
type: bug
priority: high
created_at: 2026-05-14T19:55:40Z
updated_at: 2026-05-14T20:05:39Z
---

MP lazy training is improving but strategies look overly passive after protecting passive actions from pruning. Also game.allow_preflop_limp is present in the 6-max config but ignored by MP config/action generation. Implement MP allow_preflop_limp for eager/lazy action generation and adjust lazy pruning so it does not structurally bias toward passive actions while avoiding the earlier subtree wipeout.

## Summary of Changes

- Added game.allow_preflop_limp to MpGameConfig with a backwards-compatible default of true.
- Threaded the limp flag through eager and lazy MP action generation so unopened preflop calls are omitted when disabled while folds and open sizes remain available.
- Restored normal lazy traversal pruning to the eager-compatible rule: protect immediate terminal children, but allow nonterminal branches below prune_threshold to be skipped. Persistent subtree purge remains aggressive-edge-only.
- Updated docs and targeted tests for no-limp parsing/action generation and lazy pruning semantics.

## Validation

- cargo fmt --check
- cargo test -p poker-solver-core no_limp -- --nocapture
- cargo test -p poker-solver-core lazy_pruning -- --nocapture
- cargo test -p poker-solver-core deserialize_heads_up_config -- --nocapture
- cargo test -p poker-solver-core lead_raise_split_config -- --nocapture
- cargo test -p poker-solver-core negative_action -- --nocapture
- git diff --check
