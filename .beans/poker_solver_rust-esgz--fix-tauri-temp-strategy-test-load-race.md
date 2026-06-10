---
# poker_solver_rust-esgz
title: Fix Tauri temp strategy test load race
status: completed
type: bug
priority: normal
created_at: 2026-06-03T18:24:07Z
updated_at: 2026-06-03T18:26:11Z
---

Tauri default test bucket_probs_for_hand_all_blocked_returns_zero_count intermittently fails because build_minimal_postflop_strategy saves and reloads a temporary strategy path that can return NotFound.\n\n- [x] Inspect current filesystem roundtrip in exploration.rs tests\n- [x] Implement narrow reliability fix preserving coverage\n- [x] Verify targeted Tauri test\n- [x] Verify poker-solver-tauri lib tests if feasible

## Summary of Changes

Replaced the exploration test helper temp-file save/load strategy construction with direct serde construction followed by BlueprintV2Strategy::post_deserialize(), preserving serialized-shape coverage while removing the fragile shared temp path.
