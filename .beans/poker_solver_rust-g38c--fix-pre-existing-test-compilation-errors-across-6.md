---
# poker_solver_rust-g38c
title: Fix pre-existing test compilation errors across 6 crates
status: completed
type: bug
priority: normal
created_at: 2026-04-02T01:28:08Z
updated_at: 2026-04-02T01:43:01Z
---

Recent commits added allow_preflop_limp field to GameConfig, turn_output to DatagenConfig, and changed target_exploitability to Option<f32>. Tests in core, cfvnet, rebel, tauri-app, trainer, and range-solver haven't been updated.


## Summary of Changes

Fixed all pre-existing test compilation errors and one runtime bug:

1. **cfvnet/sampler.rs**: Added missing `turn_output: None` to DatagenConfig test struct
2. **cfvnet/turn_generate.rs**: Wrapped 4 bare `target_exploitability: 0.05` in `Some()`, fixed per_file storage rotation bug (check-before-write to prevent empty trailing files)
3. **cfvnet/integration_test.rs**: Updated bet_sizes type from Vec<String> to BetSizeConfig, wrapped target_exploitability in Some()
4. **core/blueprint_v2_e2e.rs**: Added missing `allow_preflop_limp: true` to 2 GameConfig test structs
5. **tauri-app/postflop.rs**: Fixed offsuit matrix label test expectations (AKo not KAo)

All 1667 tests pass, 0 failures, under 53s total.
