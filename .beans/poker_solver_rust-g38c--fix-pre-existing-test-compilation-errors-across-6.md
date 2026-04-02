---
# poker_solver_rust-g38c
title: Fix pre-existing test compilation errors across 6 crates
status: in-progress
type: bug
created_at: 2026-04-02T01:28:08Z
updated_at: 2026-04-02T01:28:08Z
---

Recent commits added allow_preflop_limp field to GameConfig, turn_output to DatagenConfig, and changed target_exploitability to Option<f32>. Tests in core, cfvnet, rebel, tauri-app, trainer, and range-solver haven't been updated.
