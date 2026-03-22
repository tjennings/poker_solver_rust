---
# poker_solver_rust-3s09
title: Fix test_dispatch_subgame_for_wide_flop
status: todo
type: bug
created_at: 2026-03-22T16:18:10Z
updated_at: 2026-03-22T16:18:10Z
---

The test test_dispatch_subgame_for_wide_flop in crates/tauri-app/src/postflop.rs fails with 'solve should complete' and takes ~100s due to equity matrix enumeration. Currently marked #[ignore]. Needs investigation — likely related to recent range_clamp_threshold parameter addition.
