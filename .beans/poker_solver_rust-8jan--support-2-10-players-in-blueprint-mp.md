---
# poker_solver_rust-8jan
title: Support 2-10 players in blueprint_mp
status: todo
type: task
priority: normal
created_at: 2026-06-10T19:05:58Z
updated_at: 2026-06-10T19:05:58Z
parent: poker_solver_rust-osss
---

Widen the MP game model from the current 2-8 cap to 2-10 players per the unified-trainer goal. Known touch points (verified 2026-06-10): MAX_PLAYERS fixed-size arrays in blueprint_mp (lazy_mccfr.rs stacks/street_bets/contributions, exploitability.rs bucket arrays), config validation num_players 2..=8 in blueprint_mp/config.rs:49, and docs/blueprint_format.md game metadata range. MpInfosetKey seat (u8) and 4-bit per-node action packing are player-count independent. Verify TUI grids, probe scenarios, and any seat-indexed assumptions.
