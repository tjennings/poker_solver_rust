---
# poker_solver_rust-8jan
title: Support 2-10 players in blueprint_mp
status: in-progress
type: task
priority: normal
created_at: 2026-06-10T19:05:58Z
updated_at: 2026-06-22T20:07:36Z
parent: poker_solver_rust-osss
---

Widen the MP game model from the current 2-8 cap to 2-10 players per the unified-trainer goal. Known touch points (verified 2026-06-10): MAX_PLAYERS fixed-size arrays in blueprint_mp (lazy_mccfr.rs stacks/street_bets/contributions, exploitability.rs bucket arrays), config validation num_players 2..=8 in blueprint_mp/config.rs:49, and docs/blueprint_format.md game metadata range. MpInfosetKey seat (u8) and 4-bit per-node action packing are player-count independent. Verify TUI grids, probe scenarios, and any seat-indexed assumptions.


## 2026-06-22 Start Notes

Activated after resolving the full-suite timing gate. Scope for this slice:

- [ ] Research all fixed 2-8 / MAX_PLAYERS assumptions in blueprint_mp, trainer, TUI, universal export, and docs.
- [ ] Plan the minimal player-count widening that preserves the lazy sparse arena tree and avoids unrelated trainer/TUI rewrites.
- [ ] Dispatch implementation for the Rust changes under manager-mode rules.
- [ ] Verify focused tests plus the full quiet workspace suite under the one-minute gate.
- [ ] Update docs affected by player-count metadata/config limits.
