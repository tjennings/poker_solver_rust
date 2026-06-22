---
# poker_solver_rust-8jan
title: Support 2-10 players in blueprint_mp
status: completed
type: task
priority: normal
created_at: 2026-06-10T19:05:58Z
updated_at: 2026-06-22T20:41:42Z
parent: poker_solver_rust-osss
---

Widen the MP game model from the current 2-8 cap to 2-10 players per the unified-trainer goal. Known touch points (verified 2026-06-10): MAX_PLAYERS fixed-size arrays in blueprint_mp (lazy_mccfr.rs stacks/street_bets/contributions, exploitability.rs bucket arrays), config validation num_players 2..=8 in blueprint_mp/config.rs:49, and docs/blueprint_format.md game metadata range. MpInfosetKey seat (u8) and 4-bit per-node action packing are player-count independent. Verify TUI grids, probe scenarios, and any seat-indexed assumptions.


## 2026-06-22 Start Notes

Activated after resolving the full-suite timing gate. Scope for this slice:

- [ ] Research all fixed 2-8 / MAX_PLAYERS assumptions in blueprint_mp, trainer, TUI, universal export, and docs.
- [x] Plan the minimal player-count widening that preserves the lazy sparse arena tree and avoids unrelated trainer/TUI rewrites.
- [x] Dispatch implementation for the Rust changes under manager-mode rules.
- [ ] Verify focused tests plus the full quiet workspace suite under the one-minute gate.
- [x] Update docs affected by player-count metadata/config limits.


## Research / Brainstorming Notes

Local scan found the required slice is mostly bounded, but not just a config cap:

- `crates/core/src/blueprint_mp/mod.rs` defines `MAX_PLAYERS = 8`; fixed arrays follow that constant and should widen cleanly.
- `crates/core/src/blueprint_mp/types.rs` stores `PlayerSet` in `u8`; this must become a wider integer before 9-10 players are valid. The `all`, `iter`, `bits`, and `from_bits` helpers need updated tests.
- `types.rs` position tables stop at 8; 9/10-max labels must be added and round-trip tested.
- `crates/core/src/blueprint_mp/config.rs` rejects `num_players > 8` and has tests that should become accept-10/reject-11.
- `crates/core/src/blueprint_mp/mccfr.rs` deal tests iterate `2..=8`; they should include 10 to prove deal sampling/card uniqueness at the new cap.
- `crates/trainer/src/mp_tui_scenarios.rs` position parsing already reaches `utg2` for 8-max; add labels for 9/10-max scenario notation if core labels introduce them.
- `crates/core/src/blueprint_mp/game_tree.rs` module docs mention 2-8; update to 2-10 and add a lightweight 10-player lazy/eager construction smoke if existing helpers can keep the tree tiny.
- `docs/blueprint_format.md` already says `num_players` is 2 through 10; verify no stale 8-max text remains.

Minimal plan: preserve lazy sparse/arena traversal semantics, widen representation and validation only, add focused 10-player tests, and avoid unrelated trainer/TUI consolidation work.

## 2026-06-22 Implementation Notes

Implemented the focused 2-10 player widening for blueprint_mp: MAX_PLAYERS is now 10, PlayerSet uses a compact u16 bitset so seats 8 and 9 are represented, core position labels now round-trip through 10-max, config validation accepts 10 and rejects 11, and deal tests sample 2..=10. Added a lazy 10-player root construction/action-generation smoke without changing lazy action-history packing. docs/blueprint_format.md already stated num_players 2 through 10; a stale 2-8 scan over the scoped files found no remaining stale text.

Verification run in this attempt:

- cargo test -p poker-solver-core --lib blueprint_mp::types -- --nocapture
- cargo test -p poker-solver-core --lib blueprint_mp::config -- --nocapture
- cargo test -p poker-solver-core --lib blueprint_mp::mccfr::tests::deal -- --nocapture
- cargo test -p poker-solver-core --lib blueprint_mp::lazy_mccfr::tests::lazy_10_player_root_generates_actions_without_building_tree -- --nocapture
- cargo test -p poker-solver-trainer mp_tui_scenarios::tests::position_to_seat_10max_extended_utg_labels -- --nocapture

Not run: full workspace suite in this implementation pass.


## Summary of Changes

- Widened blueprint_mp from 2-8 to 2-10 players by setting `MAX_PLAYERS = 10` and moving `PlayerSet` from `u8` to `u16`, so seats 8 and 9 are represented correctly in active-player sets.
- Updated MP config validation, deal tests, position labels, TUI scenario position parsing, and lazy root action-generation coverage for 10-player games.
- Made affected terminal/game-tree/trainer test fixtures `MAX_PLAYERS`-neutral so fixed-size arrays continue compiling at the wider cap.
- Updated architecture and training docs from 2-8 to 2-10 where they describe the MP player-count limit.

Verification:

- `cargo test -p poker-solver-core --lib blueprint_mp::types -- --nocapture` passed.
- `cargo test -p poker-solver-core --lib blueprint_mp -- --nocapture` passed: 313 passed, 0 failed.
- `cargo test -p poker-solver-trainer mp_tui_scenarios::tests::position_to_seat_10max_extended_utg_labels -- --nocapture` passed.
- `/usr/bin/time -p cargo test --workspace --quiet` passed after rebuild with `real 880.44`, `user 160.18`, `sys 107.51`.
- Warm `/usr/bin/time -p cargo test --workspace --quiet` passed under the gate with `real 51.69`, `user 97.52`, `sys 17.56`.

Formatting note: full `cargo fmt --check` is not currently a scoped signal because current rustfmt wants to rewrite many pre-existing files outside this slice, including universal-format modules. Those broad formatting changes were not applied.
