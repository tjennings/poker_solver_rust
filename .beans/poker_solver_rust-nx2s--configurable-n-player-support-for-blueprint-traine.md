---
# poker_solver_rust-nx2s
title: Configurable N-player support for blueprint trainer
status: completed
type: epic
priority: normal
created_at: 2026-04-08T05:06:01Z
updated_at: 2026-04-08T15:19:01Z
---

## Summary of Changes

Complete clean-room N-player (2-8) MCCFR blueprint trainer module (`blueprint_mp`) implemented alongside existing `blueprint_v2`.

### Module Structure (10 files)
- `types.rs` — Domain newtypes: Seat, PlayerSet, Chips, Bucket, Street, Deal, DealWithBuckets
- `config.rs` — BlueprintMpConfig with lead/raise split, ForcedBet blind structure
- `info_key.rs` — 128-bit InfoKey (seat + bucket + street + SPR + 22 actions)
- `game_tree.rs` — N-player game tree with fold-continuation, side pots
- `terminal.rs` — Side pot resolution, showdown, fold payoffs
- `storage.rs` — Flat-buffer atomic regret/strategy storage
- `mccfr.rs` — External-sampling MCCFR (Pluribus-style)
- `trainer.rs` — Training loop with per-seat traverser cycling, DCFR
- `exploitability.rs` — Per-seat best-response diagnostic
- CLI: `train-blueprint-mp` subcommand with sample 3p and 6p configs

### Test Coverage
- 152 unit tests + 6 integration tests = 158 total
- All functions under 60 lines (longest: 47)
- Cross-validated against blueprint_v2 on 2-player configs

### Review Findings (deferred cleanup)
- game_tree.rs has lead/raise duplication (5 pairs) — follow-up refactor
- resolve_showdown has 6 params — could group rake into struct
- Street enum triplicated across crate — pre-existing, not new debt
