---
# poker_solver_rust-7ajt
title: Investigate subgame nut-fold all-in spot
status: completed
type: bug
priority: critical
created_at: 2026-05-05T18:23:04Z
updated_at: 2026-05-05T19:49:38Z
---

At spot sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Js|bb:check,sb:24bb,bb:all-in, the subgame appears to make BB jam 100% because SB folds every hand, including KK.

## TODOs

- [x] Reproduce the exact spot in a backend test or diagnostic.
- [x] Inspect the range-solver root configuration, player/seat mapping, and terminal payoff orientation at the BB all-in node.
- [x] Determine whether the bad strategy comes from the solver, boundary/payoff setup, or matrix extraction.
- [x] Fix the underlying bug or document the narrowed root cause if it is outside this patch.
- [x] Add regression coverage for SB not folding nut hands to a jam when call is clearly profitable.
- [x] Run targeted and full verification.
- [x] Merge to local main.

## Findings

- The comparison tool was rebuilding a fresh street root instead of the navigated solve root, which made prior diagnostics misleading.
- With root-aware comparison, the all-exact post-jam node has strong SB hands calling 100%, including AA/KK/JJ.
- The pathological overjam reproduces one decision earlier with `--river-boundary exact_subtree`: exact calls AA/KK/JJ, while the depth-limited exact-subtree solve jams them.
- `--river-boundary exact_oracle` remained close to exact in the same spot, so the next target is the exact-subtree boundary CFV contract/scale.

## Fix Notes

- Depth-boundary evaluation now uses the action tree's actual remaining stack instead of deriving stack from `effective_stack - pot / 2`.
- Exact-subtree treats zero-stack boundaries as all-in runouts and returns raw per-combination showdown CFVs at the parent solver's scale.
- The 10-iteration CLI reproduction now keeps AA/KK/KJs/KJo/JJ/AKs/AKo on call instead of all-in.

## Verification

- `cargo test -p range-solver test_boundary_remaining_stack_tracks_in_street_call`
- `cargo test -p poker-solver-tauri exact_subtree --no-run`
- `cargo run -p poker-solver-trainer --release -- compare-solve --bundle /Users/coreco/code/poker_solver_rust/local_data/blueprints/1k_100bb_brdcfr_v2 --snapshot snapshot_0013 --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|Ks8d3c|bb:check,sb:15bb,bb:call|Js|bb:check,sb:24bb' --exact-iters 50 --subgame-iters 10 --river-boundary exact_subtree --tolerance 0`
- Warm `cargo test`: passed in 50.70s.

## Summary of Changes

Merged the root-aware compare-solve diagnostics and exact-subtree boundary stack fixes to local main.
