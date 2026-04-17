---
# poker_solver_rust-eba0
title: Fix subgame range + actor label mismatch in Explorer
status: todo
type: bug
priority: high
created_at: 2026-04-17T17:05:22Z
updated_at: 2026-04-17T17:05:22Z
---

Subgame solver's range matrix disagrees with Blueprint and Exact tabs at the same decision node, and the actor label is wrong.

## Repro

Action path in explorer: SB open 2bb → BB 3bet 10bb → SB call → flop JJT → BB check → SB check → turn Q → BB bets 20bb → SB to act.

- **Blueprint tab**: actor labeled `SB` (correct — SB is IP postflop HU, and it's SB's turn).
- **Subgame tab**: actor labeled `OOP` (wrong — SB is IP, not OOP). **Range data also differs from Blueprint and Exact.**
- **Exact tab**: actor labeled `IP` (correct positionally, but inconsistent vocabulary with Blueprint's `SB`).

User confirmed the subgame range *disagrees* with the other two tabs — not just a label bug.

## Root-cause investigation (pre-work, not confirmed end-to-end)

Explore agent traced the label bug to:

**`crates/tauri-app/src/game_session.rs:1323`**
```rust
let position = if player == 0 { "OOP" } else { "IP" };
```

Inside `build_solve_cache_recursive()`, called from line ~2058 after a subgame solve completes. It hardcodes "OOP"/"IP" based on the range_solver library's internal player numbering (0/1), without consulting the V2 game tree's dealer/position mapping.

Blueprint's correct label comes from `position_label()` at lines 297–303, which uses V2 dealer info.

**Open question (data bug, not yet root-caused):** whether range_solver player 0 = OOP always holds for our subgame builds. The game is constructed at lines 1115–1126 with `range: [oop_range, ip_range]`, so if the *wrong* range is passed in either slot (e.g. hero/villain swapped for this SB-IP node), the cached matrix would show villain's range under hero's label. The user's observation that the matrix data itself differs from Blueprint strongly suggests this is a data bug, not just a label bug.

## TODOs

- [ ] Reproduce at the exact node (SB 2bb → BB 10bb → SB call → JJT → bx/bx → Qh → BB 20bb → SB) and capture Blueprint vs Subgame vs Exact range side-by-side
- [ ] Verify which slot (`oop_range` vs `ip_range`) gets hero vs villain at lines 1115–1126 for this specific node — is there an off-by-one or IP/OOP swap based on who acts first on the current street?
- [ ] Compare the matrix hands shown under Subgame to what the range-solver actually solved for — is the UI pulling from the wrong player index in `build_solve_cache_recursive()`?
- [ ] Confirm whether Exact (which also calls `build_solve_cache`) shows matching data to Blueprint — if yes, the bug is specific to the subgame *input* range construction, not the cache; if no, the cache itself is swapping.
- [ ] Fix the label: pass the initial position label ("SB"/"BB") and path depth through `build_solve_cache_recursive()` so it alternates correctly instead of hardcoding OOP/IP. Match Blueprint's vocabulary ("SB"/"BB") across all three tabs for consistency.
- [ ] Write a regression test: at a known node on the turn with SB to act, the Subgame tab must display the same hero range as Blueprint (within subgame-solve convergence tolerance).

## Files implicated

- `crates/tauri-app/src/game_session.rs` (label bug line 1323; range construction ~1115–1126; cache build ~2058; position_label ~297)
- Frontend range-matrix panel component (unknown path — find during repro)

## Why now

Subgame solving is the primary way the user validates blueprint strategy at specific nodes. If the Subgame tab shows a different range than Blueprint at the same node, every downstream check is unreliable.
