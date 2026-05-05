---
# poker_solver_rust-bl1l
title: Fix solved subgame child matrix fallback
status: done
type: bug
priority: high
created_at: 2026-05-05T14:28:35Z
updated_at: 2026-05-05T14:51:00Z
---

After solving at the root of a street, clicking a solved action such as Check leaves the Subgame tab marked solved but displays a default/blueprint-looking matrix at the child node instead of the solved child strategy.

## User Report

Screenshot 2026-05-05 09:27 shows Subgame [solve] selected after root-street solve and Check action. The action cards use BB/SB labels correctly, but the matrix appears as the default representative matrix rather than the solved subgame child matrix.

## TODOs

[x] Reproduce with backend cache/path tests around root-street solve -> check child.
[x] Find why solved child cache lookup misses or returns a default matrix.
[x] Fix Subgame/Exact source matrix selection so solved child nodes render solved matrices.
[x] Add regression coverage for root-street solve then Check navigation.
[x] Run targeted and full verification before merge to local main.

## Result

The source-aware action fallback now overlays the cached solved node after resolving the session path, so Subgame/Exact tabs keep showing the solved representative matrix when the UI action id falls back to the session action id. Added regression coverage for the root-solve -> action child matrix path.

## Verification

- `cargo test -p poker-solver-tauri source_play_overlays_cached_matrix_after_session_action_id_fallback`
- `cargo test -p poker-solver-tauri game_session::tests::source_play`
- `cargo test -p poker-solver-tauri game_session::tests::`
- `/usr/bin/time -p cargo test` passed in 53.30s
