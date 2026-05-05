---
# poker_solver_rust-bl1l
title: Fix solved subgame child matrix fallback
status: in-progress
type: bug
priority: high
created_at: 2026-05-05T14:28:35Z
updated_at: 2026-05-05T14:28:35Z
---

After solving at the root of a street, clicking a solved action such as Check leaves the Subgame tab marked solved but displays a default/blueprint-looking matrix at the child node instead of the solved child strategy.

## User Report

Screenshot 2026-05-05 09:27 shows Subgame [solve] selected after root-street solve and Check action. The action cards use BB/SB labels correctly, but the matrix appears as the default representative matrix rather than the solved subgame child matrix.

## TODOs

[ ] Reproduce with backend cache/path tests around root-street solve -> check child.
[ ] Find why solved child cache lookup misses or returns a default matrix.
[ ] Fix Subgame/Exact source matrix selection so solved child nodes render solved matrices.
[ ] Add regression coverage for root-street solve then Check navigation.
[ ] Run targeted and full verification before merge to local main.
