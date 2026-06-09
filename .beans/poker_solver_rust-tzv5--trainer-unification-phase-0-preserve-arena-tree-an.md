---
# poker_solver_rust-tzv5
title: 'Trainer unification phase 0: preserve arena tree and establish shared runtime seam'
status: in-progress
type: feature
priority: high
created_at: 2026-06-09T14:55:16Z
updated_at: 2026-06-09T14:55:16Z
parent: poker_solver_rust-osss
---

First implementation slice for the HU/multiplayer trainer unification epic. Scope: research current HU arena/lazy tree and MP lazy traversal boundaries; produce a concrete shared runtime seam that preserves the new arena tree model; add golden/parity tests before behavior migration; identify how the HU TUI becomes the single TUI shell. Non-goal: do not rewrite full traversal or delete either trainer in this slice.
