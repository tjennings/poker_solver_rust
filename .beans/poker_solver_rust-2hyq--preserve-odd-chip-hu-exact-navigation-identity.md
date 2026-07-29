---
# poker_solver_rust-2hyq
title: Preserve odd-chip HU exact navigation identity
status: in-progress
type: bug
priority: high
created_at: 2026-07-29T14:48:50Z
updated_at: 2026-07-29T14:48:50Z
parent: poker_solver_rust-g7yj
---

The exact action identity hardening regressed HU navigation for odd-chip/fractional actions because HU history still stores rounded action labels while solved paths retain exact amounts. Preserve raw HU action identity or match consistently at the same representation.
