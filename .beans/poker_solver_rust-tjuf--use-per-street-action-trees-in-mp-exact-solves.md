---
# poker_solver_rust-tjuf
title: Use per-street action trees in MP exact solves
status: in-progress
type: bug
priority: high
created_at: 2026-07-29T13:53:54Z
updated_at: 2026-07-29T13:53:54Z
parent: poker_solver_rust-g7yj
---

A UniversalMpLazy exact solve rooted on Turn currently installs the captured Turn action sizes for Flop, Turn, and River. Preserve the configured per-street action abstractions and all depth rows through the full exact tree; add a downstream river-action regression with distinct turn/river sizing.
