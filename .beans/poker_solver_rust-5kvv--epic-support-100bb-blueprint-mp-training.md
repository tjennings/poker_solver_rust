---
# poker_solver_rust-5kvv
title: 'Epic: support 100bb Blueprint MP training'
status: in-progress
type: epic
priority: high
created_at: 2026-05-07T13:18:26Z
updated_at: 2026-05-07T13:18:26Z
---

Blueprint MP must support normal 100bb 6-max training with multiple preflop raise depths. Current dense full-tree/full-storage architecture explodes to hundreds of millions of nodes and hundreds of GB of virtual storage.\n\nTarget: train 100bb 6-max configs without up-front dense allocation proportional to every public tree node/bucket/action.\n\nDesign direction:\n- Move from eager full-tree materialization toward lazy/state-based traversal or a compact betting-state graph.\n- Move from dense per-node storage toward sparse/visited infoset storage or compressed action-memory storage.\n- Keep diagnostics/snapshots compatible without full-storage scans.\n- Add 100bb regression/perf gates so this does not regress.
