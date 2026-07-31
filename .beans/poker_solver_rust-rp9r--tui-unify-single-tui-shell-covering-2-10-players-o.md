---
# poker_solver_rust-rp9r
title: 'tui-unify: single TUI shell covering 2-10 players over the runtime seam'
status: todo
type: feature
priority: normal
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-18T18:43:36Z
parent: poker_solver_rust-osss
blocked_by:
    - poker_solver_rust-tzv5
    - poker_solver_rust-mt3l
    - poker_solver_rust-8jan
---

Collapse the three TUIs (HU blueprint_tui, MP eager, MP lazy) into one shell rendering 2-player and N-player(<=10) training over the shared runtime telemetry seam. A TuiScenarioBackend abstraction over the three scenario resolvers; shared CellStrategy/HandGridState/action_color; a MultiGridView (1..6 grids + pagination) plus the single 13x13 HU grid; unified metrics bridge; config-reload for all backends; lazy action labels via mt3l fingerprints (no per-cell game re-walk). EXPLICIT retirement decisions to ratify: HU-only regret audit panel, random scenario carousel, HU exploitability chart — port or sign off as dropped. If it exceeds one review, split into (a) scenario-backend+bridge, (b) renderer/grid unification, (c) feature-retirement ratification. Size: large. From Phase 7 plan 2026-06-18.
