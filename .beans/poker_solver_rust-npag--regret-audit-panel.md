---
# poker_solver_rust-npag
title: Regret Audit Panel
status: completed
type: feature
priority: normal
created_at: 2026-03-28T03:41:34Z
updated_at: 2026-03-28T04:41:14Z
---

Add TUI panel for per-hand regret auditing at specific spots during blueprint training.

## Tasks
- [x] Task 1: Config — RegretAuditConfig struct and YAML parsing
- [x] Task 2: Audit resolution — resolve hand+spot to regret coordinates
- [x] Task 3: Metrics bridge — audit snapshot exchange
- [x] Task 4: TUI rendering — audit panel widget
- [x] Task 5: Layout integration — horizontal split
- [x] Task 6: Wiring — config → resolution → trainer → TUI
- [x] Task 7: Trainer callback — on_audit_refresh
- [x] Task 8: Sample config update

## Summary of Changes
- New YAML config section `regret_audits` under `tui:` with name, spot, hand, player fields
- Polled-snapshot approach reads `AtomicI32` regrets from `BlueprintStorage` on each TUI tick
- Horizontal split layout: sparklines 60% left, audit panel 40% right (unchanged when no audits configured)
- Tabbed audit panel with bucket trail, per-action regret/delta/trend table, derived strategy
- Up/Down arrows navigate audit tabs
- Zero MCCFR hot-path overhead
