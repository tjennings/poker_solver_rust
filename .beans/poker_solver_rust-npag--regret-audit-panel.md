---
# poker_solver_rust-npag
title: Regret Audit Panel
status: in-progress
type: feature
created_at: 2026-03-28T03:41:34Z
updated_at: 2026-03-28T03:41:34Z
---

Add TUI panel for per-hand regret auditing at specific spots during blueprint training. Design: docs/plans/2026-03-27-regret-audit-panel-design.md. Plan: docs/plans/2026-03-27-regret-audit-panel.md.

## Tasks
- [ ] Task 1: Config — RegretAuditConfig struct and YAML parsing
- [ ] Task 2: Audit resolution — resolve hand+spot to regret coordinates
- [ ] Task 3: Metrics bridge — audit snapshot exchange
- [ ] Task 4: TUI rendering — audit panel widget
- [ ] Task 5: Layout integration — horizontal split
- [ ] Task 6: Wiring — config → resolution → trainer → TUI
- [ ] Task 7: Trainer callback — on_audit_refresh
- [ ] Task 8: Sample config update
