---
# poker_solver_rust-aw8n
title: Fix MP strategy sum overflow in long training
status: in-progress
type: bug
priority: high
created_at: 2026-05-06T14:33:55Z
updated_at: 2026-05-06T14:33:55Z
---

Average-strategy storage for blueprint_mp uses AtomicI32 strategy_sums with plain fetch_add. Long 6-max runs can wrap an individual strategy counter and corrupt TUI/explorer average strategy after enough visits.\n\nChecklist:\n- [ ] Convert MP strategy_sums to non-wrapping wider atomic storage\n- [ ] Update DCFR discount and average_strategy reads for the wider type\n- [ ] Add regression coverage for values above i32::MAX\n- [ ] Update docs if storage/memory semantics change\n- [ ] Run focused and full test suites
