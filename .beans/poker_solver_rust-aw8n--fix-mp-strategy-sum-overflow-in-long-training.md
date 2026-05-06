---
# poker_solver_rust-aw8n
title: Fix MP strategy sum overflow in long training
status: completed
type: bug
priority: high
created_at: 2026-05-06T14:33:55Z
updated_at: 2026-05-06T14:43:43Z
---

Average-strategy storage for blueprint_mp used AtomicI32 strategy_sums with plain fetch_add. Long 6-max runs could wrap an individual strategy counter and corrupt TUI/explorer average strategy after enough visits.

Checklist:
- [x] Convert MP strategy_sums to non-wrapping wider atomic storage
- [x] Update DCFR discount and average_strategy reads for the wider type
- [x] Add regression coverage for values above i32::MAX
- [x] Update docs if storage/memory semantics change
- [x] Run focused and full test suites

## Summary of Changes

- Converted blueprint_mp average-strategy sums to saturating AtomicU64 storage while keeping regrets on AtomicI32.
- Updated memory reporting, average-strategy reads, and DCFR strategy-sum discounting for u64 counters.
- Added regression tests for sums above i32::MAX and saturation at u64::MAX.
- Updated architecture/training docs to describe MP strategy-sum storage.
