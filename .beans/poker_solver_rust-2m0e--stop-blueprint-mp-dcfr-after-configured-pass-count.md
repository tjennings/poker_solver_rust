---
# poker_solver_rust-2m0e
title: Stop blueprint_mp DCFR after configured pass count
status: in-progress
type: feature
priority: high
created_at: 2026-08-05T16:40:22Z
updated_at: 2026-08-05T16:44:11Z
---

Implement a Pluribus-style maximum discount-pass rule for blueprint_mp.

- [x] Verify algorithm semantics and configuration design
- [x] Establish focused baseline under runtime waiver
- [ ] Implement optional maximum executed discount passes
- [ ] Configure active HU sample for 40 passes
- [ ] Add deterministic eager/lazy/legacy compatibility tests
- [ ] Update training and architecture documentation
- [ ] Complete independent review and repairs
- [ ] Run focused and full correctness tests
- [ ] Integrate into main

The limit must stop future discount scans and lazy purge operations entirely; it must not merely cap the factor epoch.

## Focused Baseline

Existing discount scheduler baseline passed 57/57 focused core tests. The prior explicit full-suite runtime waiver is carried forward for this adjacent scheduler feature.

## Approved Design

Add MP-only `dcfr_discount_max_passes: Option<NonZeroU64>`, default unlimited, and set the active HU sample to 40. Count only successfully completed storage discount sweeps, independently from factor epoch; warmup and skipped wall-clock/iteration boundaries consume zero. The Nth pass and its lazy purge execute, while N+1 and every later scan/purge are suppressed. The counter is process-local until MP resume state exists. This is Pluribus-inspired rather than mathematically identical because MP uses post-warmup DCFR, not training-start LCFR.
