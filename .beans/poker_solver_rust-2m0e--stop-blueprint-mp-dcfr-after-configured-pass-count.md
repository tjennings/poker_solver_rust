---
# poker_solver_rust-2m0e
title: Stop blueprint_mp DCFR after configured pass count
status: completed
type: feature
priority: high
created_at: 2026-08-05T16:40:22Z
updated_at: 2026-08-05T17:21:10Z
---

Implement a Pluribus-style maximum discount-pass rule for blueprint_mp.

- [x] Verify algorithm semantics and configuration design
- [x] Establish focused baseline under runtime waiver
- [x] Implement optional maximum executed discount passes
- [x] Configure active HU sample for 40 passes
- [x] Add deterministic eager/lazy/legacy compatibility tests
- [x] Update training and architecture documentation
- [x] Complete independent review and repairs
- [x] Run focused and full correctness tests
- [x] Integrate into main

The limit must stop future discount scans and lazy purge operations entirely; it must not merely cap the factor epoch.

## Focused Baseline

Existing discount scheduler baseline passed 57/57 focused core tests. The prior explicit full-suite runtime waiver is carried forward for this adjacent scheduler feature.

## Approved Design

Add MP-only `dcfr_discount_max_passes: Option<NonZeroU64>`, default unlimited, and set the active HU sample to 40. Count only successfully completed storage discount sweeps, independently from factor epoch; warmup and skipped wall-clock/iteration boundaries consume zero. The Nth pass and its lazy purge execute, while N+1 and every later scan/purge are suppressed. The counter is process-local until MP resume state exists. This is Pluribus-inspired rather than mathematically identical because MP uses post-warmup DCFR, not training-start LCFR.

## Implementation Progress

Added the process-local optional nonzero pass cap, scheduler accounting independent of factor epoch, eager/lazy cap enforcement, final-pass telemetry, a 40-pass HU sample setting, deterministic cap/purge regression coverage, and updated training/architecture documentation. Focused MP scheduler/config tests and trainer tests pass; workspace check passes. Full core lib reached 1,289 passing tests, with four unrelated failures because the isolated worktree lacks the local baseline JSON fixture.

## Review Repair

Independent review found one documentation overstatement: 40 ten-minute passes finish at 400 post-warmup scheduler minutes only when no slots are missed. The training guide now explains that skipped slots consume no completed pass and can delay the 40th actual sweep.

## Summary of Changes

Added `dcfr_discount_max_passes` as an optional nonzero blueprint_mp limit, configured the active HU sample for 40 completed discount sweeps, and stopped all later eager scans and lazy purge work after the cap. The completed-pass counter is independent of factor epoch; warmup and skipped scheduler slots consume no passes. Updated architecture/training documentation and qualified the 400-minute example for missed slots. Independent review approved the final implementation. Post-merge verification passed: formatting, 68 focused blueprint_mp trainer tests, 342 trainer tests with one CUDA-only ignore, workspace check, and the complete workspace test suite with exit code 0 under the approved runtime waiver.
