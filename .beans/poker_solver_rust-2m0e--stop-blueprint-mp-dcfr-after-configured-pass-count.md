---
# poker_solver_rust-2m0e
title: Stop blueprint_mp DCFR after configured pass count
status: in-progress
type: feature
priority: high
created_at: 2026-08-05T16:40:22Z
updated_at: 2026-08-05T16:40:22Z
---

Implement a Pluribus-style maximum discount-pass rule for blueprint_mp.

- [ ] Verify algorithm semantics and configuration design
- [ ] Establish focused baseline under runtime waiver
- [ ] Implement optional maximum executed discount passes
- [ ] Configure active HU sample for 40 passes
- [ ] Add deterministic eager/lazy/legacy compatibility tests
- [ ] Update training and architecture documentation
- [ ] Complete independent review and repairs
- [ ] Run focused and full correctness tests
- [ ] Integrate into main

The limit must stop future discount scans and lazy purge operations entirely; it must not merely cap the factor epoch.
