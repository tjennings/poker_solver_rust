---
# poker_solver_rust-ekyc
title: Explain negative-regret DCFR discounting
status: completed
type: task
priority: normal
created_at: 2026-08-04T17:53:07Z
updated_at: 2026-08-04T17:59:32Z
---

Inspect the blueprint trainer implementation and explain whether discounting a negative cumulative regret such as -1 moves it to zero.

- [x] Locate the applicable DCFR formula and implementation
- [x] Work a concrete -1 example
- [x] Report the answer with confidence and code references

## Summary of Changes

Verified the standard beta=0 negative-regret factor and audited both HU blueprint_v2 and MP blueprint implementations. Documented the different fixed-point rounding behavior and the distinction between stored regret and regret-matching weight.
