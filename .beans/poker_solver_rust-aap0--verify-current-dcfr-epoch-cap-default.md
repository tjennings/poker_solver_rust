---
# poker_solver_rust-aap0
title: Verify current dcfr_epoch_cap default
status: completed
type: task
priority: normal
created_at: 2026-08-05T13:50:20Z
updated_at: 2026-08-05T13:53:04Z
---

Inspect current main and report the effective default for dcfr_epoch_cap, including whether HU blueprint-v2 and MP blueprint differ.

- [x] Verify config type and default
- [x] Verify runtime interpretation
- [x] Report exact current behavior with code references

## Summary of Changes

Verified that blueprint_v2 defaults dcfr_epoch_cap to None (uncapped), with select sample configurations overriding it to 40 or 80. Confirmed the cap plateaus the factor epoch while discount passes continue. Verified blueprint_mp exposes no epoch-cap field and uses an uncapped epoch; an MP YAML key would currently be ignored.
