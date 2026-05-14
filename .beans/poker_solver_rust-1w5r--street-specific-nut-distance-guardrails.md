---
# poker_solver_rust-1w5r
title: Street-specific nut-distance guardrails
status: in-progress
type: task
priority: normal
created_at: 2026-05-14T01:14:02Z
updated_at: 2026-05-14T06:14:59Z
parent: poker_solver_rust-03j0
---

Implement and validate street-specific guardrails: modest nut awareness on flop, stronger draw/nut separation on turn, strong nut hierarchy preservation on river.\n\nAcceptance: generated candidates reduce nut-distance span without destroying potential-consistency metrics.

## Metric-shape implementation plan\n\nBuild shape controls for the nut-distance channel so it acts as a guardrail rather than a second potential metric.\n\nAcceptance checklist:\n- [ ] Add config-compatible nut-distance cap and nonlinear transform controls with defaults preserving existing behavior.\n- [ ] Apply cap/transform only to the normalized nut-distance contribution in weighted child-bucket EMD gaps.\n- [ ] Add focused config and metric-combination tests.\n- [ ] Update training/architecture docs for the new controls.\n- [ ] Add stage-three candidate configs and runner for capped/nonlinear evaluation.\n- [ ] Run focused tests and smoke-check a diagnostic candidate command.
