---
# poker_solver_rust-3kn5
title: Ratify action-frequency parity as HU vs MP lazy gate
status: in-progress
type: feature
priority: high
created_at: 2026-06-23T15:36:34Z
updated_at: 2026-06-23T15:36:34Z
parent: poker_solver_rust-osss
---

Promote the new HU-vs-MP-lazy counterfactual action-frequency parity metric from diagnostic evidence to the primary GO/NO-GO criterion for the 2-player replacement gate. Calibrate it on a small representative 2p harness run, preserve structural/schema/coverage failures as hard NO-GO, treat unresolved average-strategy accounting and local row-L1 as diagnostics/warnings rather than automatic blockers when action-frequency parity passes, and emit clear artifacts showing frequencies, tolerances, and worst spots. Acceptance: research/brainstorming completed; focused tests prove verdict semantics; calibration evidence recorded; docs updated if user-facing output/API changes; full hot workspace suite passes under one minute.
