---
# poker_solver_rust-3kn5
title: Ratify action-frequency parity as HU vs MP lazy gate
status: in-progress
type: feature
priority: high
created_at: 2026-06-23T15:36:34Z
updated_at: 2026-06-23T15:39:40Z
parent: poker_solver_rust-osss
---

Promote the new HU-vs-MP-lazy counterfactual action-frequency parity metric from diagnostic evidence to the primary GO/NO-GO criterion for the 2-player replacement gate. Calibrate it on a small representative 2p harness run, preserve structural/schema/coverage failures as hard NO-GO, treat unresolved average-strategy accounting and local row-L1 as diagnostics/warnings rather than automatic blockers when action-frequency parity passes, and emit clear artifacts showing frequencies, tolerances, and worst spots. Acceptance: research/brainstorming completed; focused tests prove verdict semantics; calibration evidence recorded; docs updated if user-facing output/API changes; full hot workspace suite passes under one minute.

## 2026-06-23 Start Notes

Preflight:

- Working tree clean on codex/blueprint-lazy-tree-roadmap after tracker commit.
- Cold/noisy `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_3kn5_preflight.log 2>&1'` passed but took `real 109.19`.
- Hot redirected rerun `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_3kn5_preflight_hot.log 2>&1'` passed under the gate with `real 44.61`, `user 100.87`, `sys 15.01`.

Scope checklist:

- [ ] Research/confirm GO-NO-GO semantics when action-frequency parity is primary and accounting mismatch becomes diagnostic.
- [ ] Brainstorm the smallest maintainable harness/API/report change for ratification and calibration evidence.
- [ ] Implement verdict semantics: structural/schema/coverage/frequency threshold failures remain hard NO-GO; accounting mismatch and local row-L1 become diagnostics/warnings.
- [ ] Add or update calibration run evidence for the small representative 2p harness fixture.
- [ ] Update report artifacts/tests so reasons vs warnings are explicit and hard-gate behavior is covered.
- [ ] Update docs only if public/user-facing workflow changes.
- [ ] Run focused tests, diff hygiene, and hot full workspace suite under one minute.
