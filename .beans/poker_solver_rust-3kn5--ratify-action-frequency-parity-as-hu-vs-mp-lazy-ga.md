---
# poker_solver_rust-3kn5
title: Ratify action-frequency parity as HU vs MP lazy gate
status: in-progress
type: feature
priority: high
created_at: 2026-06-23T15:36:34Z
updated_at: 2026-06-23T15:43:52Z
parent: poker_solver_rust-osss
---

Promote the new HU-vs-MP-lazy counterfactual action-frequency parity metric from diagnostic evidence to the primary GO/NO-GO criterion for the 2-player replacement gate. Calibrate it on a small representative 2p harness run, preserve structural/schema/coverage failures as hard NO-GO, treat unresolved average-strategy accounting and local row-L1 as diagnostics/warnings rather than automatic blockers when action-frequency parity passes, and emit clear artifacts showing frequencies, tolerances, and worst spots. Acceptance: research/brainstorming completed; focused tests prove verdict semantics; calibration evidence recorded; docs updated if user-facing output/API changes; full hot workspace suite passes under one minute.

## 2026-06-23 Start Notes

Preflight:

- Working tree clean on codex/blueprint-lazy-tree-roadmap after tracker commit.
- Cold/noisy `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_3kn5_preflight.log 2>&1'` passed but took `real 109.19`.
- Hot redirected rerun `/usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_3kn5_preflight_hot.log 2>&1'` passed under the gate with `real 44.61`, `user 100.87`, `sys 15.01`.

Scope checklist:

- [x] Research/confirm GO-NO-GO semantics when action-frequency parity is primary and accounting mismatch becomes diagnostic.
- [x] Brainstorm the smallest maintainable harness/API/report change for ratification and calibration evidence.
- [ ] Implement verdict semantics: structural/schema/coverage/frequency threshold failures remain hard NO-GO; accounting mismatch and local row-L1 become diagnostics/warnings.
- [ ] Add or update calibration run evidence for the small representative 2p harness fixture.
- [ ] Update report artifacts/tests so reasons vs warnings are explicit and hard-gate behavior is covered.
- [ ] Update docs only if public/user-facing workflow changes.
- [ ] Run focused tests, diff hygiene, and hot full workspace suite under one minute.

## Research / Brainstorming Notes

Decision: promote combo-weighted root action-frequency parity to the primary behavioral gate, but split hard blockers from diagnostics. `reasons` should mean hard NO-GO only; add non-blocking `warnings` for accounting/local-row diagnostics.

Hard NO-GO must remain for failed structural checks, root schema/input mismatch, missing action-frequency evidence, non-169 canonical preflop distribution, missing MP root rows, skipped/invalid/zero/off-normalized rows, unmatched HU/MP action mass, insufficient included combo weight, uniform fallback rows, and primary action-frequency threshold breaches.

Warnings/diagnostics should include unresolved average-strategy accounting, max/mean root strategy L1, weighted row L1, max row L1, and worst bucket/action diagnostics. Direct per-bucket strategy L1 is no longer a verdict gate because HU and MP lazy average-strategy accounting differ by design today.

Current tiny harness calibration evidence: the default 2-iteration fixture only visits 2 of 169 MP root buckets, with included normalized weight about 0.012066 and skipped weight about 0.987934. It must remain NO-GO on coverage. The 1-iteration smoke is even sparser. Passing-path tests should therefore use synthetic complete-coverage reports until a real full-coverage training run is available.

Report wording should make clear that GO means root combo-weighted action frequencies matched within tolerance over complete canonical preflop coverage. It does not prove per-hand policy equivalence, post-root equivalence, EV parity, or exploitability parity.

Implementation plan:

- Add `warnings: Vec<String>` with serde default to `HuMpLazyReport`.
- Print warnings in `report.txt`/human summary and preserve JSON serialization.
- Update `finalize_report` to accept hard reasons and warnings, move accounting mismatch to warning, and move root strategy L1 threshold messages to warnings.
- Keep absent action-frequency evidence as hard NO-GO, including structurally valid preflight reports.
- Add tests proving clean action-frequency evidence can produce GO despite accounting/local-L1 warnings, while coverage/frequency/schema failures remain NO-GO.
- Update module docs; no broad docs required because there is no CLI/user workflow change.
