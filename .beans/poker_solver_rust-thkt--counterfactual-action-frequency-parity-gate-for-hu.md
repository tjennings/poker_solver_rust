---
# poker_solver_rust-thkt
title: Counterfactual action-frequency parity gate for HU vs MP lazy
status: in-progress
type: feature
priority: high
created_at: 2026-06-23T13:28:24Z
updated_at: 2026-06-23T13:39:34Z
parent: poker_solver_rust-osss
---

Build an evidence-producing counterfactual action-frequency parity evaluator for HU blueprint_v2 vs MP lazy-sparse 2-player strategies. The evaluator should use matched public infosets and a shared/canonical evaluation distribution so aggregate action-frequency deltas are not confounded by each backend's own reach distribution. It should extend or sit beside the existing hu_mp_lazy convergence harness, report human-readable poker frequencies (open, fold, call, raise/all-in as applicable), keep local policy-distance diagnostics for attribution, and emit durable artifacts. Acceptance: plan/research completed; focused tests cover normalization, weighted frequency aggregation, mismatch handling, and report serialization; full hot workspace suite passes under the one-minute gate.

## 2026-06-23 Start Notes

Preflight:

- Working tree clean on codex/blueprint-lazy-tree-roadmap after tracker commit.
- Cold/noisy /usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_thkt_preflight.log 2>&1' passed but took real 329.54, dominated by rustdoc/test artifact work.
- Hot redirected rerun /usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_thkt_preflight_hot.log 2>&1' passed under the gate with real 44.61, user 95.88, sys 14.31.

Scope checklist:

- [x] Research counterfactual action-frequency equivalence metrics for HU vs MP lazy and identify confounders/tolerances.
- [x] Brainstorm the smallest credible evaluator/report extension that reuses the existing hu_mp_lazy harness.
- [ ] Implement matched-public-infoset action-frequency aggregation with a shared canonical weighting distribution.
- [ ] Preserve local policy-distance diagnostics for attribution and keep the conservative GO/NO-GO semantics.
- [ ] Emit durable report artifacts for frequency deltas and worst spots.
- [ ] Add focused tests for normalization, weighted aggregation, mismatch handling, and serialization.
- [ ] Run focused tests, git diff --check, and the hot full workspace suite under one minute.

## Research / Brainstorming Notes

Conclusion: add a root-only counterfactual action-frequency parity section to the existing `HuMpLazyReport`, not a separate harness. Raw sampled action frequency is rejected as a gate because it is confounded by RNG, traversal schedule, native reach distribution, sparse visitation, and the unresolved HU/MP average-strategy accounting difference.

Primary metric for this slice:

- Matched public infoset: root only.
- Matched actions: existing normalized root action descriptors from `hu_mp_lazy`.
- Shared external distribution: combo-count weighted canonical preflop buckets, using `CanonicalHand::from_index(bucket).num_combos()` so total mass is 1326 combos. This is preferred over uniform 169-bucket weighting because uniform buckets overweight pairs and suited hands relative to actual dealt hand mass.
- Formula: `F_X(a) = (1 / 1326) * sum_b combo_count(b) * pi_X(a | root, b)` for HU and MP-lazy projected onto matched root actions. Report per-action frequency deltas, total frequency L1, max action delta, and local weighted row-L1 diagnostics.

Conservative semantics:

- This remains a diagnostic/GO-NO-GO gate, not proof that HU can be retired.
- Keep the existing unresolved accounting reason until HU traverser-only strategy sums and MP lazy sampled-opponent sums are reconciled or the evaluator is calibrated around that difference.
- NO-GO on root schema mismatch, invalid/nonfinite probability rows, invalid bucket distribution, no comparable evidence, incomplete MP root row coverage after enough iterations, or action-frequency threshold breach.

Implementation slice:

- Add nested action-frequency report structs to `HuMpLazyReport`.
- Add config thresholds: max per-action frequency delta, total action-frequency L1, and number of worst spots to retain.
- Extend the existing root comparison loop to compute combo-weighted aggregate action frequencies and worst bucket/action deltas.
- Emit `root_action_frequencies.csv` and `root_action_frequency_spots.csv`, plus a report.txt action-frequency section.
- Keep postflop/deeper matched public-state rollout, CLI/TUI wiring, production baseline runs, and EV/exploitability parity out of this slice.
