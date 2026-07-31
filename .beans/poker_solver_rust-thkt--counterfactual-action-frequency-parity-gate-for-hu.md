---
# poker_solver_rust-thkt
title: Counterfactual action-frequency parity gate for HU vs MP lazy
status: completed
type: feature
priority: high
created_at: 2026-06-23T13:28:24Z
updated_at: 2026-06-23T14:29:17Z
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
- [x] Implement matched-public-infoset action-frequency aggregation with a shared canonical weighting distribution.
- [x] Preserve local policy-distance diagnostics for attribution and keep the conservative GO/NO-GO semantics.
- [x] Emit durable report artifacts for frequency deltas and worst spots.
- [x] Add focused tests for normalization, weighted aggregation, mismatch handling, and serialization.
- [x] Run focused tests, git diff --check, and the hot full workspace suite under one minute.

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

## Summary of Changes

Implemented a root-only counterfactual action-frequency parity section in `crates/convergence-harness/src/hu_mp_lazy.rs`. The existing HU-vs-MP-lazy harness now reports combo-count weighted root action frequencies under the shared canonical 169-hand preflop distribution, using `CanonicalHand::from_index(bucket).num_combos()` and total mass 1326.

The metric uses raw source probabilities at matched action indices. It does not renormalize after filtering comparable actions, so unmatched/filtered HU or MP action mass is surfaced as coverage evidence and prevents threshold pass. Missing MP sparse root rows are skipped and reported instead of being silently treated as uniform fallback parity.

Added nested action-frequency report types, config thresholds, aggregate action rows, worst bucket/action spots, coverage totals, report.txt output, and CSV artifacts: `root_action_frequencies.csv` and `root_action_frequency_spots.csv`. Existing local strategy-distance diagnostics remain for attribution, and the harness remains conservative NO-GO because average-strategy accounting is still unreconciled.

Review fixed a blocking metric bug where projected rows were originally renormalized and could hide filtered action mass. Final review found no blockers.

Verification:

- `cargo test -p convergence-harness hu_mp_lazy -- --nocapture` passed (11 tests).
- `cargo test -p convergence-harness --tests -- --nocapture` passed (110 lib + 13 main tests, 3 ignored integration tests).
- `git diff --check` passed.
- `rustfmt --check crates/convergence-harness/src/hu_mp_lazy.rs` passed.
- Hot redirected full workspace suite passed with `real 44.96`, `user 96.62`, `sys 14.52`.
