---
# poker_solver_rust-thkt
title: Counterfactual action-frequency parity gate for HU vs MP lazy
status: in-progress
type: feature
priority: high
created_at: 2026-06-23T13:28:24Z
updated_at: 2026-06-23T13:36:03Z
parent: poker_solver_rust-osss
---

Build an evidence-producing counterfactual action-frequency parity evaluator for HU blueprint_v2 vs MP lazy-sparse 2-player strategies. The evaluator should use matched public infosets and a shared/canonical evaluation distribution so aggregate action-frequency deltas are not confounded by each backend's own reach distribution. It should extend or sit beside the existing hu_mp_lazy convergence harness, report human-readable poker frequencies (open, fold, call, raise/all-in as applicable), keep local policy-distance diagnostics for attribution, and emit durable artifacts. Acceptance: plan/research completed; focused tests cover normalization, weighted frequency aggregation, mismatch handling, and report serialization; full hot workspace suite passes under the one-minute gate.

## 2026-06-23 Start Notes

Preflight:

- Working tree clean on codex/blueprint-lazy-tree-roadmap after tracker commit.
- Cold/noisy /usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_thkt_preflight.log 2>&1' passed but took real 329.54, dominated by rustdoc/test artifact work.
- Hot redirected rerun /usr/bin/time -p sh -c 'cargo test --workspace --quiet >/tmp/poker_solver_thkt_preflight_hot.log 2>&1' passed under the gate with real 44.61, user 95.88, sys 14.31.

Scope checklist:

- [ ] Research counterfactual action-frequency equivalence metrics for HU vs MP lazy and identify confounders/tolerances.
- [ ] Brainstorm the smallest credible evaluator/report extension that reuses the existing hu_mp_lazy harness.
- [ ] Implement matched-public-infoset action-frequency aggregation with a shared canonical weighting distribution.
- [ ] Preserve local policy-distance diagnostics for attribution and keep the conservative GO/NO-GO semantics.
- [ ] Emit durable report artifacts for frequency deltas and worst spots.
- [ ] Add focused tests for normalization, weighted aggregation, mismatch handling, and serialization.
- [ ] Run focused tests, git diff --check, and the hot full workspace suite under one minute.
