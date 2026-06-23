---
# poker_solver_rust-thkt
title: Counterfactual action-frequency parity gate for HU vs MP lazy
status: in-progress
type: feature
priority: high
created_at: 2026-06-23T13:28:24Z
updated_at: 2026-06-23T13:28:24Z
parent: poker_solver_rust-osss
---

Build an evidence-producing counterfactual action-frequency parity evaluator for HU blueprint_v2 vs MP lazy-sparse 2-player strategies. The evaluator should use matched public infosets and a shared/canonical evaluation distribution so aggregate action-frequency deltas are not confounded by each backend's own reach distribution. It should extend or sit beside the existing hu_mp_lazy convergence harness, report human-readable poker frequencies (open, fold, call, raise/all-in as applicable), keep local policy-distance diagnostics for attribution, and emit durable artifacts. Acceptance: plan/research completed; focused tests cover normalization, weighted frequency aggregation, mismatch handling, and report serialization; full hot workspace suite passes under the one-minute gate.
