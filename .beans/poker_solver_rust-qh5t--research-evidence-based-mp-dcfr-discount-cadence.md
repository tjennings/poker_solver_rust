---
# poker_solver_rust-qh5t
title: Research evidence-based MP DCFR discount cadence
status: completed
type: task
priority: normal
created_at: 2026-08-05T14:05:35Z
updated_at: 2026-08-05T14:10:30Z
---

Review primary Libratus, Pluribus, and DCFR evidence for discount scheduling, compare wall-clock versus iteration-based epochs, and recommend a defensible cadence for the current blueprint_mp trainer.

- [x] Verify published system schedules from primary sources
- [x] Analyze cadence invariance and current trainer semantics
- [x] Recommend an evidence-based default and validation experiment
- [x] Report confidence and uncertainties

Findings:
- Canonical DCFR discounts every full CFR iteration; no interval ablation establishes a universal optimum.
- The sampled MCCFR experiment in the DCFR paper used batched LCFR every 10^7 nodes touched.
- Pluribus used batched LCFR every 10 wall-clock minutes for 400 minutes (40 epochs), then stopped. Published Libratus does not report this schedule.
- For blueprint_mp, 10 minutes is the strongest production prior, but should be translated after measuring warmed-up throughput and aligned to batch boundaries.
- Current exact-modulo triggering can realize lcm(batch_size, interval) rather than the configured interval. Factor epoch is coupled to interval, and beta=0 plus integer truncation makes cadence the negative-regret half-life.
- Recommended validation: 2.5/10/40-minute or equivalent work-normalized cadence sweep, no-discount control, fixed seeds/budgets, exploitability/BR metrics, quantization and scan-overhead telemetry, plus stop-after-40 versus indefinite.
