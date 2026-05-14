---
# poker_solver_rust-ropf
title: Audit CFVNet v5 training data distribution
status: completed
type: task
priority: high
created_at: 2026-05-14T05:14:35Z
updated_at: 2026-05-14T05:20:47Z
parent: poker_solver_rust-lnpl
---

Measure the local CFVNet training data distribution using the same reach-density, target-magnitude, player, pot, and SPR buckets that explain the compare-solve failure.

## Summary of Changes

Audited `local_data/cfvnet/turn_boundary/v2` via manifest inspection and direct binary sampling.

Findings:
- Dataset has 1,000 shards / 10,000,000 records, `turn_boundary`, `exact_river`, record size 17,256 bytes.
- Manifest coverage is entirely `range_source=blueprint` and almost entirely `entropy_high` (9,987,414 / 10,000,000 records).
- Corrected paired sample: 20,000 records, balanced OOP/IP (10,000 each), zero bad records.
- No sampled all-in records; all sampled records had positive effective stack.
- No sampled player-range density below 30%; OOP and IP player reach buckets were 100% `>=30%`.
- No sampled target mean-abs bucket `>=0.75`; OOP was 9,995 / 10,000 in `0.25-0.75`, IP was 10,000 / 10,000 in `0.25-0.75`.
- Runtime failure buckets from comparison (`IP reach 15-30%`, `IP target >=0.75`, IP all-in) have no observed support in the sampled v5 training data.

Conclusion: the v5 failure is consistent with training-data distribution mismatch, not CFVNet boundary normalization. Long-term fix should add constrained/sparse range and high-magnitude/all-in-pressure strata to turn-boundary data generation and expose those buckets in manifest/audit tooling.
