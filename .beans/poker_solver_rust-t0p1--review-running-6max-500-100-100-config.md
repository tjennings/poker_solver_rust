---
# poker_solver_rust-t0p1
title: Review running 6max 500-100-100 config
status: completed
type: task
priority: normal
created_at: 2026-05-07T03:22:07Z
updated_at: 2026-05-07T03:24:02Z
---

Inspect the user's running 6max 500/100/100 Blueprint MP config after postflop action pruning and report any consistency issues or risks without modifying the config.



## Summary of Changes

Reviewed sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml after the user's postflop action pruning. Did not modify the config.

Findings:
- YAML shape is valid and still points at 169/500/100/100 buckets under ./local_data/buckets/500f_100t_100r_v1.
- Postflop action pruning is coherent and explains the reported speedup: flop now has one lead and one raise size, turn one lead and one raise size, and river one lead plus two raise sizes.
- Preflop raise sizes [0.33x, 1.0x] are effectively duplicate min-raises in common facing-open spots because undersized multiplier raises are clamped to min_raise_to and duplicate action sizes are dropped.
- The output_dir name still matches the broad 500f_100t_100r experiment, not the newly pruned action abstraction; consider renaming before long comparative runs if artifact clarity matters.

Caveat: Could not inspect the running process command line because ps is blocked in this sandbox. No tests or training command were launched for this read-only review.
