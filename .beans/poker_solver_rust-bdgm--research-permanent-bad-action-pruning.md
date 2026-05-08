---
# poker_solver_rust-bdgm
title: Research permanent bad-action pruning
status: completed
type: task
priority: high
created_at: 2026-05-08T01:51:27Z
updated_at: 2026-05-08T01:53:18Z
parent: poker_solver_rust-5kvv
---

Review CFR/regret-pruning literature and poker-solving prior art for periodically banning extremely bad actions from lazy sparse Blueprint MP training before choosing an implementation direction.

Tasks:
- [x] Review local docs and existing pruning behavior.
- [x] Review online papers/prior art on regret-based pruning and action abstraction refinement.
- [x] Summarize candidate approaches, risks, and recommended first experiment.

## Summary of Changes

Reviewed local lazy_sparse pruning behavior and external literature. Key takeaway: start with observe-mode dormant action masking rather than irreversible deletion. Prior art supports temporary regret-based pruning, best-response/total RBP for space savings in two-player zero-sum settings, average-strategy sampling for many-action MCCFR, and dynamic action abstraction/refinement. Permanent bans need strong safeguards because poker action abstraction can be nonmonotonic and multiplayer/raked Blueprint MP weakens theory assumptions.

Recommended first experiment: add action-pruning observe telemetry with warmup, visit floors, persistence windows, protected actions, and by-street/action/SPR/history attribution before enforcing any mask.
