---
# poker_solver_rust-v1uw
title: 'Blueprint action memory: dormant actions via total regret pruning'
status: todo
type: task
priority: high
created_at: 2026-05-05T15:01:30Z
updated_at: 2026-05-05T15:01:54Z
parent: poker_solver_rust-ohyt
blocked_by:
    - poker_solver_rust-zu7v
---

Order 3. Strengthen pruning from traversal skipping into action dormancy that can reduce storage pressure for persistently bad actions.

Implementation notes:
- Build on existing regret pruning behavior without regressing terminal/fold handling.
- Track actions whose cumulative regret is deeply negative and mathematically ineligible for reactivation until a future iteration/epoch.
- Represent dormant actions with compact metadata instead of hot regret/strategy slots where possible.
- Rehydrate/reactivate actions when the regret bound permits.
- Make this optional/configurable because multiplayer guarantees are empirical, not two-player-zero-sum clean.

Acceptance criteria:
- Existing pruning tests still pass and are extended for dormancy/reactivation.
- Sizing/telemetry reports dormant action counts and estimated memory saved.
- Training can run with dormancy disabled for baseline comparisons.
- docs/training.md explains the config knobs and caveats.
