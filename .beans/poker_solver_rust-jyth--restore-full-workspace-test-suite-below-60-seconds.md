---
# poker_solver_rust-jyth
title: Restore full workspace test suite below 60 seconds
status: in-progress
type: bug
priority: high
created_at: 2026-08-04T18:01:38Z
updated_at: 2026-08-04T18:01:38Z
---

The mandatory pre-development baseline `/usr/bin/time -p cargo test --workspace --quiet` passes all tests but takes 62.83 seconds, exceeding the repository hard limit. Diagnose and reduce reliable warm-suite wall time below 60 seconds without weakening meaningful coverage.

- [ ] Reproduce and attribute the slowest critical-path tests or startup costs
- [ ] Design the smallest safe runtime reduction
- [ ] Implement through a rust-developer worktree
- [ ] Obtain independent code review
- [ ] Verify the full quiet workspace suite passes below 60 seconds
