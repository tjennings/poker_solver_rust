---
# poker_solver_rust-venl
title: Reduce full cargo test runtime below one minute
status: in-progress
type: bug
priority: critical
created_at: 2026-08-05T14:30:33Z
updated_at: 2026-08-05T14:30:33Z
blocking:
    - poker_solver_rust-slxt
---

The mandatory baseline `cargo test` exceeded 60 seconds and was interrupted at 104.37 seconds. Diagnose and make the entire suite complete under one minute without losing coverage before wall-clock DCFR feature implementation continues.

- [ ] Identify the slow build/test target with reproducible timing
- [ ] Implement the smallest safe runtime repair
- [ ] Verify full cargo test passes under 60 seconds
- [ ] Review and integrate the repair
