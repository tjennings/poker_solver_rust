---
# poker_solver_rust-venl
title: Reduce full cargo test runtime below one minute
status: in-progress
type: bug
priority: critical
created_at: 2026-08-05T14:30:33Z
updated_at: 2026-08-05T14:49:06Z
blocking:
    - poker_solver_rust-slxt
---

The mandatory baseline `cargo test` exceeded 60 seconds and was interrupted at 104.37 seconds. Diagnose and make the entire suite complete under one minute without losing coverage before wall-clock DCFR feature implementation continues.

- [ ] Identify the slow build/test target with reproducible timing
- [ ] Implement the smallest safe runtime repair
- [ ] Verify full cargo test passes under 60 seconds
- [ ] Review and integrate the repair

## Review Correction

Independent review rejected commit 02cbaab5: the Tauri explorer test was already ignored, so adding an ignore reason is runtime-neutral and the branch will not be integrated. Subsequent worktree verification also contaminated the shared target directory and produced invalid Cargo waits. Runtime validation must be repeated from the primary worktree with no concurrent Cargo process after a complete no-run build.
