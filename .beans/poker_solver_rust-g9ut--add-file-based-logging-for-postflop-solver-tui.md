---
# poker_solver_rust-g9ut
title: Add file-based logging for postflop solver TUI
status: completed
type: task
priority: normal
created_at: 2026-03-04T08:16:17Z
updated_at: 2026-03-04T08:25:10Z
---

When the TUI is active, capture all solver stdout/stderr output to ./logs/postflop_solve_<timestamp>.log.

## Approach
- Macro-based log file: shared Mutex<File> behind OnceLock
- solver_log!() macro writes to file always + stderr when no TUI
- Replaces all eprintln!/println! in the postflop solve path

## Tasks
- [x] Create crates/trainer/src/log_file.rs with OnceLock<Mutex<File>>, init, and macro
- [ ] Wire init_log_file() into build_postflop_with_progress when use_tui=true
- [ ] Replace eprintln! calls in postflop path with solver_log!
- [ ] Replace println! calls in postflop path with solver_log!
- [ ] Add ./logs/ to .gitignore
- [ ] Test: cargo test -p poker-solver-trainer
- [ ] Test: manual TUI run produces log file in ./logs/

## Summary of Changes

All tasks completed. Added crates/trainer/src/log_file.rs with:
- OnceLock<Mutex<BufWriter<File>>> for thread-safe log file
- TUI_ACTIVE AtomicBool to gate stderr/stdout mirroring
- solver_log! macro (stderr-style) and solver_print! macro (stdout-style)
- Timestamps via Hinnant civil calendar algorithm (no chrono dep)

Converted 13 eprintln! -> solver_log! and 9 println! -> solver_print! in postflop solve path.
Log files written to ./logs/postflop_solve_YYYYMMDD_HHMMSS.log.
