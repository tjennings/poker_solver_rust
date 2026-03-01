---
# poker_solver_rust-3qu8
title: Postflop TUI Dashboard
status: in-progress
type: feature
created_at: 2026-03-01T16:24:44Z
updated_at: 2026-03-01T16:24:44Z
---

Replace indicatif progress bars in solve-postflop with a full-screen ratatui TUI dashboard showing real-time solver metrics.

## Tasks
- [ ] Task 1: Update dependencies (swap indicatif for ratatui/crossterm/dashmap/atty)
- [ ] Task 2: Create TuiMetrics shared state
- [ ] Task 3: Instrument solver hot path with atomic counters
- [ ] Task 4: Build TUI renderer
- [ ] Task 5: Wire TUI into solve-postflop pipeline
- [ ] Task 6: TTY fallback for non-interactive use
- [ ] Task 7: Integration test

## Design
See docs/plans/2026-03-01-postflop-tui-dashboard-design.md
## Implementation Plan
See docs/plans/2026-03-01-postflop-tui-dashboard.md
