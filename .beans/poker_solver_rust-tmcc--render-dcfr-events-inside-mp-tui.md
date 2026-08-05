---
# poker_solver_rust-tmcc
title: Render DCFR events inside MP TUI
status: in-progress
type: bug
priority: high
created_at: 2026-08-05T19:03:52Z
updated_at: 2026-08-05T19:17:35Z
---

Route blueprint_mp DCFR scheduler output through TUI-owned state so discount events render on the bottom status/debug line instead of writing directly to stdout/stderr and corrupting the alternate-screen UI. Preserve useful non-TUI logging.

- [x] Research current DCFR output and TUI state flow
- [x] Design TUI/non-TUI event routing and message lifetime
- [x] Establish clean focused baseline
- [x] Implement bottom debug/status line
- [x] Add deterministic tests
- [ ] Complete independent review and repairs
- [x] Run focused and full correctness verification
- [ ] Integrate into main

## Baseline

Focused baseline passed 68/68 blueprint_mp trainer tests and 39/39 MP TUI tests. The complete workspace suite passed immediately before this adjacent repair under the existing long-suite runtime waiver.

## Approved Design

Add an optional runner-owned, thread-safe blueprint_mp training-event sink shared by eager and lazy paths. When a TUI sink is installed, DCFR schedule/pass/cap events update a latest-event slot in `BlueprintTuiMetrics` and do not write to the terminal; without a sink, preserve existing detailed stderr output. Render a compact completed-pass status on a dedicated one-line row immediately above the existing hotkey/snapshot footer, with latest-event-wins semantics, deterministic expiry for ordinary events, and durable cap-reached status. Install the sink before starting training. Remove redundant TUI-only snapshot success/failure `eprintln!` calls. Add core routing/formatting, adapter propagation, metrics lifetime, and Ratatui rendering tests.

## Implementation Progress

Added a runner-owned typed MP training event sink with stderr fallback, eager and lazy propagation, a latest-event TUI status slot with 60-second expiry and durable cap state, and a clipped status row above the independent hotkey/snapshot footer. Removed TUI-only snapshot stderr writes and documented the routing. Focused core trainer and lazy-adapter suites, metrics/TUI suites, full core lib, full trainer bin, formatting, and workspace check all pass.
