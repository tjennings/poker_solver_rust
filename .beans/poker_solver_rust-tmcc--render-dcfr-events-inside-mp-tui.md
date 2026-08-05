---
# poker_solver_rust-tmcc
title: Render DCFR events inside MP TUI
status: in-progress
type: bug
priority: high
created_at: 2026-08-05T19:03:52Z
updated_at: 2026-08-05T19:03:52Z
---

Route blueprint_mp DCFR scheduler output through TUI-owned state so discount events render on the bottom status/debug line instead of writing directly to stdout/stderr and corrupting the alternate-screen UI. Preserve useful non-TUI logging.

- [ ] Research current DCFR output and TUI state flow
- [ ] Design TUI/non-TUI event routing and message lifetime
- [ ] Establish clean focused baseline
- [ ] Implement bottom debug/status line
- [ ] Add deterministic tests
- [ ] Complete independent review and repairs
- [ ] Run focused and full correctness verification
- [ ] Integrate into main
