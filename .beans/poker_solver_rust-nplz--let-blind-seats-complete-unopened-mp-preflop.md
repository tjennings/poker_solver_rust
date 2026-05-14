---
# poker_solver_rust-nplz
title: Let blind seats complete unopened MP preflop
status: in-progress
type: bug
priority: high
created_at: 2026-05-14T20:12:03Z
updated_at: 2026-05-14T20:12:03Z
---

The MP allow_preflop_limp=false gate currently removes every unopened preflop Call, including SB completion. It should only remove cold limps from seats with no posted blind; SB can call/complete and BB can check when action reaches it.
