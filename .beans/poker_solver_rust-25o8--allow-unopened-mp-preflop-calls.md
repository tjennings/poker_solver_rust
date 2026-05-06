---
# poker_solver_rust-25o8
title: Allow unopened MP preflop calls
status: in-progress
type: task
priority: high
created_at: 2026-05-06T05:44:17Z
updated_at: 2026-05-06T05:44:17Z
---

Update blueprint_mp unopened preflop action generation in the current checkout so unopened positions can fold, call/limp, or use configured non-all-in open sizes. Keep raise and all-in unavailable until after a voluntary opening action.
