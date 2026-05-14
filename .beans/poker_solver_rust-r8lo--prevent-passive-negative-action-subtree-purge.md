---
# poker_solver_rust-r8lo
title: Prevent passive negative-action subtree purge
status: in-progress
type: bug
priority: critical
created_at: 2026-05-14T17:51:17Z
updated_at: 2026-05-14T17:51:17Z
---

After first negative-action purge, open strategies for positions after UTG appear wiped. Root cause is likely purging/blocking passive MP action edges such as UTG fold, whose child history contains later players' unopened spots. Restrict persistent negative-action subtree purge to aggressive actions and update regression tests/docs.
