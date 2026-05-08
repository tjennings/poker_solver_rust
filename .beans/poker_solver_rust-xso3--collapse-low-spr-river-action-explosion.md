---
# poker_solver_rust-xso3
title: Collapse low-SPR river action explosion
status: in-progress
type: bug
priority: high
created_at: 2026-05-08T01:21:06Z
updated_at: 2026-05-08T01:21:06Z
---

Lazy sparse insert attribution shows sustained new infosets concentrated on river, SPR bucket 0, history length 16-31, mostly 2-action nodes. Investigate and collapse low-SPR river betting actions that are creating leak-like state growth.\n\nTasks:\n- [ ] Inspect lazy MP action generation around low-SPR river states.\n- [ ] Add a conservative collapse/guard for river SPR bucket 0 action generation.\n- [ ] Add focused lazy action tests covering low-SPR river behavior.\n- [ ] Update training docs if the lazy_sparse backend behavior changes.\n- [ ] Run focused lazy tests.
