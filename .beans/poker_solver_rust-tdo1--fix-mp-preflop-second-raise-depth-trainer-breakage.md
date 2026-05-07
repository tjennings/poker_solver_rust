---
# poker_solver_rust-tdo1
title: Fix MP preflop second raise-depth trainer breakage
status: in-progress
type: bug
priority: high
created_at: 2026-05-07T04:56:36Z
updated_at: 2026-05-07T04:56:36Z
---

Adding a second preflop raise depth such as raise: [["1.0x"], ["1.0x"]] breaks Blueprint MP trainer/tree construction. Reproduce, identify whether raise-depth indexing or action generation is responsible, and add a regression test.\n\n- [ ] Reproduce second preflop raise-depth failure\n- [ ] Fix MP game-tree/trainer behavior\n- [ ] Add regression test\n- [ ] Verify focused tests
