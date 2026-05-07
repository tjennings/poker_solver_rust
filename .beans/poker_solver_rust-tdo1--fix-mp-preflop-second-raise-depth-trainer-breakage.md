---
# poker_solver_rust-tdo1
title: Fix MP preflop second raise-depth trainer breakage
status: in-progress
type: bug
priority: high
created_at: 2026-05-07T04:56:36Z
updated_at: 2026-05-07T05:00:15Z
---

Adding a second preflop raise depth such as raise: [["1.0x"], ["1.0x"]] breaks Blueprint MP trainer/tree construction. Reproduce, identify whether raise-depth indexing or action generation is responsible, and add a regression test.\n\n- [x] Reproduce second preflop raise-depth failure\n- [ ] Fix MP game-tree/trainer behavior\n- [ ] Add regression test\n- [ ] Verify focused tests

## Reproduction Notes

Using the dirty 500f/100t/100r config with stack_depth: 200 and two preflop raise-depth rows produced: MP Tree: 272,170,499 nodes; MP Storage: 28,399,883,894 slots; virtual storage: 340.8 GB. The CLI reports this as 100bb deep because BB is 2 chips. The committed config was stack_depth: 40, so the current breakage is at least partly the config becoming 100bb while named 20bb.
