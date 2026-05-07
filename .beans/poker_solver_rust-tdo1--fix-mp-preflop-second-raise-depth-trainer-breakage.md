---
# poker_solver_rust-tdo1
title: Fix MP preflop second raise-depth trainer breakage
status: completed
type: bug
priority: high
created_at: 2026-05-07T04:56:36Z
updated_at: 2026-05-07T16:07:31Z
parent: poker_solver_rust-5kvv
---

Adding a second preflop raise depth such as raise: [["1.0x"], ["1.0x"]] breaks Blueprint MP trainer/tree construction. Reproduce, identify whether raise-depth indexing or action generation is responsible, and add a regression test.\n\n- [x] Reproduce second preflop raise-depth failure\n- [x] Fix MP game-tree/trainer behavior\n- [x] Add regression test\n- [x] Verify focused tests

## Reproduction Notes

Using the dirty 500f/100t/100r config with stack_depth: 200 and two preflop raise-depth rows produced: MP Tree: 272,170,499 nodes; MP Storage: 28,399,883,894 slots; virtual storage: 340.8 GB. The CLI reports this as 100bb deep because BB is 2 chips. The committed config was stack_depth: 40, so the current breakage is at least partly the config becoming 100bb while named 20bb.

## Scope Correction

100bb is an intended and normal target depth. The reproduced failure should be treated as an architecture limitation in the eager Blueprint MP tree/storage pipeline, not as a reason to constrain configs to 20bb.

## Summary of Changes

Resolved by routing 100bb 6-max configs with multiple preflop raise rows through the lazy_sparse backend instead of eager dense tree/storage construction. Added a committed 100bb lazy_sparse smoke config and regression test in poker_solver_rust-mniy that advances one meta-iteration and checks sparse storage remains bounded. The eager preflight still flags this shape as unsafe, but lazy_sparse no longer blocks it.

Verification: cargo test -q -p poker-solver-trainer lazy_sparse; cargo test -q -p poker-solver-trainer inspect_mp_config
