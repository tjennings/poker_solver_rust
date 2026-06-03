---
# poker_solver_rust-wozh
title: Verify Blueprint MP preflop bucket and sparse key identity
status: todo
type: bug
priority: high
created_at: 2026-05-20T13:27:56Z
updated_at: 2026-05-20T13:27:56Z
parent: poker_solver_rust-kiqt
---

Ensure preflop hand buckets and sparse infoset keys are identical between traversal storage writes and diagnostic/TUI reads.

## Subtasks

- [ ] Verify CanonicalHand index mapping for suited Ax, offsuit Ax, pairs, and trash hands
- [ ] Verify preflop uses exact 169 canonical buckets, not postflop bucket files or equity fallback
- [ ] Verify sparse keys include the same seat, street, street-local bucket, history, and history length in writer and reader
- [ ] Compare key construction from lazy traversal against LazyResolvedSpot::key_for_bucket
- [ ] Add a focused regression that writes one strategy row and reads it through the same path the TUI uses
- [ ] Document any key namespace assumptions that are not obvious
