---
# poker_solver_rust-1c5b
title: Write per-hand chip EV to snapshot
status: in-progress
type: feature
created_at: 2026-03-08T13:34:35Z
updated_at: 2026-03-08T13:34:35Z
---

Accumulate chip EV per canonical preflop hand (169 classes) from traverse_external return values. Write hand_ev.json to each snapshot directory.

## Tasks
- [ ] Add ev_accum/ev_count arrays (169 entries) to BlueprintTrainer
- [ ] Capture traverse_external return values, map hole cards to canonical 169 index, accumulate atomically
- [ ] Write hand_ev.json in save_snapshot (169-entry array of avg chip EVs)
- [ ] Add test verifying hand_ev.json is written
- [ ] All tests pass, clippy clean
