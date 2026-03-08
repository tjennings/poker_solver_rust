---
# poker_solver_rust-1c5b
title: Write per-hand chip EV to snapshot
status: completed
type: feature
priority: normal
created_at: 2026-03-08T13:34:35Z
updated_at: 2026-03-08T13:42:29Z
---

Accumulate chip EV per canonical preflop hand (169 classes) from traverse_external return values. Write hand_ev.json to each snapshot directory.

## Tasks
- [x] Add ev_accum/ev_count arrays (169 entries) to BlueprintTrainer
- [x] Capture traverse_external return values, map hole cards to canonical 169 index, accumulate atomically
- [x] Write hand_ev.json in save_snapshot (169-entry array of avg chip EVs)
- [x] Add test verifying hand_ev.json is written
- [x] All tests pass, clippy clean

## Summary of Changes
Single file change to trainer.rs: added AtomicI64/AtomicU64 arrays for EV accumulation, captured traverse_external return values in the training loop, and writes hand_ev.json (169-entry JSON object mapping hand names to avg chip EV) alongside each snapshot.
