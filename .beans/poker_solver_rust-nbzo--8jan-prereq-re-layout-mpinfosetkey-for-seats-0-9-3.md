---
# poker_solver_rust-nbzo
title: '8jan-prereq: re-layout MpInfosetKey for seats 0-9 (3-bit seat mask blocker)'
status: in-progress
type: task
priority: high
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-22T15:14:51Z
parent: poker_solver_rust-osss
blocking:
    - poker_solver_rust-8jan
---

HARD BLOCKER for 2-10 players: the 128-bit MpInfosetKey uses a 3-bit SEAT_MASK (0x7), capping seats at 7 — seats 8-9 silently collide/alias. Re-layout the key (info_key.rs SEAT_SHIFT/MASK, BUCKET_SHIFT/MASK) to give seat >=4 bits, stealing from the 28-bit bucket or reserved bits without breaking the bucket street-namespace or the 22-slot/90-bit action history. Add round-trip + aliasing goldens across seats 0..=9 x buckets x histories; constructing a key beyond capacity must panic/error, never silently truncate. Sparse snapshot schema back-compat is NOT required. Must land BEFORE widening MAX_PLAYERS (8jan). Size: medium. From Phase 7 plan 2026-06-18.
