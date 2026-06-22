---
# poker_solver_rust-nbzo
title: '8jan-prereq: re-layout MpInfosetKey for seats 0-9 (3-bit seat mask blocker)'
status: completed
type: task
priority: high
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-22T18:13:14Z
parent: poker_solver_rust-osss
blocking:
    - poker_solver_rust-8jan
---

HARD BLOCKER for 2-10 players: the 128-bit MpInfosetKey uses a 3-bit SEAT_MASK (0x7), capping seats at 7 — seats 8-9 silently collide/alias. Re-layout the key (info_key.rs SEAT_SHIFT/MASK, BUCKET_SHIFT/MASK) to give seat >=4 bits, stealing from the 28-bit bucket or reserved bits without breaking the bucket street-namespace or the 22-slot/90-bit action history. Add round-trip + aliasing goldens across seats 0..=9 x buckets x histories; constructing a key beyond capacity must panic/error, never silently truncate. Sparse snapshot schema back-compat is NOT required. Must land BEFORE widening MAX_PLAYERS (8jan). Size: medium. From Phase 7 plan 2026-06-18.

## Summary of Changes

Widened the packed `InfoKey128` seat field from 3 to 4 bits (bits 63-60, capacity 0-15) by taking 1 bit from the reserved padding; the 28-bit bucket + its street namespace (street<<14 | 14-bit local) and the 90-bit/22-slot action history are preserved exactly. Replaced the silent `& 0x7` truncation with a hard overflow guard (panics on seat >= capacity) in both `InfoKey128::pack_header` and `MpInfosetKey::from_parts`, sharing one `SEAT_BITS`-derived source of truth (`SEAT_MASK` derived from `SEAT_BITS`; `SEAT_CAPACITY` shared across both types).

Eliminates the latent silent aliasing of seats 8-9 -> 0-1 BEFORE `MAX_PLAYERS` is widened (that is bean 8jan). Does NOT bump MAX_PLAYERS or resize fixed arrays. The active lazy-sparse path (`MpInfosetKey`, plain u8 seat) and the universal lazy exporter were unaffected (verified).

Tests: aliasing golden over 10 seats x 8 buckets x 5 histories (incl. the previously-aliasing 8/9 vs 0/1 pairs), round-trip 0..=15, overflow-panic, street-namespace-across-seats, 22-action capacity, and two proptests (round-trip + no-collision). Approved on the first review round (0 fix rounds); a follow-up commit hardened the seat constants per the simplicity finding.

Verification: `cargo build --workspace --tests` clean; nbzo tests green (worktree run 1698 passed / 4 pre-existing baseline_validation sandbox failures; core run 1253 passed). Merged to codex/blueprint-lazy-tree-roadmap. NOTE: the final warm full-suite timing run in the main checkout was still completing at handoff; prior runs were green; suite wall-time is variable and tracked in z8jx.
