---
# poker_solver_rust-4rxu
title: BoundaryTracer holds unbounded open file handles per ordinal
status: todo
type: bug
priority: normal
created_at: 2026-04-24T13:12:55Z
updated_at: 2026-04-24T13:12:55Z
---

Surfaced during Task 14 iter-15 E2E run on branch feat/option-a-gadget-tree (2026-04-24).

The tracer's handles field is a HashMap<usize, BufWriter<File>> with no cap — one open file per traced ordinal, held for the lifetime of the tracer. On the JhTh9h|…|7d turn subgame with ~247 real boundaries + 2 gadget boundaries, this hit macOS' default 256 fd soft limit:

    thread '<unnamed>' panicked at crates/tauri-app/src/boundary_trace.rs:845:37:
    failed to open boundary_249.txt: Too many open files (os error 24)

Immediate mitigation (commit TBD): skip_leading_ordinals filters out gadget ordinals 0/1 from tracing, bringing the count back to 247 which fits under the limit. This unblocks Task 14.

Latent issue: any non-gadget subgame with 250+ boundaries will still fail. Proper fix: bounded fd pool with LRU eviction, or open-write-close per trace event.

Blocked-by poker_solver_rust-02wj (Option A ship).
Trigger: when the first report of non-gadget trace failure comes in, or proactively if a planned subgame config exceeds ~200 boundaries.
