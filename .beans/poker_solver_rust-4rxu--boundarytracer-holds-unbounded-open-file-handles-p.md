---
# poker_solver_rust-4rxu
title: BoundaryTracer holds unbounded open file handles per ordinal
status: in-progress
type: bug
priority: normal
created_at: 2026-04-24T13:12:55Z
updated_at: 2026-06-22T18:27:45Z
---

Surfaced during Task 14 iter-15 E2E run on branch feat/option-a-gadget-tree (2026-04-24).

The tracer's handles field is a HashMap<usize, BufWriter<File>> with no cap — one open file per traced ordinal, held for the lifetime of the tracer. On the JhTh9h|…|7d turn subgame with ~247 real boundaries + 2 gadget boundaries, this hit macOS' default 256 fd soft limit:

    thread '<unnamed>' panicked at crates/tauri-app/src/boundary_trace.rs:845:37:
    failed to open boundary_249.txt: Too many open files (os error 24)

Immediate mitigation (commit TBD): skip_leading_ordinals filters out gadget ordinals 0/1 from tracing, bringing the count back to 247 which fits under the limit. This unblocks Task 14.

Latent issue: any non-gadget subgame with 250+ boundaries will still fail. Proper fix: bounded fd pool with LRU eviction, or open-write-close per trace event.

Blocked-by poker_solver_rust-02wj (Option A ship).
Trigger: when the first report of non-gadget trace failure comes in, or proactively if a planned subgame config exceeds ~200 boundaries.

## 2026-06-22 Gate Failure Evidence

Before starting trainer consolidation follow-up work, `cargo test --workspace` failed in the main checkout after `real 495.83s`. Failure:

```text
boundary_trace::tests::tracer_writes_txt_file panicked at crates/tauri-app/src/boundary_trace.rs:1606:9:
trace file should exist as .txt
```

Focused reruns passed:

```text
cargo test -p poker-solver-tauri boundary_trace::tests::tracer_writes_txt_file -- --nocapture
cargo test -p poker-solver-tauri --lib boundary_trace::tests::tracer_ -- --nocapture
```

This is now blocking the pre-development suite gate. The likely fix direction remains eliminating long-lived per-ordinal `BufWriter<File>` handles or making trace test temp directories unique and race-proof, with preference for fixing the production fd issue rather than only hardening the test.
