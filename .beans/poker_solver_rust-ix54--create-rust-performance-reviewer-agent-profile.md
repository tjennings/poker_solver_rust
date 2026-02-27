---
# poker_solver_rust-ix54
title: Create Rust Performance Reviewer agent profile
status: completed
type: task
priority: normal
created_at: 2026-02-26T18:32:16Z
updated_at: 2026-02-26T18:38:10Z
---

Create a new agent at ~/.claude/agents/ following the pattern of idiomatic-rust-enforcer and software-architect agents. The agent should encode all key techniques from the Rust Performance Book as a systematic review framework.

## Summary of Changes

Created `~/.claude/agents/rust-perf-reviewer.md` — a Rust Performance Reviewer agent profile encoding all key techniques from the Rust Performance Book by Nicholas Nethercote.

**Agent covers 11 review areas:**
1. Heap Allocations (Vec, String, Clone, collection reuse patterns)
2. Hashing (FxHashMap, AHashMap, nohash_hasher)
3. Iterators (collect avoidance, filter_map, chunks_exact, copied)
4. Bounds Checks (iteration vs indexing, slicing, assertions)
5. Type Sizes (boxing large variants, smaller indices, assert_eq_size)
6. Inlining (#[inline], #[cold], function splitting)
7. Standard Library micro-patterns (swap_remove, lazy eval, zero-page)
8. I/O (buffering, stdout locking)
9. Parallelism (rayon, crossbeam, wrapper consolidation)
10. Build Configuration (LTO, codegen-units, allocators, target-cpu)
11. Logging & Assertions (debug_assert in hot paths)

**Includes:**
- Anti-pattern table with 10 common perf mistakes
- 14-item review checklist
- Structured review process (profile → check → estimate → fix → verify)
- Severity-rated output format (High/Medium/Low impact)
- Persistent agent memory at `~/.claude/agent-memory/rust-perf-reviewer/`
- Poker-solver-aware notes (billions of iterations, inner loops matter)
