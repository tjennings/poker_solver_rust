---
# poker_solver_rust-hfnv
title: 'Epic: improve potential-aware postflop bucketing fidelity'
status: todo
type: epic
priority: high
created_at: 2026-05-06T16:38:52Z
updated_at: 2026-05-06T16:38:52Z
---

Track fixes and design work to make the blueprint postflop bucket abstraction match potential-aware intent more closely.

Scope:
- Fix correctness bugs found in the clustering audit
- Keep preflop lossless canonical hand mapping; lossy bucketing begins on flop
- Preserve ordered child-bucket distances through CFVNet and EMD paths
- Add a principled way to represent nut-distance / dominance within potential-aware features
- Strengthen diagnostics so they measure the same canonical lookups used by training/runtime
