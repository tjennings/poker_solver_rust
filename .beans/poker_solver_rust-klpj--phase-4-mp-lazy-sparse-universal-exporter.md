---
# poker_solver_rust-klpj
title: 'Phase 4: MP lazy sparse universal exporter'
status: todo
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-09T18:41:22Z
parent: poker_solver_rust-a29s
---

Export realized MP lazy sparse rows into the universal format for analysis/read-only loading. Acceptance: SparseSnapshotEntry rows are sorted by semantic key and exported with verbatim semantic identity; strategy sums normalize correctly; zero-sum rows use documented uniform fallback; artifact is marked non-resumable.
