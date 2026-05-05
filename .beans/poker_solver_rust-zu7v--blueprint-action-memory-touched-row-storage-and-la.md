---
# poker_solver_rust-zu7v
title: 'Blueprint action memory: touched-row storage and lazy DCFR discounts'
status: todo
type: task
priority: high
created_at: 2026-05-05T15:01:26Z
updated_at: 2026-05-05T15:01:52Z
parent: poker_solver_rust-ohyt
blocked_by:
    - poker_solver_rust-ljo7
---

Order 2. Replace or augment flat mmap slot storage with truly sparse/touched-row storage and avoid full-buffer discount sweeps. Related older bean: poker_solver_rust-zry2.

Implementation notes:
- Allocate per (decision node, bucket) rows on first regret/strategy write rather than reserving physical pages by broad scans.
- Keep reads of untouched rows logically zero/uniform.
- Replace apply_dcfr_discount full-buffer scans with logical per-row discount epochs/factors, applying materialization only when a row is touched.
- Preserve thread safety under parallel MCCFR.
- Provide fallback or compatibility mode for small trees if useful.

Acceptance criteria:
- Training no longer requires scanning every regret/strategy slot for DCFR discount.
- Resident memory tracks touched rows, not full virtual storage.
- Current 6-max config still trains and snapshots/loads correctly.
- Tests cover untouched row reads, concurrent first touch, lazy discount correctness, and snapshot round trip.
- docs/architecture.md documents the storage model.
