---
# poker_solver_rust-6op8
title: 'Phase 0: universal dense format spec'
status: completed
type: task
priority: high
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-09T20:02:19Z
parent: poker_solver_rust-a29s
---

Write the committed schema spec for the universal dense blueprint format. Acceptance: manifest fields, binary file headers, row/action descriptor structs, checksums/fingerprints, compatibility policy, missing-row policy, and explicit non-goals are documented in repo docs. No production writer required.

## Summary of Changes

- Added `docs/blueprint_format.md` as the Phase 0 universal dense blueprint format specification.
- Documented manifest fields, binary payload headers, row and action descriptors, probability normalization, bucket metadata, checksums, fingerprints, compatibility policy, missing-row policy, reader/writer architecture, validation plan, and explicit non-goals.
- Linked the planned format from architecture, training, and Explorer docs.
- Left production writers/readers unchanged; Phase 0 is spec-only.
