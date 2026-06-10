---
# poker_solver_rust-0iye
title: 'Phase 7: trainer/cloud/docs migration'
status: todo
type: task
priority: normal
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-10T16:59:37Z
parent: poker_solver_rust-a29s
---

Roll the universal format into trainer/cloud outputs behind a migration-safe path. Acceptance: trainer writes universal artifacts under a config flag or compatibility window; cloud docs and training/architecture/explorer docs are updated; old HU bundles remain loadable.

## Scope Note (2026-06-10)

Per user guidance on the parent bean: no legacy data migration required. This phase reduces to wiring trainers to write universal bundles natively (plus docs/cloud references); no bulk-conversion or historical-config compatibility work.
