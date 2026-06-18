---
# poker_solver_rust-0iye
title: 'Phase 7: trainer/cloud/docs migration'
status: todo
type: task
priority: normal
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-18T18:43:36Z
parent: poker_solver_rust-a29s
---

Roll the universal format into trainer/cloud outputs behind a migration-safe path. Acceptance: trainer writes universal artifacts under a config flag or compatibility window; cloud docs and training/architecture/explorer docs are updated; old HU bundles remain loadable.

## Scope Note (2026-06-10)

Per user guidance on the parent bean: no legacy data migration required. This phase reduces to wiring trainers to write universal bundles natively (plus docs/cloud references); no bulk-conversion or historical-config compatibility work.

## Scope Note (2026-06-10 goal recap)

Reframed: ONE training workflow (lazy sparse, 2-10 players) and ONE TUI. This phase is not just 'trainer writes universal natively' — it is the consolidation point where HU blueprint_v2 training, the MP eager dense backend, and separate TUIs get retired rather than migrated. Needs its own planning pass before implementation.

## Phase 7 plan decomposition (2026-06-18)

0iye = the two near-term, retires-nothing slices (Phase A of the consolidation):
- **0iye-core** (medium): add snapshot.format = legacy|universal|both (default legacy) to BlueprintV2Config.SnapshotConfig + BlueprintMpConfig.MpSnapshotConfig; wire save_snapshot / save_mp_snapshot / save_lazy_mp_snapshot to also emit a universal bundle via the existing in-memory exporters. Native bytes must be byte-identical to running export-universal/-mp post-hoc on the same snapshot. Explorer loads every natively-written bundle. Legacy output stays. (Lazy native write still emits Opaque actions until mt3l.)
- **0iye-docs-cloud** (small): add a 'train' subcommand that auto-detects HU-V2 vs MP config and dispatches (NOTE: cloud user-data.sh.tpl:43 already calls a 'train' subcommand that does NOT exist today — real bug to fix); keep train-blueprint/train-blueprint-mp as aliases; update docs/cloud.md, training.md, blueprint_format.md.

Full consolidation roadmap (Phases B/C) tracked in sibling beans: nbzo (seat re-layout) -> 8jan (widen 2-10), tzv5 (finish runtime seam), mt3l (action identity), wa53 (lazy 2p equivalence GO/NO-GO), rp9r (one TUI), hoq8 (retire eager), lu4f (retire HU). Retirements are gated/incremental; HU+eager LOAD paths are never removed (Explorer).
