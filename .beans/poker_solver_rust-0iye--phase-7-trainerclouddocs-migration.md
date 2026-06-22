---
# poker_solver_rust-0iye
title: 'Phase 7: trainer/cloud/docs migration'
status: completed
type: task
priority: normal
created_at: 2026-06-09T18:40:56Z
updated_at: 2026-06-22T14:47:54Z
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

## Summary of Changes

Trainers now write universal dense bundles natively at snapshot time, behind a `snapshot.format` config flag (legacy | universal | both, default legacy via serde, so existing configs are unaffected). Merged across the Phase 7 0iye merge (+ post-merge fixes).

- All three backends wired identically to HU gating (write_legacy={legacy,both}, write_universal={universal,both}; universal-only drops legacy strategy files and writes the bundle): HU (trainer save_snapshot), MP eager (save_mp_snapshot), MP lazy (save_lazy_mp_snapshot) + both MP TUI bridges (bridge_mp_iterations / bridge_mp_lazy_iterations). Each reuses the existing in-memory exporter + write_bundle + retain_config_yaml (no IO duplication).
- Native output is byte-identical to the post-hoc export-universal/-mp path (manifests modulo the per-run created_at timestamp), proven by per-backend integration tests AND per-backend tests through the actual save_*_snapshot functions (universal-only asserts legacy files absent + bundle loads).
- New `train` subcommand auto-detects HU-V2 vs MP config (game.players vs game.num_players) and dispatches to the correct trainer; train-blueprint/train-blueprint-mp kept as aliases. Fixes a latent bug: cloud/user-data.sh.tpl invoked a `train` subcommand that did not exist.
- Docs updated: training.md (snapshot.format + train dispatcher), cloud.md, blueprint_format.md, architecture.md.

Retires nothing; post-hoc export commands and legacy output retained; Explorer loads every natively-written bundle.

## Process note
A required `format` field added to SnapshotConfig broke 3 tauri-app test-context initializers — missed by the scoped (-p core -p trainer) reviews and by `cargo build --workspace` (which does not compile test targets). Caught at integration via `cargo build --workspace --tests` and fixed (commit 7117dd6b). Full --tests build now clean; warm suite 49.65s this run (variable; suite-runtime fragility tracked in z8jx).

## Deferred (tracked elsewhere)
- Lazy native write still emits Opaque actions (mt3l).
- Stage B/C consolidation (2-10 players nbzo/8jan, runtime seam tzv5, one TUI rp9r, lazy-2p equivalence wa53, retirements hoq8/lu4f).
