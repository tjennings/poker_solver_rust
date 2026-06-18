---
# poker_solver_rust-lu4f
title: 'retire-hu: remove the HU blueprint_v2 TRAINING path (keep HU bundle loading forever)'
status: todo
type: task
priority: normal
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-18T18:43:36Z
parent: poker_solver_rust-osss
blocked_by:
    - poker_solver_rust-wa53
    - poker_solver_rust-rp9r
    - poker_solver_rust-hoq8
---

Last retirement. Route the train dispatcher's 2-player case to lazy sparse MP; remove the TrainBlueprint training entry + BlueprintTrainer loop + HU-only TUI/audit code not absorbed by tui-unify; docs collapse to ONE trainer section. NEVER remove: blueprint_universal loader LegacyHu+UniversalHu read paths, hu_export, BlueprintV2Strategy reconstruction (Explorer hard requirement) — all historical HU bundles must keep loading. GATE (all must hold): lazy-2p-equivalence GO within tolerance, re-confirmed on a production-representative game; tui-unify covers 2-player with no must-keep HU feature lost; native universal writes proven; eager already retired; any HU-only analysis the Explorer needs (CBV/EV in cbv_compute.rs) reimplemented for N players or explicitly dropped with sign-off. Size: large. From Phase 7 plan 2026-06-18.
