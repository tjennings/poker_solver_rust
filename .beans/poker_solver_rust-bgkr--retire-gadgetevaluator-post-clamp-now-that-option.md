---
# poker_solver_rust-bgkr
title: Retire GadgetEvaluator (post-clamp) now that Option A ships
status: todo
type: task
priority: low
created_at: 2026-04-24T06:55:28Z
updated_at: 2026-04-24T06:55:55Z
blocked_by:
    - poker_solver_rust-02wj
---

Bean lay5's post-clamp GadgetEvaluator is retained in crates/tauri-app/src/gadget.rs for A/B diagnostic comparison via --gadget-clamp CLI flag during Option A rollout. Once Option A stabilises and diagnostic need passes, delete:

- GadgetEvaluator struct + BoundaryEvaluator impl
- apply_gadget_clamp helper
- diag_fire / set_gadget_diag_stride / set_gadget_diag_enabled
- GADGET_DIAG_COUNTS / GADGET_DIAG_STRIDE static state
- --gadget-clamp CLI flag + routing in compare_solve.rs
- setup_clamp_boundaries in compare_solve.rs
- compute_opt_out_provider in compare_solve.rs
- The entire clamp-related section of docs/training.md

Keep:
- OptOutProvider trait + ConstantOptOut + BlueprintCbvOptOut (used by Option A)
- chip_cfv_to_bcfv helper
- StaticGadgetEvaluator / make_gadget_game (Option A public surface)

Blocked-by poker_solver_rust-02wj (Option A ship).
Trigger: 2+ weeks post-ship with no regressions reported, and iter-15 harness result indicates no further A/B diagnostic needed.
