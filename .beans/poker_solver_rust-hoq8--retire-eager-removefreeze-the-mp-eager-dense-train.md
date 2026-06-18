---
# poker_solver_rust-hoq8
title: 'retire-eager: remove/freeze the MP eager dense training backend'
status: todo
type: task
priority: normal
created_at: 2026-06-18T18:43:16Z
updated_at: 2026-06-18T18:43:36Z
parent: poker_solver_rust-osss
blocked_by:
    - poker_solver_rust-0iye
    - poker_solver_rust-8jan
    - poker_solver_rust-rp9r
---

First retirement (lower blast radius than HU). Make lazy_sparse the default/only MP TRAINING backend; remove or demote-to-read-only-diagnostic the eager training path (run_training/MpStorage materialization); update inspect-mp-config messaging. KEEP the mp_eager_export + loader UniversalMpEager READ path forever so existing eager bundles still load in the Explorer. GATE (all must hold): lazy serves 2-10 (8jan widen done), native universal writes proven for lazy (0iye), one TUI covers MP (tui-unify), AND the lazy-resume gap is reconciled — lazy resume implemented OR resume formally declared unsupported (eager is today's ONLY resumable MP path; see reject_lazy_sparse_resume at main.rs:2911/3501). Size: medium. From Phase 7 plan 2026-06-18.
