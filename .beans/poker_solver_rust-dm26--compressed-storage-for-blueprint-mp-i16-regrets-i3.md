---
# poker_solver_rust-dm26
title: Compressed storage for blueprint_mp — i16 regrets, i32 strategy sums
status: todo
type: task
priority: high
created_at: 2026-04-09T02:22:54Z
updated_at: 2026-04-09T02:22:54Z
---

Halve memory by compressing regret storage from AtomicI32 to AtomicI16 and strategy sums from AtomicI64 to AtomicI32. Reduces 91 GB to ~45 GB for the 6-player 200-bucket config, enabling richer action abstractions.

Key changes:
- AtomicI32 regrets → AtomicI16, reduce REGRET_SCALE from 1000 to ~20
- AtomicI64 strategy sums → AtomicI32
- Update all read/write sites in storage.rs, mccfr.rs, trainer.rs (DCFR discount)
- Update telemetry scan in mp_tui.rs (push_regret_telemetry)
- Validate: regret overflow with i16 range (-32768..32767) at new scale
- Validate: strategy sum overflow with i32 at typical training durations with DCFR gamma=2
