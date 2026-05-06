---
# poker_solver_rust-xbmw
title: 'Clarify docs: preflop is lossless, bucketing starts postflop'
status: todo
type: task
priority: low
created_at: 2026-05-06T16:39:36Z
updated_at: 2026-05-06T16:39:36Z
parent: poker_solver_rust-hfnv
---

Docs currently describe the potential-aware pipeline as river→turn→flop→preflop EMD, but the intended solver behavior is lossless canonical hand mapping preflop and lossy potential-aware bucketing beginning on the flop.

Acceptance:
- Update docs/architecture.md and docs/training.md to state preflop uses canonical 169-hand mapping
- Clarify that postflop bucket counts control flop/turn/river abstraction quality
- Ensure sample configs and CLI output do not imply preflop EMD bucketing
- Keep the audit correction reflected in docs
