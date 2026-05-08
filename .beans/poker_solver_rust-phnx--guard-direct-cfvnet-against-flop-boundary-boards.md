---
# poker_solver_rust-phnx
title: Guard Direct CFVNet against flop boundary boards
status: scrapped
type: bug
priority: critical
created_at: 2026-05-08T21:01:47Z
updated_at: 2026-05-08T21:02:36Z
---

Direct turn-boundary CFVNet is being evaluated on 3-card flop boards and panics in worker threads. Trace the UI/Tauri solve path, prevent unsupported direct model usage on flop boundaries, and return a clear error or compatible fallback instead of panicking.\n\n- [ ] Reproduce/locate boundary evaluator construction for 3-card boards\n- [ ] Patch validation or evaluator error handling\n- [ ] Add regression coverage\n- [ ] Run focused and full verification\n- [ ] Commit code and bean

## Reasons for Scrapping\n\nUser confirmed the unsupported 3-card board panic was caused by local configuration/use rather than a code defect to fix in this turn.
