---
# poker_solver_rust-4qmh
title: Support spot encoding for Universal MP sessions
status: in-progress
type: bug
priority: high
created_at: 2026-07-30T16:06:49Z
updated_at: 2026-07-30T16:06:49Z
---

game_encode_spot_core and game_load_spot_core only read GameSessionState.session. Universal MP loads into GameSessionState.mp_session, so Copy spot and Load spot return `No game session active` even after game_new succeeds.

- [ ] Add shared spot encoding for legacy and Universal MP session histories
- [ ] Add Universal MP spot replay/reset support for actions and board deals
- [ ] Route core encode/load commands to the active backend
- [ ] Preserve solve invalidation and frontend state behavior
- [ ] Add regression coverage for Universal MP copy/load round trips
- [ ] Run focused Tauri tests and frontend verification
- [ ] Document verification and close the bean
