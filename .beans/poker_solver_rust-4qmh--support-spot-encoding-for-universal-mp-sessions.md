---
# poker_solver_rust-4qmh
title: Support spot encoding for Universal MP sessions
status: completed
type: bug
priority: high
created_at: 2026-07-30T16:06:49Z
updated_at: 2026-07-30T16:37:08Z
---

game_encode_spot_core and game_load_spot_core only read GameSessionState.session. Universal MP loads into GameSessionState.mp_session, so Copy spot and Load spot return `No game session active` even after game_new succeeds.

- [x] Add shared spot encoding for legacy and Universal MP session histories
- [x] Add Universal MP spot replay/reset support for actions and board deals
- [x] Route core encode/load commands to the active backend
- [x] Preserve solve invalidation and frontend state behavior
- [x] Add regression coverage for Universal MP copy/load round trips
- [x] Run focused Tauri tests and frontend verification
- [x] Document verification and close the bean


## Summary of Changes

Universal MP sessions now share spot encoding with legacy HU sessions. Copy/load routes to the active `mp_session` or legacy `session`; MP replay resets to root, replays position-labeled actions, and deals board segments while invalidating stale solve state. Added shared formatter coverage and a live Universal MP copy/load round-trip test. Verified with the targeted integration test (1 passed) and `cargo test -p poker-solver-tauri --lib spot -- --nocapture` (38 passed). Updated docs/explorer.md.
