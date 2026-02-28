---
# poker_solver_rust-x64v
title: Make postflop_model_path optional in PreflopTrainingConfig
status: scrapped
type: bug
priority: normal
created_at: 2026-02-28T02:58:06Z
updated_at: 2026-02-28T02:59:16Z
---

solve-preflop fails when config doesn't include postflop_model_path. The field should be Option<PathBuf> since the preflop solver already handles None postflop gracefully (falls back to raw equity).

## Reasons for Scrapping\n\nUser requested rollback.
