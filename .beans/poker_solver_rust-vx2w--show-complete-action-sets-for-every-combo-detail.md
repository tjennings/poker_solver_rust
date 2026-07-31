---
# poker_solver_rust-vx2w
title: Show complete action sets for every combo detail
status: completed
type: bug
priority: high
created_at: 2026-07-29T20:21:11Z
updated_at: 2026-07-29T20:30:04Z
---

Postflop combo detail cards render different visible action sets because frontend/src/GameExplorer.tsx filters combo action rows below 0.5% and bar segments below 0.1%. The backend ComboDetail probabilities are positional vectors aligned to the shared matrix action list, so this is a presentation ambiguity rather than per-combo action legality.

- [x] Render every matrix action row for every visible combo
- [x] Preserve zero-frequency rows with clear 0% values
- [x] Keep the shared action ordering and color mapping
- [x] Add/update frontend regression coverage
- [x] Run focused frontend tests/build
- [x] Document verification and close the bean


## Summary of Changes

Combo detail cards now render every shared matrix action in order, including explicit 0.0% rows. The backend action vectors were unchanged. Added `getComboActionRows` coverage. Verified with `npx vitest run src/GameExplorer.test.tsx` (4 passed) and `npm run build` (success). The test covers the row-building contract rather than mounting the full component; runtime/build verification found no issue.
