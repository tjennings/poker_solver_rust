---
# poker_solver_rust-vx2w
title: Show complete action sets for every combo detail
status: in-progress
type: bug
priority: high
created_at: 2026-07-29T20:21:11Z
updated_at: 2026-07-29T20:21:11Z
---

Postflop combo detail cards render different visible action sets because frontend/src/GameExplorer.tsx filters combo action rows below 0.5% and bar segments below 0.1%. The backend ComboDetail probabilities are positional vectors aligned to the shared matrix action list, so this is a presentation ambiguity rather than per-combo action legality.

- [ ] Render every matrix action row for every visible combo
- [ ] Preserve zero-frequency rows with clear 0% values
- [ ] Keep the shared action ordering and color mapping
- [ ] Add/update frontend regression coverage
- [ ] Run focused frontend tests/build
- [ ] Document verification and close the bean
