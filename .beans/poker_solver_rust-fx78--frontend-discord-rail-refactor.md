---
# poker_solver_rust-fx78
title: Frontend Discord Rail Refactor
status: completed
type: feature
priority: normal
created_at: 2026-03-05T19:05:24Z
updated_at: 2026-03-05T19:15:31Z
---

Replace tab-based navigation with Discord-like left rail, modernize theme with gradients/shadows, simplify Explore dataset loading flow. Plan: docs/plans/2026-03-05-frontend-discord-rail.md

## Summary of Changes

- Replaced tab-based navigation with Discord-like 64px left rail (compass, flask, swords, gear icons)
- Settings icon pushed to bottom via flex spacer
- Active icon gets cyan pill indicator and tinted background
- Created stub Train and Settings views ("Coming soon" placeholders)
- Removed hamburger menu from Explorer entirely
- Added "Load Dataset" action card in the action strip when no dataset is loaded
- Modernized theme: body gradient, panel gradient backgrounds, box shadows, softer borders
- All panels/cards upgraded from flat #16213e to gradient+shadow treatment
- 5 files changed, TypeScript clean, build passes
