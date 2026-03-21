---
# poker_solver_rust-x2ap
title: Fix all compiler warnings (16 dead code warnings across 3 crates)
status: in-progress
type: task
created_at: 2026-03-21T06:34:01Z
updated_at: 2026-03-21T06:34:01Z
---

Remove dead code causing 16 compiler warnings:

## poker-solver-core (2 warnings)
- [ ] `actions_match` unused in subgame_cfr.rs:941
- [ ] `showdown_equity` field never read in cfv_subgame_solver.rs:170

## poker-solver-trainer (12 warnings)
- [ ] `dominant_action_color` unused in blueprint_tui_widgets.rs:427
- [ ] All items in log_file.rs unused (TUI_ACTIVE, LOG_FILE, init_log_file, write_to_log, format_timestamp, days_to_ymd, install_panic_hook)
- [ ] `name` field never read in validate_blueprint.rs:6
- [ ] `SpotValidationResult` never constructed in validate_blueprint.rs:15
- [ ] `compute_strategy_l2_distance` unused in validate_blueprint.rs:31
- [ ] `player` field never read in validate_blueprint.rs:171

## cfvnet (2 warnings)
- [ ] `build_turn_game` unused in turn_generate.rs:272
- [ ] `solve_and_extract` unused in turn_generate.rs:319
