# TUI Live Config Reload

**Date**: 2026-03-29
**Status**: Approved

## Problem

Scenario panels and regret audits are configured at startup. To add/remove
panels, you must stop training, edit the YAML, and restart — losing the
current training state unless you snapshot first.

## Design

Press `c` to re-read the YAML config and rebuild scenario + regret audit
panels without stopping training.

### Flow

1. TUI event loop: `KeyCode::Char('c')` sets `config_reload_trigger: Arc<AtomicBool>`
2. Trainer: detects trigger in `check_timed_actions`, calls `reload_tui_config()`
   - Re-reads YAML from disk (trainer has the config path)
   - Re-parses `tui.scenarios` and `tui.regret_audits`
   - Re-resolves spot notations via `resolve_spot()` / `resolve_regret_audit()`
   - Updates `scenario_node_indices`, `scenario_ev_tracker`, callbacks
   - Pushes new state to metrics via `Mutex<Option<ReloadedTuiState>>`
3. TUI render thread: checks mutex each frame, swaps in new panels if present
4. Legend bar updated to show all keybindings

### Legend Bar

With audits: `[p]ause [r]efresh [s]napshot [c]onfig ←/→ scenario ↑/↓ audit [q]uit`
Without:    `[p]ause [r]efresh [s]napshot [c]onfig ←/→ tab [q]uit`

### Files Changed

- `blueprint_tui_metrics.rs` — add `config_reload_trigger` + `reloaded_tui_state` Mutex
- `trainer.rs` — add trigger field, check in timed actions, reload logic
- `main.rs` — extract scenario/audit resolution into reusable function
- `blueprint_tui.rs` — add `c` keybinding, consume reloaded state, update legend

### What Doesn't Change

Training loop is unaffected. Only TUI display and tracking callbacks get swapped.
