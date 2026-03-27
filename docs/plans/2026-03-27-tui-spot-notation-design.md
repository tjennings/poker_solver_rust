# TUI Spot Notation Design

**Date:** 2026-03-27
**Status:** Approved

## Goal

Replace the TUI scenario config format with the Tauri GameSession spot notation, unifying how action paths are specified across the codebase.

## Current Format

```yaml
scenarios:
  - name: "SB Cbet K72"
    player: SB
    actions: ["raise-0", "call"]
    board: ["Kh", "7s", "2d"]
    street: flop
```

Uses semantic action names (`raise-0`, `fold`, `call`) with separate `player`, `board`, and `street` fields.

## New Format

```yaml
scenarios:
  - name: "SB Cbet K72"
    spot: "sb:4bb,bb:call|Kh7s2d"
```

Uses Tauri's spot notation: `position:label` pairs separated by commas, `|` separating board card segments.

## Label Format

Action labels use Tauri's BB-based format:
- `fold`, `check`, `call`, `all-in` for fixed actions
- `{chips/2}bb` for bets/raises (e.g., `4bb` for a bet of 8 chips)

Matching is case-insensitive.

## Parsing

`resolve_spot(tree, spot)` splits on `|`, identifies action segments (contain `:`) vs board segments (no `:`), walks the tree matching labels against available actions at each decision node, and returns `(node_idx, board_cards)`.

## Files Changed

1. `blueprint_tui_config.rs` — simplify `ScenarioConfig` to `{name, spot}`
2. `blueprint_tui_scenarios.rs` — add `resolve_spot()`, `format_tree_action_bb()`; `resolve_action_path` removed
3. `main.rs` — update scenario init to use `resolve_spot`
4. `blueprint_v2_200bkt_sapcfr.yaml` — update example scenarios
5. Tests updated for new format
