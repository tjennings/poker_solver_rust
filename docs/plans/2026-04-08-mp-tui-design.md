# N-Player Blueprint TUI — Design Document

**Date:** 2026-04-08
**Status:** Approved

## Overview

Adapt the blueprint trainer TUI for N-player support. Three changes: position-aware spot encoding, dynamic position labels, and a 6-up tiled strategy grid layout with pagination. The existing `blueprint_v2` TUI is untouched — this is wired to the `train-blueprint-mp` code path.

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Grid layout | 6 per page, tabbed after 6 | See all positions at a glance; paginate for more |
| Compact grid | 4 chars × 1 row per cell, half-block rendering | Fits 6 grids in standard terminal |
| Spot encoding | Position labels (utg:5bb) | N-player positions replace hardcoded sb/bb |
| Position field in config | Not needed | Resolver determines acting player from tree walk |
| Backward compat | Existing v2 TUI untouched | Separate code path for blueprint_mp |

## 1. Position Labels

Function in `blueprint_mp/types.rs`:

```rust
pub fn position_label(seat: Seat, num_players: u8) -> &'static str
```

Mapping (SB=seat 0, BB=seat 1, remaining fill rightward):

| Players | Labels |
|---------|--------|
| 2 | btn, bb |
| 3 | btn, sb, bb |
| 4 | co, btn, sb, bb |
| 5 | hj, co, btn, sb, bb |
| 6 | utg, hj, co, btn, sb, bb |
| 7 | utg, mp, hj, co, btn, sb, bb |
| 8 | utg, utg1, mp, hj, co, btn, sb, bb |

Reverse: `parse_position(label, num_players) -> Option<Seat>`

## 2. Spot Encoding

**Format:** `"position:action,position:action|street|position:action,..."`

**Examples:**
- UTG opens: `""` (root node, UTG is first to act)
- BTN vs UTG open: `"utg:5bb,hj:fold,co:fold"`
- BB vs BTN open: `"utg:fold,hj:fold,co:fold,btn:5bb,sb:fold"`
- 3-bet pot flop: `"utg:5bb,hj:fold,co:fold,btn:3x,sb:fold,bb:fold|utg:check"`

The spot resolver walks the game tree following the action sequence, mapping position labels to seat indices via `parse_position()`. The decision node at the end of the walk determines whose turn it is (via `seat` field).

## 3. Compact Grid Widget

**Cell:** 4 chars wide × 1 row tall. Unicode half-block (`▐`) for sub-character color resolution (8 color slots per cell).

**Color scheme** (same as current TUI):
- Fold: gray RGB(140, 140, 140)
- Check: blue RGB(80, 140, 220)
- Call: green RGB(60, 179, 113)
- Bet/Raise: red gradient by size (darkest = largest)
- All-in: purple RGB(180, 30, 180)

**Grid dimensions:** ~55 cols × 15 rows (1 title + 1 header + 13 data rows)
- Row labels: single column, rank character (A, K, Q, ... 2)
- Column headers: rank characters
- Title: position label + scenario name

**Convergence indicator:** Grid frame border color (black = converged, blue = moving).

## 4. Page Layout

```
┌─────────────────────────────────────────────────────┐
│  Metrics (sparklines, throughput, exploitability)    │
├──────────────────┬──────────────────┬───────────────┤
│  UTG open        │  HJ vs UTG      │  CO vs UTG    │
│  [13x13 grid]    │  [13x13 grid]   │  [13x13 grid] │
├──────────────────┼──────────────────┼───────────────┤
│  BTN vs UTG      │  SB vs UTG      │  BB vs UTG    │
│  [13x13 grid]    │  [13x13 grid]   │  [13x13 grid] │
├─────────────────────────────────────────────────────┤
│  [Page 1/2]  ← → navigate  q quit  p pause         │
└─────────────────────────────────────────────────────┘
```

- **6 grids per page** (constant, configurable)
- **Auto-layout:** `grids_per_row = terminal_width / grid_width`, wrap remaining
- **Pagination:** Left/Right arrows switch pages. Indicator in hotkey bar.
- **Scenarios > 6:** Flow to page 2, 3, etc. (12 scenarios = 2 pages)

## 5. Config

Same `scenarios:` config block, just uses position labels in spot encoding:

```yaml
tui:
  enabled: true
  refresh_rate_ms: 250
  scenarios:
    - name: "UTG open"
      spot: ""
    - name: "HJ vs UTG"
      spot: "utg:5bb"
    - name: "CO vs UTG"
      spot: "utg:5bb,hj:fold"
    - name: "BTN vs UTG"
      spot: "utg:5bb,hj:fold,co:fold"
    - name: "SB vs UTG"
      spot: "utg:5bb,hj:fold,co:fold,btn:fold"
    - name: "BB vs UTG"
      spot: "utg:5bb,hj:fold,co:fold,btn:fold,sb:fold"
```

## 6. Integration

- `PlayerLabel` enum removed — use `Seat` + `position_label()` dynamically
- `BlueprintTuiMetrics` gains `num_players: u8` field
- Spot resolver uses `parse_position()` for position label → seat mapping
- Grid refresh callback produces `[[CellStrategy; 13]; 13]` per scenario (unchanged data)
- Compact `HandGridWidget` renders each grid using half-block technique
- Metrics panel unchanged
- `CellStrategy` struct unchanged
- Training loop → metrics push pattern unchanged
- Existing `blueprint_v2` TUI untouched
