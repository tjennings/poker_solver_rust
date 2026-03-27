# Spot Encoding System — Design

**Date:** 2026-03-27

## Problem

No canonical way to refer to a specific game state. Users can't share spots for debugging or quickly navigate to a specific position.

## Encoding Format

Human-readable, pipe-separated segments alternating between actions and board cards:

```
sb:2bb,bb:7bb,sb:call|AhKdQc|bb:check,sb:4bb,bb:call|7s|sb:check,bb:10bb
```

Rules:
- Actions: `position:label` comma-separated. Position is lowercased (`sb`, `bb`). Label is lowercased during encoding (e.g. `fold`, `check`, `call`, `4bb`, `all-in`). Matching during parsing is case-insensitive.
- A segment is a board deal if it contains no `:` character; otherwise it's actions.
- Board segments are split into 2-char chunks and dealt one card at a time via `deal_card`.
- After an all-in, remaining streets appear as consecutive board segments with no intervening action segments.
- Board cards: 2-char cards concatenated (3 for flop, 1 for turn, 1 for river)
- `|` separates street transitions
- First segment is always preflop actions
- Empty trailing segment or ending at a card deal = stopped there
- Action matching: exact match on position and label. Error with available actions listed if no match.

## Example Spots

### Preflop Only

```
sb:2bb,bb:fold
sb:2bb,bb:7bb,sb:fold
sb:2bb,bb:7bb,sb:15bb,bb:call
```

### Preflop → Flop (first action)

```
sb:2bb,bb:call|Ah7d2c
sb:2bb,bb:7bb,sb:call|KhQhJh
sb:fold
```

### Flop Actions

```
sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb,bb:call
sb:2bb,bb:call|Ah7d2c|bb:check,sb:check
sb:2bb,bb:7bb,sb:call|KsQd3c|bb:10bb,sb:all-in,bb:call
```

### Flop → Turn

```
sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb,bb:call|Kh
sb:2bb,bb:call|Ah7d2c|bb:check,sb:check|3s
sb:2bb,bb:7bb,sb:call|KhQhJh|bb:5bb,sb:call|2c
```

### Turn Actions

```
sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb,bb:call|Kh|bb:check,sb:10bb,bb:fold
sb:2bb,bb:call|Ah7d2c|bb:check,sb:check|3s|bb:5bb,sb:call
sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb,bb:call|Kh|bb:check,sb:check
```

### Turn → River

```
sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb,bb:call|Kh|bb:check,sb:check|Qc
sb:2bb,bb:call|Ah7d2c|bb:check,sb:check|3s|bb:5bb,sb:call|Jd
sb:2bb,bb:7bb,sb:call|KhQhJh|bb:5bb,sb:call|2c|bb:8bb,sb:call|4s
```

### River Actions

```
sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb,bb:call|Kh|bb:check,sb:check|Qc|bb:10bb,sb:fold
sb:2bb,bb:call|Ah7d2c|bb:check,sb:check|3s|bb:5bb,sb:call|Jd|bb:check,sb:21bb,bb:all-in,sb:call
sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb,bb:call|Kh|bb:check,sb:10bb,bb:call|Qc|bb:check,sb:check
```

### All-in Runouts

```
sb:2bb,bb:7bb,sb:all-in,bb:call|AhKdQc|7s|2d
sb:2bb,bb:call|Td9d6h|bb:all-in,sb:call|Kh|Qc
sb:2bb,bb:7bb,sb:call|KhQhJh|bb:10bb,sb:all-in,bb:call|2c|4s
```

## Backend

### GameSession methods (shared by Tauri, devserver, CLI)

```rust
/// Encode current game state as a spot string.
pub fn encode_spot(&self) -> String

/// Parse a spot string and replay to that state.
/// Resets to preflop root (including weights and board), then replays
/// each action and card deal. Errors with available actions if no match.
pub fn load_spot(&mut self, spot: &str) -> Result<(), String>
```

### Tauri commands + core functions

```rust
game_encode_spot() -> String
game_load_spot(spot: String) -> GameState
```

### Encoding logic (encode_spot)

Walk `action_history`, track street changes. For each action, emit `{position}:{label}` (lowercased position). When street changes, emit `|` + board cards for that street.

### Parsing logic (load_spot)

1. Split on `|` into segments
2. For each segment, determine if it's actions or board cards (board cards = 2-char patterns without `:`)
3. For action segments: split on `,`, parse `position:label`, find matching action in current `GameState.actions`, call `play_action`
4. For board segments: parse card strings, call `deal_card` for each
5. On error: return `"Action 'sb:4bb' not found. Available: sb:check, sb:3bb, sb:5bb, sb:all-in"`

## Frontend

### Vertical toolbar (left of action strip)

Two icon buttons stacked vertically:
- **Copy** (clipboard icon) — calls `game_encode_spot`, copies to clipboard, brief "Copied!" toast
- **Load** (upload icon) — opens modal

### Blueprint summary card (after toolbar, before action cards)

- Blueprint name (from bundleName)
- Stack depth (e.g. "50bb")
- Click "Change" to go back to blueprint picker

### Load modal

Simple modal:
- Text input (paste spot encoding)
- "Load" button — calls `game_load_spot`, closes modal on success, shows error inline on failure
- "Cancel" button

## CLI: `inspect-spot` Command

```bash
cargo run -p poker-solver-trainer --release -- inspect-spot \
  -c sample_configurations/blueprint_v2_1kbkt.yaml \
  --spot "sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb"
```

Loads the blueprint, parses the spot encoding using the SAME `load_spot` code as Tauri, then dumps all available data for that decision point:

### Output

```
Spot: sb:2bb,bb:call|Td9d6h|bb:check,sb:4bb
Street: Flop
Board: Td 9d 6h
Position: BB (to act)
Pot: 12BB | Stacks: BB 44BB / SB 44BB

=== Strategy (169 hands) ===
Hand     Fold    Call    Raise   EV
AA       0.0%    72.3%   27.7%  +4.2
AKs      0.0%    85.1%   14.9%  +3.1
AKo      12.3%   81.2%    6.5%  +1.8
...

=== Top Actions ===
Fold:  32.1% of range
Call:  51.4% of range
Raise: 16.5% of range

=== Equity vs Opponent Range ===
Range equity: 48.2%
Nut advantage: +3.1%

=== Combo Details (selected) ===
AhKh: Fold 0.0% | Call 91.2% | Raise 8.8% | EV +5.1
Td9d: Fold 0.0% | Call 12.3% | Raise 87.7% | EV +8.9
7s6s: Fold 45.1% | Call 54.9% | Raise 0.0% | EV -1.2
```

### Implementation

The CLI uses:
1. `GameSession::from_exploration_state` (same as Tauri)
2. `GameSession::load_spot` (exact same parsing code)
3. `GameSession::get_state` for matrix/actions/EVs
4. Formatting is CLI-only (table output)

The core spot encoding/decoding lives in `game_session.rs` — shared across Tauri, devserver, and CLI with zero duplication.

## Files to Change

**Phase 1 (Tauri + devserver):**
- `crates/tauri-app/src/game_session.rs` — encode_spot, load_spot methods + Tauri commands
- `crates/devserver/src/main.rs` — devserver endpoints
- `frontend/src/GameExplorer.tsx` — toolbar, summary card, copy/load UI

**Phase 2 (CLI):**
- `crates/trainer/src/main.rs` — `inspect-spot` subcommand
- Reuses `game_session.rs` encode/load logic directly
