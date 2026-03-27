# Spot Encoding System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add human-readable spot encoding/decoding to GameSession, with copy/load UI in GameExplorer and a CLI `inspect-spot` command.

**Architecture:** `encode_spot()` and `load_spot()` methods on `GameSession` in `game_session.rs` — shared by Tauri, devserver, and CLI. Frontend adds a toolbar with copy/load buttons. CLI `inspect-spot` subcommand reuses the same backend code.

**Tech Stack:** Rust (GameSession backend), TypeScript/React (frontend), clap (CLI)

---

## Phase 1: Backend + Frontend

### Task 1: Implement `encode_spot` on GameSession

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Write tests**

```rust
#[test]
fn encode_spot_preflop_fold() {
    // After SB folds, encoding should be "sb:fold"
}

#[test]
fn encode_spot_preflop_to_flop() {
    // After preflop actions + flop deal, encoding should include board cards
    // e.g. "sb:2bb,bb:call|Td9d6h"
}

#[test]
fn encode_spot_empty_history() {
    // At preflop root with no actions, encoding should be ""
}
```

Note: these tests need a GameSession with actions played. Use the existing `make_test_session()` helper if available, or construct a minimal one. The test should call `play_action` to build history, then verify `encode_spot()` output.

**Step 2: Implement `encode_spot`**

Add to `GameSession` impl block (after `back()`, around line 670):

```rust
/// Encode the current game state as a human-readable spot string.
///
/// Format: `sb:2bb,bb:call|AhKdQc|bb:check,sb:4bb`
/// - Actions are `position:label` (lowercased), comma-separated
/// - `|` separates street transitions (board card deals)
/// - Board segments are card strings concatenated (e.g. "AhKdQc")
pub fn encode_spot(&self) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut current_actions: Vec<String> = Vec::new();
    let mut prev_street = String::new();
    let mut board_idx = 0; // tracks which board cards we've emitted

    for rec in &self.action_history {
        // When street changes, emit board cards for the transition
        if rec.street != prev_street && !prev_street.is_empty() {
            // Flush current actions
            if !current_actions.is_empty() {
                parts.push(current_actions.join(","));
                current_actions.clear();
            }
            // Emit board cards for this transition
            let cards_for_street = match prev_street.as_str() {
                "Preflop" => &self.board[board_idx..board_idx.min(self.board.len()).max(board_idx)
                    .min(board_idx + 3)],
                _ => {
                    let end = (board_idx + 1).min(self.board.len());
                    &self.board[board_idx..end]
                }
            };
            // Simpler: emit based on how many new cards appeared
            let new_cards = match prev_street.as_str() {
                "Preflop" => 3, // flop = 3 cards
                _ => 1,         // turn/river = 1 card
            };
            let end = (board_idx + new_cards).min(self.board.len());
            let board_str: String = self.board[board_idx..end].join("");
            board_idx = end;
            parts.push(board_str);
        }
        prev_street = rec.street.clone();
        current_actions.push(format!("{}:{}", rec.position.to_lowercase(), rec.label.to_lowercase()));
    }

    // Flush remaining actions
    if !current_actions.is_empty() {
        parts.push(current_actions.join(","));
    }

    // If we're at a chance node (board cards dealt but no actions on new street),
    // emit the remaining board cards
    if board_idx < self.board.len() {
        let remaining: String = self.board[board_idx..].join("");
        parts.push(remaining);
    }

    parts.join("|")
}
```

**Step 3: Run tests, commit**

```bash
cargo test -p poker-solver-tauri -- encode_spot
git commit -m "feat: encode_spot produces human-readable spot strings"
```

---

### Task 2: Implement `load_spot` on GameSession

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`

**Step 1: Write tests**

```rust
#[test]
fn load_spot_empty_string() {
    // Empty string should be a no-op (stay at preflop root)
}

#[test]
fn load_spot_invalid_action_errors() {
    // "sb:invalid" should error with available actions listed
}

#[test]
fn load_spot_board_segment_parsed() {
    // "sb:2bb,bb:call|Td9d6h" should deal 3 flop cards
}
```

**Step 2: Implement `load_spot`**

Add to `GameSession` impl block:

```rust
/// Parse a spot encoding and replay to that state.
///
/// Resets to preflop root (including weights, board, action history),
/// then replays each action and board card deal from the encoding.
pub fn load_spot(&mut self, spot: &str) -> Result<(), String> {
    let spot = spot.trim();
    if spot.is_empty() {
        return Ok(());
    }

    // Reset to root
    self.node_idx = self.tree.root;
    self.board.clear();
    self.action_history.clear();
    self.weights = [vec![1.0f32; 1326], vec![1.0f32; 1326]];

    let segments: Vec<&str> = spot.split('|').collect();

    for segment in segments {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }

        if segment.contains(':') {
            // Action segment: "sb:2bb,bb:call"
            let actions: Vec<&str> = segment.split(',').collect();
            for action_str in actions {
                let action_str = action_str.trim();
                let (pos, label) = action_str.split_once(':')
                    .ok_or_else(|| format!("Invalid action format: '{action_str}'. Expected 'position:label'"))?;

                // Get current state to find matching action
                let state = self.get_state();
                let position = state.position.to_lowercase();
                if pos.to_lowercase() != position {
                    return Err(format!(
                        "Position mismatch: '{pos}' but current position is '{position}'"
                    ));
                }

                // Find matching action by label (case-insensitive)
                let matched = state.actions.iter().find(|a| {
                    a.label.to_lowercase() == label.to_lowercase()
                });

                match matched {
                    Some(action) => {
                        self.play_action(&action.id)?;
                    }
                    None => {
                        let available: Vec<String> = state.actions.iter()
                            .map(|a| format!("{}:{}", position, a.label.to_lowercase()))
                            .collect();
                        return Err(format!(
                            "Action '{}:{}' not found. Available: {}",
                            pos, label, available.join(", ")
                        ));
                    }
                }
            }
        } else {
            // Board segment: "AhKdQc" or "7s" or "2d"
            // Split into 2-char chunks
            let chars: Vec<char> = segment.chars().collect();
            if chars.len() % 2 != 0 {
                return Err(format!("Invalid board segment: '{segment}'. Must be pairs of rank+suit."));
            }
            for chunk in chars.chunks(2) {
                let card: String = chunk.iter().collect();
                self.deal_card(&card)?;
            }
        }
    }

    Ok(())
}
```

**Step 3: Run tests, commit**

```bash
cargo test -p poker-solver-tauri -- load_spot
git commit -m "feat: load_spot parses spot encoding and replays to state"
```

---

### Task 3: Tauri commands and devserver endpoints

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (commands + core functions)
- Modify: `crates/tauri-app/src/lib.rs` (export)
- Modify: `crates/tauri-app/src/main.rs` (register)
- Modify: `crates/devserver/src/main.rs` (endpoints)

**Step 1: Add core functions and Tauri commands**

```rust
pub fn game_encode_spot_core(session_state: &GameSessionState) -> Result<String, String> {
    let guard = session_state.session.read();
    let session = guard.as_ref().ok_or("No game session active")?;
    Ok(session.encode_spot())
}

pub fn game_load_spot_core(session_state: &GameSessionState, spot: &str) -> Result<GameState, String> {
    let mut guard = session_state.session.write();
    let session = guard.as_mut().ok_or("No game session active")?;
    session.load_spot(spot)?;
    Ok(session.get_state())
}

#[tauri::command]
pub fn game_encode_spot(
    session_state: tauri::State<'_, GameSessionState>,
) -> Result<String, String> {
    game_encode_spot_core(&session_state)
}

#[tauri::command]
pub fn game_load_spot(
    session_state: tauri::State<'_, GameSessionState>,
    spot: String,
) -> Result<GameState, String> {
    game_load_spot_core(&session_state, &spot)
}
```

**Step 2: Register in lib.rs, main.rs, devserver**

Follow the existing pattern for `game_solve` / `game_cancel_solve`:
- `lib.rs`: add to `pub use game_session::{ ... }` block
- `main.rs`: add to `invoke_handler` list
- `devserver/main.rs`: add handler functions + routes

**Step 3: Build, test, commit**

```bash
cargo build -p poker-solver-tauri -p poker-solver-devserver
cargo test -p poker-solver-tauri
git commit -m "feat: game_encode_spot and game_load_spot commands"
```

---

### Task 4: Frontend — vertical toolbar + blueprint summary

**Files:**
- Modify: `frontend/src/GameExplorer.tsx`

**Step 1: Add toolbar and summary card**

In the action strip, before the action cards, add:

1. **Vertical toolbar** — 2 icon buttons stacked:
   - Copy (clipboard icon SVG) — calls `game_encode_spot`, writes to clipboard
   - Load (upload icon SVG) — opens load modal

2. **Blueprint summary card** — shows `bundleName`, stack depth, "Change" button

The toolbar replaces the current Back/New/bundleName button cluster. The blueprint summary card goes between the toolbar and the action cards.

**Step 2: Add copy handler**

```tsx
const copySpot = async () => {
    try {
        const spot = await invoke<string>('game_encode_spot', {});
        await navigator.clipboard.writeText(spot);
        // Brief visual feedback (e.g. change icon briefly)
    } catch (e) {
        setError(String(e));
    }
};
```

**Step 3: Build, verify layout**

```bash
cd frontend && npx tsc --noEmit
```

**Step 4: Commit**

```
git commit -m "feat: vertical toolbar with copy spot button + blueprint summary card"
```

---

### Task 5: Frontend — load spot modal

**Files:**
- Modify: `frontend/src/GameExplorer.tsx`

**Step 1: Add modal state and component**

```tsx
const [showLoadModal, setShowLoadModal] = useState(false);
const [loadSpotInput, setLoadSpotInput] = useState('');
const [loadSpotError, setLoadSpotError] = useState<string | null>(null);

const loadSpot = async () => {
    try {
        setLoadSpotError(null);
        const s = await invoke<GameState>('game_load_spot', { spot: loadSpotInput });
        setState(s);
        setShowLoadModal(false);
        setLoadSpotInput('');
    } catch (e) {
        setLoadSpotError(String(e));
    }
};
```

**Step 2: Render modal**

A simple overlay modal with:
- Text input (monospace font, full width)
- Error display (red, inline)
- Load + Cancel buttons

Match the existing dark theme (background `#1a1a2e`, borders `#334155`).

**Step 3: Wire load button in toolbar to open modal**

**Step 4: Build, test, commit**

```bash
cd frontend && npx tsc --noEmit
git commit -m "feat: load spot modal with text input and error display"
```

---

## Phase 2: CLI `inspect-spot`

### Task 6: Add `inspect-spot` subcommand to trainer CLI

**Files:**
- Modify: `crates/trainer/src/main.rs` (or wherever CLI subcommands are defined)
- Reference: `crates/tauri-app/src/game_session.rs` (reuse GameSession)

**Step 1: Add clap subcommand**

```rust
#[derive(clap::Args)]
struct InspectSpotArgs {
    /// Path to blueprint config YAML
    #[arg(short, long)]
    config: String,

    /// Spot encoding string
    #[arg(long)]
    spot: String,
}
```

**Step 2: Implement handler**

The handler:
1. Loads the blueprint config, builds the V2 tree/strategy (same as `load_blueprint_v2_core`)
2. Creates a `GameSession` from the loaded state
3. Calls `session.load_spot(&args.spot)`
4. Calls `session.get_state()` to get the matrix, actions, EVs
5. Formats and prints the output table

Key output sections:
- Spot metadata: street, board, position, pot, stacks
- Strategy table: 13x13 hands with action frequencies + EV
- Top-level action frequencies (fold/call/raise aggregate)
- Combo details for interesting hands

**Step 3: Build, test, commit**

```bash
cargo run -p poker-solver-trainer --release -- inspect-spot \
    -c sample_configurations/blueprint_v2_1kbkt.yaml \
    --spot "sb:2bb,bb:call"
git commit -m "feat: inspect-spot CLI command for spot analysis"
```

---

### Task 7: Round-trip test — encode then load

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` (test module)

**Step 1: Write integration test**

```rust
#[test]
fn encode_then_load_round_trip() {
    // 1. Create session, play some actions
    // 2. Encode the spot
    // 3. Create fresh session, load the encoded spot
    // 4. Verify both sessions have the same state (action_history, board, position)
}
```

This validates that `encode_spot` and `load_spot` are inverses.

**Step 2: Run, commit**

```bash
cargo test -p poker-solver-tauri -- round_trip
git commit -m "test: encode/load spot round-trip verification"
```
