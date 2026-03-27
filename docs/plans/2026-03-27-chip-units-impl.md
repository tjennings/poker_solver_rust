# Unify on Chips Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Change the V2 tree from BB units (small_blind=0.5) to chips (small_blind=1), remove all `* 2` / `/ 2` unit conversions from internal code, and keep BB display only at the UI/CLI boundary.

**Architecture:** Update config defaults and YAML files to chips, remove `* 2.0` conversions in the display layer, remove `bb_scale` concept, add explicit `chips_to_bb()` helper at the display boundary. The range-solver already uses chips — no change needed there.

**Tech Stack:** Rust (core, tauri-app), TypeScript (frontend), YAML configs

---

### Task 1: Update config defaults to chips

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs`

**Step 1: Change default values**

Find the `GameConfig` defaults (around line 306-320). Change:
- `small_blind: 0.5` → `small_blind: 1.0`
- `big_blind: 1.0` → `big_blind: 2.0`
- `stack_depth: 100.0` → `stack_depth: 200.0`

**Step 2: Update any tests that assert on these defaults**

Search for tests that check `small_blind == 0.5` or `stack_depth == 100` and update them.

**Step 3: Build core crate**

```bash
cargo build -p poker-solver-core
cargo test -p poker-solver-core
```

Fix any test failures caused by the default change.

**Step 4: Commit**

```
git commit -m "refactor: config defaults use chips (sb=1, bb=2, stack=200)"
```

---

### Task 2: Update all YAML config files

**Files:**
- Modify: `sample_configurations/*.yaml` — every file with `game:` section

**Step 1: Update each YAML file**

For each config file:
- `stack_depth: X` → `stack_depth: X * 2` (e.g., 50 → 100, 100 → 200)
- `small_blind: 0.5` → `small_blind: 1`
- `big_blind: 1.0` → `big_blind: 2`

Leave action_abstraction bet sizes unchanged (they're pot fractions, unitless).

Preflop action sizes like `["2bb"]`, `["4bb", "7bb"]` need attention — these are labels that the tree builder parses. Check if the tree builder interprets "2bb" as 2× big_blind or as a literal 2 BB. If it uses `big_blind` from config, the values will automatically scale. If it hardcodes BB=1, update them.

**Step 2: Verify a config loads**

```bash
cargo run -p poker-solver-trainer --release -- inspect-spot \
    -c sample_configurations/blueprint_v2_1kbkt_sapcfr.yaml \
    --spot ""
```

Should show `Pot: 1BB` (pot = 3 chips = 1.5 BB... wait, that's the blind posting). Verify the numbers make sense. Pot at root should be SB + BB = 1 + 2 = 3 chips = 1.5 BB.

**Step 3: Commit**

```
git commit -m "config: update all YAML files to chip units (sb=1, bb=2)"
```

---

### Task 3: Remove `* 2.0` conversions in display layer

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs`
- Modify: `crates/tauri-app/src/exploration.rs`

**Step 1: game_session.rs — remove `* 2.0` on pot/stacks**

Search for `* 2.0` in game_session.rs. The key locations from the audit:
- `compute_pot()` — uses `pot_at_v2_node(...) * 2.0`. Remove the `* 2.0`.
- `compute_stacks()` — uses `(stack_depth - invested) * 2.0`. Remove the `* 2.0`.

After this change, pot/stacks returned by `get_state()` are in chips.

**Step 2: exploration.rs — remove `* 2.0` on pot/stacks**

- Line ~1276: `let pot = (node_pot * 2.0) as u32;` → `let pot = node_pot as u32;`
- Line ~1280-1281: stack computation with `* 2.0` → remove

**Step 3: exploration.rs — remove `bb_scale` concept**

The `bb_scale` parameter (0.5 for subgame, 1.0 for preflop) exists because the tree was in BB and action amounts needed conversion. With everything in chips:
- `bb_scale` is no longer needed
- Action amounts from the tree are already in chips
- The display conversion (chips → "Xbb") happens at the label formatting boundary

Find all uses of `bb_scale` and replace with `1.0`, then remove the parameter entirely.

**Step 4: Build and test**

```bash
cargo build -p poker-solver-tauri
cargo test -p poker-solver-tauri
```

**Step 5: Commit**

```
git commit -m "refactor: remove * 2.0 conversions and bb_scale from display layer"
```

---

### Task 4: Update action label formatting

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` — `v2_action_info` and similar
- Modify: `crates/tauri-app/src/game_session.rs` — `format_tree_action`, `build_game_actions`
- Modify: `crates/tauri-app/src/game_session.rs` — `range_solver_action_to_game_action`

**Step 1: Preflop action labels**

The V2 tree's `TreeAction::Raise(amount)` stores the raise-to amount. With chips, a raise to 4 chips = 2 BB. The label should be "2bb".

In `format_tree_action` or wherever labels are generated:
```rust
// Before: amount was in BB, label was "{amount}bb"
// After: amount is in chips, label is "{amount / 2}bb"
TreeAction::Raise(amount) => format!("{}bb", amount / 2),
```

Wait — but preflop raise sizes in the config are strings like `"2bb"`. The tree builder parses these. If the tree builder now sees `big_blind: 2` and parses `"2bb"` as `2 * big_blind = 4 chips`, that's correct. The label needs to convert back: `4 chips / 2 = 2bb`.

**Step 2: Subgame/range-solver action labels**

In `range_solver_action_to_game_action`:
```rust
// Already in chips. Convert to BB for label.
range_solver::Action::Bet(amt) => {
    let bb = *amt as f64 / 2.0;
    (format!("{bb:.0}bb"), "bet")
}
```

This is already correct — it divides by 2. Keep it.

**Step 3: V2 tree action labels in exploration.rs**

The `v2_action_info` function formats action labels. Check what it does with amounts and update for chips:
```rust
// amount is now in chips. Display as BB.
let bb = (amount * bb_scale) // bb_scale removed, amount already in chips
// → let bb = amount / 2.0;
```

**Step 4: Build, verify labels make sense**

```bash
cargo run -p poker-solver-trainer --release -- inspect-spot \
    -c sample_configurations/blueprint_v2_1kbkt_sapcfr.yaml \
    --spot "sb:2bb"
```

Should show "2bb" action, not "4bb" or "1bb".

**Step 5: Commit**

```
git commit -m "refactor: action labels convert chips to BB at display boundary"
```

---

### Task 5: Remove simulation harness `* 2` conversion

**Files:**
- Modify: `crates/core/src/simulation.rs`

**Step 1: Find and remove conversion**

Line ~201-203:
```rust
// Before: stack_depth is in BB. Internal units: 1 BB = 2 units
let stacks = vec![(stack_depth * 2) as f32; 2];
// After: stack_depth is already in chips
let stacks = vec![stack_depth as f32; 2];
```

**Step 2: Build, test, commit**

```bash
cargo test -p poker-solver-core
git commit -m "refactor: simulation uses chip values directly from config"
```

---

### Task 6: Update SPR computation

**Files:**
- Modify: `crates/core/src/info_key.rs`

**Step 1: Check SPR formula**

Line ~40: `(eff_stack * 2 / pot).min(31)`

With chips, `eff_stack` and `pot` are already in the same unit. The `* 2` was converting from BB to chips. Remove it:
```rust
(eff_stack / pot).min(31)
```

**Step 2: Build, test, commit**

```bash
cargo test -p poker-solver-core
git commit -m "refactor: SPR computation uses chips directly"
```

---

### Task 7: Frontend display boundary

**Files:**
- Modify: `frontend/src/GameExplorer.tsx`
- Modify: `frontend/src/Explorer.tsx`

**Step 1: Add `chips_to_bb` helper**

```tsx
const chipsToBB = (chips: number): string => `${(chips / 2).toFixed(1)}`;
```

**Step 2: Verify EV display**

`cell.ev / 2` in GameExplorer.tsx — this is already the correct display conversion (chips → BB). Keep it.

**Step 3: Verify pot/stack display**

In `ActionBlock` header (Explorer.tsx):
```tsx
// stack and pot come from GameState, which is now in chips
// Display as BB:
<span className="stack">
    {chipsToBB(stack)}BB / {chipsToBB(pot)}BB
</span>
```

Check that the `ActionBlock` component already does `/ 2` for display. If it was doing `/ 2` before the refactor (when values were already in BB-ish units from the `* 2` conversion), it would now show half. The `* 2` removal in Task 3 means pot/stacks from `get_state()` are now raw chips, so the `/ 2` in the ActionBlock header IS the correct display conversion.

Verify by checking what `ActionBlock` does with `stack` and `pot` props.

**Step 4: Build, verify**

```bash
cd frontend && npx tsc --noEmit
```

**Step 5: Commit**

```
git commit -m "refactor: frontend displays chips as BB at display boundary"
```

---

### Task 8: CLI display boundary

**Files:**
- Modify: `crates/trainer/src/inspect_spot.rs`

**Step 1: Verify pot/stack display**

The CLI already has `state.pot / 2` for display. With chips, this is correct. Verify the labels say "BB".

**Step 2: Verify EV display**

`ev / 2.0` — already correct. Verify.

**Step 3: Verify action labels**

Action labels come from the backend (already converted in Task 4). No CLI change needed.

**Step 4: Commit if changes needed**

```
git commit -m "refactor: CLI display uses chips_to_bb conversion"
```

---

### Task 9: Update test fixtures and struct literals

**Files:**
- Multiple test files that construct `TrainingConfig`, `GameConfig`, `TreeConfig` with hardcoded values

**Step 1: Search and update all test struct literals**

```bash
grep -rn "small_blind.*0\.5\|big_blind.*1\.0\|stack_depth.*100\b" crates/
```

Update every test that uses the old BB defaults to use chip defaults (sb=1, bb=2, stack=200).

**Step 2: Run full test suite**

```bash
cargo test --workspace
```

Fix ALL failures. This is the critical verification step.

**Step 3: Commit**

```
git commit -m "test: update all test fixtures to chip units"
```

---

### Task 10: Update training TUI display

**Files:**
- Modify: `crates/core/src/blueprint_v2/trainer.rs` — TUI scenario display
- Modify: `crates/trainer/src/blueprint_tui_scenarios.rs` (if exists)

**Step 1: Check TUI pot/stack display**

The training TUI shows pot and stack sizes. Verify these are converted to BB for display.

**Step 2: Commit if changes needed**

```
git commit -m "refactor: training TUI displays BB at boundary"
```

---

### Task 11: Update documentation

**Files:**
- Modify: `docs/training.md`
- Modify: `docs/architecture.md`

**Step 1: Document the unit convention**

Add a section to architecture.md:
```
## Unit Convention
All internal values (pot, stacks, bet sizes, EVs) are in chips.
1 BB = 2 chips. Config files use chips.
Display to users converts chips to BB (÷ 2).
```

**Step 2: Update config examples in training.md**

Update all example YAML snippets to show chip values.

**Step 3: Commit**

```
git commit -m "docs: document chip unit convention"
```

---

### Task 12: Full integration verification

**Step 1: Run full test suite**

```bash
cargo test --workspace
cd frontend && npx tsc --noEmit && npx vitest run
```

**Step 2: Manual verification**

1. Load a blueprint in the UI — pot/stacks should display as BB
2. Run `inspect-spot` — values should match the UI
3. Train a new blueprint (short run) — verify it starts correctly
4. Verify action labels show correct BB values ("2bb", "7bb")

**Step 3: Final commit**

```
git commit -m "verify: full integration test of chip unit unification"
```
