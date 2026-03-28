# TUI Spot Resolution Error Indicator — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Show an error message in the TUI grid when a scenario's spot notation fails to resolve against the game tree.

**Architecture:** Add `error_message: Option<String>` to `HandGridState`. The grid widget checks this field before rendering cells — if set, it renders centered error text instead of the 13×13 matrix. The scenario resolution in `main.rs` sets the error when `resolve_spot` returns `None`.

**Tech Stack:** Rust, ratatui (TUI framework)

---

### Task 1: Add `error_message` field to `HandGridState`

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_widgets.rs:55-65`

**Step 1: Write the failing test**

Add a test in the existing `#[cfg(test)]` module at the bottom of `blueprint_tui_widgets.rs`:

```rust
#[timed_test]
fn error_message_renders_instead_of_grid() {
    let mut state = mock_grid_state();
    state.error_message = Some("Spot failed to resolve: sb:2bb,bb:7bb".to_string());

    let widget = HandGridWidget { state: &state };
    let area = Rect::new(0, 0, 80, 42);
    let mut buf = Buffer::empty(area);
    (&widget).render(area, &mut buf);

    // The error message should appear somewhere in the buffer.
    let content: String = (0..area.height)
        .flat_map(|y| (0..area.width).map(move |x| buf.cell((x, y)).unwrap().symbol().chars().next().unwrap_or(' ')))
        .collect();
    assert!(content.contains("Spot failed to resolve"), "expected error text in buffer, got: {content}");
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-trainer error_message_renders -- --nocapture`
Expected: FAIL — `error_message` field does not exist on `HandGridState`

**Step 3: Add the field to `HandGridState`**

In `crates/trainer/src/blueprint_tui_widgets.rs`, add to the struct after line 64 (`iteration_at_snapshot`):

```rust
    /// When set, the grid renderer shows this error message instead of the 13×13 matrix.
    pub error_message: Option<String>,
```

Then add `error_message: None,` to every existing construction site. There are two:

1. `mock_grid_state()` in the test module (~line 469): add `error_message: None,`
2. `main.rs` scenario construction (~line 303): add `error_message: None,`

Also check for any other construction sites:

Run: `grep -rn "HandGridState {" crates/trainer/src/ --include="*.rs"`

Add `error_message: None,` to each one.

**Step 4: Run test to verify it compiles but fails on assertion**

Run: `cargo test -p poker-solver-trainer error_message_renders -- --nocapture`
Expected: FAIL — the error text is not rendered yet (grid still draws normally)

**Step 5: Commit**

```bash
git add crates/trainer/src/blueprint_tui_widgets.rs crates/trainer/src/main.rs
git commit -m "feat(tui): add error_message field to HandGridState"
```

---

### Task 2: Render error message instead of grid when set

**Files:**
- Modify: `crates/trainer/src/blueprint_tui_widgets.rs:118-135` (the `render` method of `HandGridWidget`)

**Step 1: Add early return for error state**

In the `impl Widget for &HandGridWidget<'_>` render method, right after the opening (line 119), before the title row section, add:

```rust
        // ── Error state ────────────────────────────────────────────────
        if let Some(ref msg) = self.state.error_message {
            let style = Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD);
            // Render scenario name on first line
            let title = format!(" {} ", self.state.scenario_name);
            buf.set_string(area.x, area.y, &title, Style::default().bold());
            // Center the error message vertically and horizontally
            let msg_y = area.y + area.height / 3;
            let msg_x = area.x + area.width.saturating_sub(msg.len() as u16) / 2;
            buf.set_string(msg_x, msg_y, msg, style);
            return;
        }
```

**Step 2: Run the failing test to verify it passes**

Run: `cargo test -p poker-solver-trainer error_message_renders -- --nocapture`
Expected: PASS

**Step 3: Run the full trainer test suite**

Run: `cargo test -p poker-solver-trainer`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/trainer/src/blueprint_tui_widgets.rs
git commit -m "feat(tui): render error message instead of grid when spot fails"
```

---

### Task 3: Set error message when `resolve_spot` returns `None`

**Files:**
- Modify: `crates/trainer/src/main.rs:278-322` (scenario resolution block)

**Step 1: Replace the silent fallback with error_message**

Change the scenario resolution closure from:

```rust
let (node_idx, board) = blueprint_tui_scenarios::resolve_spot(
    &trainer.tree,
    &sc.spot,
)
.unwrap_or_else(|| {
    eprintln!("WARNING: scenario '{}' spot '{}' failed to resolve, using root", sc.name, sc.spot);
    (trainer.tree.root, vec![])
});
```

To:

```rust
let (node_idx, board, error_message) = match blueprint_tui_scenarios::resolve_spot(
    &trainer.tree,
    &sc.spot,
) {
    Some((idx, board)) => (idx, board, None),
    None => {
        let msg = format!("Spot failed to resolve: {}", sc.spot);
        eprintln!("WARNING: scenario '{}': {msg}", sc.name);
        (trainer.tree.root, vec![], Some(msg))
    }
};
```

Then in the `ResolvedScenario` construction (~line 303), set the field:

```rust
error_message,
```

**Step 2: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: Compiles successfully

**Step 3: Run all trainer tests**

Run: `cargo test -p poker-solver-trainer`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/trainer/src/main.rs
git commit -m "feat(tui): surface spot resolution failures as visible errors"
```

---

### Task 4: Final verification

**Step 1: Run the full test suite**

Run: `cargo test`
Expected: All tests pass

**Step 2: Run clippy**

Run: `cargo clippy -p poker-solver-trainer`
Expected: No warnings

**Step 3: Manual smoke test (optional)**

Temporarily set a spot to a bad value in a YAML config and run the trainer briefly to confirm the error renders in the TUI.
