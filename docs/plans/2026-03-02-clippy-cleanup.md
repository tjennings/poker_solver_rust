# Clippy Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate all 180 clippy warnings and 1 error to get a clean `cargo clippy --all-targets` pass.

**Architecture:** Two phases — first investigate the 19 "always returns zero" / "no effect" warnings for potential bugs, then mechanically fix remaining ~160 warnings in parallel by file group. All zero-op warnings turn out to be intentional `0 * n * n` indexing patterns in array layouts like `pos * n * n + hand * n + opponent`, so they need `#[allow]` annotations.

**Tech Stack:** Rust, clippy with `pedantic` lint group

---

## Agent Team & Execution Order

| Phase | Agent | Task | Isolation |
|-|-|-|-|
| 1 (sequential) | rust-developer | Task 1: Fix zero-op/no-effect warnings (bug investigation) | worktree |
| 2 (parallel) | rust-developer A | Task 2: `preflop/tree.rs` (28 warnings) | worktree |
| 2 (parallel) | rust-developer B | Task 3: `postflop_abstraction.rs` + `postflop_exhaustive.rs` + `postflop_bundle.rs` + `postflop_hands.rs` (27 warnings) | worktree |
| 2 (parallel) | rust-developer C | Task 4: `postflop_mccfr.rs` + `solver.rs` + `equity.rs` + `equity_table_cache.rs` + `postflop_tree.rs` + `postflop_model.rs` + `config.rs` (35 warnings) | worktree |
| 2 (parallel) | rust-developer D | Task 5: `agent.rs` + `simulation.rs` + `flops.rs` + misc core + `trainer/` + `tauri-app/` + tests (40 warnings) | worktree |
| 3 (sequential) | manager | Task 6: Final verification | main branch |

---

### Task 1: Fix Zero-Op / No-Effect Warnings (Bug Investigation)

**Files:**
- Modify: `crates/core/src/preflop/solver.rs` (lines 1251, 1253, 1313, 1391, 1409)
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` (line 743)
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs` (lines 1134-1135)
- Modify: `crates/tauri-app/src/exploration.rs` (lines 1139-1140)

**Context:** All 19 warnings are the same pattern: `0 * n * n` or `168 * n + 0` in array index expressions like `pos * n * n + hand * n + opponent`. The `0 *` represents `pos=0` (IP position) and `+ 0` represents `opponent=0`. These are intentional — they make the indexing formula consistent across all call sites. The fix is to allow the lint per-expression.

**Step 1: Add allow attributes to solver.rs test code**

In `crates/core/src/preflop/solver.rs`, add `#[allow(clippy::identity_op, clippy::erasing_op)]` before each test function containing these patterns:
- `postflop_showdown_value_uses_hand_avg` (line ~1244)
- `postflop_showdown_value_limped_pot_uses_equity` (line ~1305)
- `postflop_showdown_value_interpolates_spr` (line ~1383)

```rust
#[allow(clippy::identity_op, clippy::erasing_op)]
#[timed_test]
fn postflop_showdown_value_uses_hand_avg() {
```

**Step 2: Add allow attribute to postflop_abstraction.rs test code**

In `crates/core/src/preflop/postflop_abstraction.rs`, add `#[allow(clippy::identity_op, clippy::erasing_op)]` before the `postflop_values_get_by_flop_index` test (line ~738):

```rust
#[allow(clippy::identity_op, clippy::erasing_op)]
#[timed_test]
fn postflop_values_get_by_flop_index() {
```

**Step 3: Add allow attribute to postflop_exhaustive.rs test code**

In `crates/core/src/preflop/postflop_exhaustive.rs`, add `#[allow(clippy::identity_op, clippy::erasing_op)]` before the test containing lines 1134-1135.

**Step 4: Fix exploration.rs (the hard error)**

In `crates/tauri-app/src/exploration.rs` at lines 1139-1140, add an inline allow:

```rust
#[allow(clippy::identity_op, clippy::erasing_op)]
let vp0 = hand_avg[0 * n * n + hand_index * n + v_idx];
let vp1 = hand_avg[1 * n * n + hand_index * n + v_idx];
```

Note: This is production code (not a test), so the allow must be on the statement, not the function. Use a block-level allow:

```rust
#[allow(clippy::identity_op, clippy::erasing_op)]
{
    let vp0 = hand_avg[0 * n * n + hand_index * n + v_idx];
    let vp1 = hand_avg[1 * n * n + hand_index * n + v_idx];
    // ... rest of the code using vp0, vp1
}
```

**Step 5: Verify**

Run: `cargo clippy --all-targets 2>&1 | grep -c "always return zero\|has no effect"`
Expected: `0`

**Step 6: Commit**

```bash
git add -A && git commit -m "fix(clippy): allow intentional zero-op indexing patterns in array layouts"
```

---

### Task 2: Clean Up `preflop/tree.rs` (28 warnings)

**Files:**
- Modify: `crates/core/src/preflop/tree.rs`

**Warnings to fix:**
- 12x `variables can be used directly in the format! string` — inline variables into format strings
- 9x `wildcard matches only a single variant` — replace `_ =>` with explicit variant names
- 6x `using contains() instead of iter().any()` — use `.contains(&val)`
- 1x `parameter is only used in recursion` — add `#[allow(clippy::only_used_in_recursion)]` if the parameter is part of the public API, or refactor if it's truly unused

**Step 1: Fix format string warnings**

Find all `format!("{}", var_name)` patterns and convert to `format!("{var_name}")`. Same for `println!`, `write!`, `eprintln!`, etc. Example:
```rust
// Before
format!("{}", node_idx)
// After
format!("{node_idx}")
```

**Step 2: Fix wildcard match warnings**

Find all `match` expressions where `_ =>` matches only one remaining variant. Replace with the explicit variant name. Example:
```rust
// Before
match action {
    Action::Fold => ...,
    Action::Check => ...,
    _ => ...,  // only matches Action::Call
}
// After
match action {
    Action::Fold => ...,
    Action::Check => ...,
    Action::Call => ...,
}
```

**Step 3: Fix contains() warnings**

Replace `.iter().any(|x| *x == val)` or `.iter().any(|&x| x == val)` with `.contains(&val)`.

**Step 4: Fix recursion-only parameter**

Investigate the parameter. If it's needed for the recursive structure, allow the lint. If truly unused, remove it.

**Step 5: Verify**

Run: `cargo clippy -p poker-solver-core 2>&1 | grep "preflop/tree.rs" | head -5`
Expected: no warnings from this file

**Step 6: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: all tests pass

**Step 7: Commit**

```bash
git add crates/core/src/preflop/tree.rs && git commit -m "fix(clippy): clean up preflop/tree.rs warnings"
```

---

### Task 3: Clean Up Postflop Abstraction Group (27 warnings)

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs` (~20 warnings)
- Modify: `crates/core/src/preflop/postflop_bundle.rs` (~3 warnings)
- Modify: `crates/core/src/preflop/postflop_exhaustive.rs` (~19 warnings, minus ~4 zero-ops from Task 1)
- Modify: `crates/core/src/preflop/postflop_hands.rs` (~1 warning)

#### postflop_abstraction.rs warnings:
- 4x `casting usize to u16` — add `#[allow(clippy::cast_possible_truncation)]` per-site
- 2x `loop variable i is used to index out` — convert to iterator with `.iter_mut().enumerate()`
- 2x `casting usize to f64` — add `#[allow(clippy::cast_precision_loss)]` per-site
- 1x `casting u64 to f64` — add `#[allow(clippy::cast_precision_loss)]`
- 1x `casting f64 to usize may truncate` — add `#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]`
- 1x `function could have #[must_use]` — add `#[must_use]`
- 1x `needless for_each` — convert `.iter_mut().for_each(|v| *v = x)` to `for v in iter { *v = x; }`
- 1x `item in documentation is missing backticks` — add backticks around code items in doc comments

#### postflop_bundle.rs warnings:
- 2x `redundant closure` — replace `|e| e.ok()` with `Result::ok` and `|e| e.file_name()` with `DirEntry::file_name`
- 1x `map_or can be simplified` — replace `.map_or(false, |n| ...)` with `.is_some_and(|n| ...)`

#### postflop_exhaustive.rs warnings (after Task 1):
- 3x `binding name too similar` — rename `prev_pa`/`cur_pa` to `prev_pruned`/`cur_pruned` or similar
- 2x `casting usize to u32` — `#[allow(clippy::cast_possible_truncation)]`
- 2x `casting usize to u16` — `#[allow(clippy::cast_possible_truncation)]`
- 1x `function has too many lines (141/100)` — add `#[allow(clippy::too_many_lines)]` (refactoring would be too invasive)
- 1x `function has too many lines (106/100)` — add `#[allow(clippy::too_many_lines)]`
- 1x `function has too many arguments (8/7)` — add `#[allow(clippy::too_many_arguments)]`
- 1x `function could have #[must_use]` — add `#[must_use]`
- 1x `manual implementation of midpoint` — use `.midpoint()` or `(a.checked_add(b).unwrap()) / 2`
- 1x `more concise to loop over references` — use `for item in &collection` instead of `collection.iter()`
- 1x `casts from u32 to u64 can use From` — use `u64::from(val)`

#### postflop_hands.rs warning:
- 1x `loop variable i is used to index map` — convert to `.iter_mut().enumerate()`

**Step 1: Apply all fixes to each file**

Follow the patterns described above. For cast truncation, place the allow on the statement or enclosing block.

**Step 2: Verify**

Run: `cargo clippy -p poker-solver-core 2>&1 | grep -E "postflop_(abstraction|bundle|exhaustive|hands)\.rs"`
Expected: no warnings from these files

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core`

**Step 4: Commit**

```bash
git add crates/core/src/preflop/postflop_abstraction.rs crates/core/src/preflop/postflop_bundle.rs crates/core/src/preflop/postflop_exhaustive.rs crates/core/src/preflop/postflop_hands.rs
git commit -m "fix(clippy): clean up postflop abstraction/bundle/exhaustive/hands warnings"
```

---

### Task 4: Clean Up Solver/Equity/Config Group (35 warnings)

**Files:**
- Modify: `crates/core/src/preflop/postflop_mccfr.rs` (~15 warnings)
- Modify: `crates/core/src/preflop/solver.rs` (~3 warnings after Task 1)
- Modify: `crates/core/src/preflop/equity.rs` (~10 warnings)
- Modify: `crates/core/src/preflop/equity_table_cache.rs` (~10 warnings)
- Modify: `crates/core/src/preflop/postflop_tree.rs` (~4 warnings)
- Modify: `crates/core/src/preflop/postflop_model.rs` (~4 warnings)
- Modify: `crates/core/src/preflop/config.rs` (~3 warnings)

#### postflop_mccfr.rs:
- 4x `argument passed by reference, more efficient by value` — change `&Card` parameters to `Card` (6 bytes, under 8-byte limit). Update call sites.
- 2x `doc list item without indentation` — fix markdown indentation in doc comments
- 2x `casting usize to u32` — `#[allow(clippy::cast_possible_truncation)]`
- 2x `casting usize to u16` — `#[allow(clippy::cast_possible_truncation)]`
- 1x `very complex type` — add a type alias for the complex type
- 1x `function has too many lines (118/100)` — `#[allow(clippy::too_many_lines)]`
- 1x `manual is_multiple_of` — use `.is_multiple_of()` if available, or `% == 0`
- 1x `item in documentation missing backticks` — add backticks
- 1x `casts from i32 to f64 can use From` — use `f64::from(val)`

#### solver.rs (after Task 1):
- 3x `item in documentation missing backticks` — add backticks around code items

#### equity.rs:
- 7x `strict comparison of f32 or f64` — replace `==` with `(a - b).abs() < f64::EPSILON` or similar tolerance
- 3x `#[ignore] without reason` — add reason strings: `#[ignore = "slow equity computation"]`

#### equity_table_cache.rs:
- 3x `method could have #[must_use]` — add `#[must_use]` attribute
- 3x `#[ignore] without reason` — add reason strings
- 1x `this can be std::io::Error::other()` — use `std::io::Error::other(msg)`
- 1x `item in documentation missing backticks` — add backticks
- 1x `docs for function returning Result missing # Errors section` — add `# Errors` doc section
- 1x `casting usize to u32` — `#[allow(clippy::cast_possible_truncation)]`

#### postflop_tree.rs:
- 2x `casting usize to u32` — `#[allow(clippy::cast_possible_truncation)]`
- 1x `using contains() instead of iter().any()` — use `.contains()`
- 1x `match arms have identical bodies` — combine the arms with `|`

#### postflop_model.rs:
- 4x `item in documentation missing backticks` — add backticks

#### config.rs:
- 1x `explicit lifetimes could be elided` — remove the explicit `'de` lifetime
- 1x `casting u64 to f64` — `#[allow(clippy::cast_precision_loss)]`
- 1x `casting i64 to f64` — `#[allow(clippy::cast_precision_loss)]`

**Step 1: Apply all fixes file by file**

**Step 2: Verify**

Run: `cargo clippy -p poker-solver-core 2>&1 | grep -c "warning:"` — should be significantly reduced
Specifically check each file has zero warnings.

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core`

**Step 4: Commit**

```bash
git add crates/core/src/preflop/
git commit -m "fix(clippy): clean up postflop_mccfr, solver, equity, config, tree, model warnings"
```

---

### Task 5: Clean Up Remaining Files (40 warnings)

**Files:**
- Modify: `crates/core/src/agent.rs` (~8 warnings)
- Modify: `crates/core/src/simulation.rs` (~5 warnings)
- Modify: `crates/core/src/flops.rs` (~5 warnings)
- Modify: `crates/core/src/info_key.rs` (~3 warnings)
- Modify: `crates/core/src/blueprint/strategy.rs` (~3 warnings)
- Modify: `crates/core/src/blueprint/subgame_cfr.rs` (~3 warnings)
- Modify: `crates/core/src/cfr/dcfr.rs` (~1 warning)
- Modify: `crates/core/src/cfr/parallel.rs` (~3 warnings)
- Modify: `crates/core/src/card_utils.rs` (~1 warning)
- Modify: `crates/core/src/equity.rs` (~1 warning — dead code)
- Modify: `crates/core/tests/full_pipeline.rs` (~3 warnings)
- Modify: `crates/core/tests/postflop_diagnostics.rs` (~1 warning)
- Modify: `crates/core/tests/postflop_imperfect_recall.rs` (~1 warning)
- Modify: `crates/core/tests/preflop_convergence.rs` (~3 warnings)
- Modify: `crates/trainer/src/main.rs` (~4 warnings)
- Modify: `crates/trainer/src/hand_trace.rs` (~1 warning)
- Modify: `crates/trainer/src/lhe_viz.rs` (~2 warnings)
- Modify: `crates/trainer/src/tui_metrics.rs` (~1 warning)
- Modify: `crates/tauri-app/src/exploration.rs` (~1 warning — complex type)

#### agent.rs:
- 8x `unnecessary hashes around raw string literal` — change `r#"..."#` to `r"..."`

#### simulation.rs:
- 5x `unnecessary hashes around raw string literal` — change `r#"..."#` to `r"..."`

#### flops.rs:
- 5x `variables can be used directly in format! string` — inline variables

#### info_key.rs:
- 3x `casting usize to u16` — `#[allow(clippy::cast_possible_truncation)]`

#### blueprint/strategy.rs:
- 3x `casting usize to u32` — `#[allow(clippy::cast_possible_truncation)]`

#### blueprint/subgame_cfr.rs:
- 1x `loop variable j is used to index matrix` — convert to iterator
- 1x `loop variable i is used to index matrix` — convert to iterator
- 1x `manual RangeInclusive::contains` — use `(a..=b).contains(&val)`

#### cfr/dcfr.rs:
- 1x `multiplication by -1 can be written more succinctly` — use unary negation `-val` instead of `val * -1`

#### cfr/parallel.rs:
- 2x `manually reimplementing div_ceil` — use `.div_ceil()`
- 1x `item in documentation missing backticks` — add backticks

#### card_utils.rs:
- 1x `casting usize to u8` — `#[allow(clippy::cast_possible_truncation)]`

#### equity.rs:
- 1x `function prewarm_cache is never used` — remove or add `#[allow(dead_code)]` with a comment explaining it's for future use

#### tests/full_pipeline.rs:
- 3x `expression creates reference immediately dereferenced` — remove `&` from `&*expr` patterns

#### tests/postflop_diagnostics.rs + postflop_imperfect_recall.rs:
- 2x `borrowed expression implements required traits` — simplify borrows

#### tests/preflop_convergence.rs:
- 1x `loop variable h is used to index` — convert to iterator
- 1x `manual is_multiple_of` — use `% == 0`
- 1x `casting to same type unnecessary` — remove `as u64` cast

#### trainer/main.rs:
- 3x `if statement can be collapsed` — combine nested `if` into `if a && b`
- 1x `loop variable row is used to index grid` — convert to iterator

#### trainer/hand_trace.rs:
- 1x `loop variable tex_id is used to index flops` — convert to iterator

#### trainer/lhe_viz.rs:
- 1x `expression creates reference immediately dereferenced` — simplify
- 1x `loop variable row is used to index RANK_ORDER` — convert to iterator

#### trainer/tui_metrics.rs:
- 1x `function has too many arguments (8/7)` — `#[allow(clippy::too_many_arguments)]`

#### tauri-app/exploration.rs:
- 1x `very complex type` — add a type alias

**Step 1: Apply all fixes**

Work through each file, applying the described fixes.

**Step 2: Verify**

Run: `cargo clippy --all-targets 2>&1 | grep "^warning:" | grep -v "generated\|duplicates\|build failed\|waiting" | wc -l`
Expected: `0`

**Step 3: Run tests**

Run: `cargo test`

**Step 4: Commit**

```bash
git add -A
git commit -m "fix(clippy): clean up remaining warnings across agent, simulation, flops, tests, trainer, tauri"
```

---

### Task 6: Final Verification

**Step 1: Run full clippy check**

Run: `cargo clippy --all-targets 2>&1`
Expected: zero warnings, zero errors

**Step 2: Run full test suite**

Run: `cargo test`
Expected: all tests pass (known timer failures in `blueprint/subgame_cfr`, `preflop/bundle` are pre-existing)

**Step 3: Squash/organize commits if needed**

Ensure commit history is clean and each commit message is descriptive.
