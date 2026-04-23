# Libratus-style Safe Re-solving Gadget Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Implement the `BlueprintCbvOptOut` provider + CLI flags + Tauri/UI wiring + validation runs for the Libratus-style safe re-solving gadget, per the validated design at `docs/plans/2026-04-23-deepstack-gadget.md`.

**Architecture:** The skeleton (`OptOutProvider` trait, `ConstantOptOut`, `GadgetEvaluator`, `--gadget` CLI flag, 9 tests) already shipped on branch `feat/subgame-exact-parity`. Remaining work builds the production opt-out source (blueprint bucketed CBVs), plumbs the gadget through every code path that uses a `BoundaryEvaluator` (currently only exact_subtree in compare-solve), and runs validation iterations 9–11 against the subgame-exact-parity harness.

**Tech Stack:** Rust (range-solver, tauri-app, trainer crates), TypeScript/React (frontend), Tauri IPC, `beans` issue tracker.

**Parent bean:** `poker_solver_rust-lay5`.

**Branch:** `feat/subgame-exact-parity`.

**Pre-commit:** Every commit in this project requires `hex:software-simplicity`, `hex:code-reviewer`, `hex:spec-compliance` reviews + `touch .reviews-done` before `git commit` succeeds (pre-commit hook).

---

## Hexagonal layer organisation

- **Domain** (Phase 1): Pure logic for `BlueprintCbvOptOut` — lookup hand → bucket → CBV → bcfv units.
- **Coordination** (Phase 2): CLI flag parsing, wiring provider into evaluator setup functions in `compare_solve.rs` and `game_session.rs`.
- **Adapters** (Phase 3): Tauri command parameter, Settings UI checkbox, frontend config.
- **Validation** (Phase 4): Harness runs + progress-doc + error-matrix updates.

Outside-in is not a perfect fit here because the domain (`BlueprintCbvOptOut`) is the riskiest new code — we start there to de-risk, then wire out.

---

## Phase 1 — Domain: `BlueprintCbvOptOut`

### Task 1: Update gadget module docstring

**Files:**
- Modify: `crates/tauri-app/src/gadget.rs:1-8`

**Step 1: Replace the "DeepStack-style" docstring with Libratus framing**

Change lines 1–8 of `gadget.rs` from:

```rust
//! DeepStack-style range gadget for safe subgame solving.
```

to:

```rust
//! Libratus-style safe re-solving gadget.
//!
//! Clamps opponent per-hand CFV upward to a pre-computed opt-out floor
//! (typically blueprint CBVs). Makes subgame boundary evaluators "safe"
//! in the sense that the reported opponent CFV is never worse than the
//! blueprint would guarantee. See `docs/plans/2026-04-23-deepstack-gadget.md`
//! for the full design and the distinction from DeepStack-proper
//! (which requires a cfvnet retrain; bean poker_solver_rust-akg3).
```

**Step 2: Build to verify no breakage**

Run: `cargo build -p poker-solver-tauri`
Expected: clean compile.

**Step 3: Commit**

```bash
touch .reviews-done  # pre-commit gate (no reviews needed for doc-only edit)
git add crates/tauri-app/src/gadget.rs
git commit -m "docs(gadget): rename DeepStack→Libratus in module docstring"
```

---

### Task 2: Add `chip_cfv_to_bcfv` unit conversion helper

**Files:**
- Modify: `crates/tauri-app/src/gadget.rs` (add free function near the top of the file, below `ConstantOptOut`)
- Test: `crates/tauri-app/src/gadget.rs` (add to `mod tests`)

**Step 1: Write the failing test**

Add to `gadget.rs` test module:

```rust
#[test]
fn chip_cfv_to_bcfv_converts_correctly() {
    // half_pot = 73 chips (representative of our test spot)
    // chip_cfv of +73 means "won one half-pot" → bcfv = 1.0
    assert!((chip_cfv_to_bcfv(73.0, 73.0) - 1.0).abs() < 1e-6);
    // chip_cfv of 0 → bcfv = 0
    assert_eq!(chip_cfv_to_bcfv(0.0, 73.0), 0.0);
    // chip_cfv of -73 → bcfv = -1.0 (lost one half-pot)
    assert!((chip_cfv_to_bcfv(-73.0, 73.0) - (-1.0)).abs() < 1e-6);
    // chip_cfv of +36.5 → bcfv = 0.5
    assert!((chip_cfv_to_bcfv(36.5, 73.0) - 0.5).abs() < 1e-6);
}

#[test]
#[should_panic(expected = "half_pot must be positive")]
fn chip_cfv_to_bcfv_zero_half_pot_panics() {
    chip_cfv_to_bcfv(10.0, 0.0);
}
```

**Step 2: Run test to verify failure**

Run: `cargo test -p poker-solver-tauri gadget::tests::chip_cfv_to_bcfv`
Expected: FAIL — `chip_cfv_to_bcfv` not found.

**Step 3: Implement the minimal function**

Add to `gadget.rs` before `impl OptOutProvider for ConstantOptOut`:

```rust
/// Convert a per-hand chip CFV into pot-normalised bcfv units.
///
/// Blueprint `CbvTable` stores per-bucket CBVs in chips; the gadget's
/// `OptOutProvider` contract returns bcfv (1.0 = one half-pot won). This
/// helper lets `BlueprintCbvOptOut` do the conversion at construction.
pub fn chip_cfv_to_bcfv(chip_cfv: f32, half_pot_chips: f32) -> f32 {
    assert!(half_pot_chips > 0.0, "half_pot must be positive");
    chip_cfv / half_pot_chips
}
```

**Step 4: Run tests to verify pass**

Run: `cargo test -p poker-solver-tauri gadget::tests::chip_cfv_to_bcfv`
Expected: 2 tests PASS.

**Step 5: Commit**

```bash
touch .reviews-done
git add crates/tauri-app/src/gadget.rs
git commit -m "feat(gadget): chip_cfv_to_bcfv unit conversion helper"
```

---

### Task 3: `BlueprintCbvOptOut` struct + construction panics

**Files:**
- Modify: `crates/tauri-app/src/gadget.rs` (add new struct + tests)

**Context you need to read:**
- `crates/tauri-app/src/postflop.rs:1260-1280` — `CbvContext` definition (`cbv_table`, `abstract_tree`, `all_buckets`, `strategy`).
- `crates/tauri-app/src/postflop.rs:419-450` — `SolveBoundaryEvaluator::new` shows the shape of a boundary evaluator constructor that consumes `CbvContext`.
- `crates/tauri-app/src/postflop.rs:588` — an example of using `abstract_start_node` to query the blueprint tree.
- `crates/core/src/blueprint_v2/cbv.rs` — `CbvTable` structure and lookup methods.
- `crates/core/src/blueprint_v2/mccfr.rs` — `AllBuckets::lookup` signature.

**Step 1: Write the failing test for construction**

Add to `gadget.rs` tests:

```rust
#[test]
#[should_panic(expected = "CbvTable has no values")]
fn blueprint_cbv_construct_panics_on_empty_table() {
    use poker_solver_core::blueprint_v2::cbv::CbvTable;
    let empty_table = CbvTable {
        values: vec![],
        node_offsets: vec![],
        buckets_per_node: vec![],
    };
    let _ = BlueprintCbvOptOut::new_for_test(
        Arc::new(empty_table),
        0, // abstract_node_idx
        vec![(0u8, 1u8)],  // opponent_private_cards  (1 hand)
        vec![(2u8, 3u8)],  // player_private_cards
        73.0, // half_pot_chips
    );
}
```

**Step 2: Verify it fails**

Run: `cargo test -p poker-solver-tauri gadget::tests::blueprint_cbv_construct_panics`
Expected: FAIL — `BlueprintCbvOptOut` not found.

**Step 3: Implement the struct and a test-only constructor**

Add to `gadget.rs`:

```rust
use poker_solver_core::blueprint_v2::cbv::CbvTable;

/// Opt-out provider that pulls per-hand CFVs from a blueprint's `CbvTable`.
///
/// CBVs are read at construction (once per boundary) and cached as a
/// per-player `Vec<f32>` in bcfv units. At runtime, `opt_out_cfvs` is a
/// pure vec clone.
pub struct BlueprintCbvOptOut {
    /// Per-player, per-hand pre-computed bcfv opt-out values.
    /// Index: `per_hand_cbv_bcfv[player][hand_idx]`.
    per_hand_cbv_bcfv: [Vec<f32>; 2],
}

impl BlueprintCbvOptOut {
    /// Test-only constructor that accepts a raw `CbvTable` and hand lists.
    /// Production callers use `BlueprintCbvOptOut::from_cbv_context`
    /// (added in Task 4).
    #[cfg(test)]
    pub(crate) fn new_for_test(
        cbv_table: Arc<CbvTable>,
        _abstract_node_idx: u32,
        oop_private_cards: Vec<(u8, u8)>,
        ip_private_cards: Vec<(u8, u8)>,
        _half_pot_chips: f32,
    ) -> Self {
        assert!(
            !cbv_table.values.is_empty(),
            "CbvTable has no values; cannot construct BlueprintCbvOptOut"
        );
        Self {
            per_hand_cbv_bcfv: [
                vec![0.0; oop_private_cards.len()],
                vec![0.0; ip_private_cards.len()],
            ],
        }
    }
}
```

**Step 4: Run test to verify pass**

Run: `cargo test -p poker-solver-tauri gadget::tests::blueprint_cbv_construct_panics`
Expected: PASS.

**Step 5: Commit**

```bash
# Dispatch three reviews first (see /hex:software-simplicity, /hex:code-reviewer, /hex:spec-compliance)
touch .reviews-done
git add crates/tauri-app/src/gadget.rs
git commit -m "feat(gadget): BlueprintCbvOptOut struct with empty-table panic"
```

---

### Task 4: `BlueprintCbvOptOut::from_cbv_context` production constructor

**Files:**
- Modify: `crates/tauri-app/src/gadget.rs` (add production constructor + test)

**Context you need:**
- The `AllBuckets::lookup_bucket(board, c1, c2)` signature — returns `Option<u32>` for the bucket index, `None` if unreachable.
- `CbvTable::cbv(node_idx, bucket, player)` — signature may be `cbv(node_idx: u32, bucket: u32, player: u8) -> f32` or similar; grep `crates/core/src/blueprint_v2/cbv.rs` for the actual method name.

**Step 1: Write the failing test — missing bucket panics**

Add:

```rust
#[test]
#[should_panic(expected = "no bucket found for hand")]
fn blueprint_cbv_lookup_panics_on_missing_bucket() {
    // Construct with a bogus hand that AllBuckets will fail to resolve.
    // The test uses a stub AllBuckets that always returns None.
    use std::sync::Arc;
    struct StubBuckets;
    impl poker_solver_core::blueprint_v2::mccfr::BucketLookup for StubBuckets {
        fn lookup_bucket(&self, _board: &[u8], _c1: u8, _c2: u8) -> Option<u32> {
            None
        }
    }
    // ...minimal CbvContext with StubBuckets; call from_cbv_context; assert panic.
    // (Skip writing the full test body if AllBuckets isn't trait-dispatched —
    // in that case use a CbvTable + AllBuckets fixture where one hand's
    // card combination doesn't resolve, e.g. a card on the board.)
    todo!("Fill in once BucketLookup trait shape is verified");
}
```

**Note to implementer:** read `crates/core/src/blueprint_v2/mccfr.rs` and `crates/tauri-app/src/postflop.rs:588` to understand how `SolveBoundaryEvaluator` does the lookup. Adapt this test shape to whatever dispatch mechanism `AllBuckets` uses. If trait-based, use a stub; if concrete, use a known bad input.

**Step 2: Run the test (after filling in the body) to verify failure**

Run: `cargo test -p poker-solver-tauri gadget::tests::blueprint_cbv_lookup_panics`
Expected: FAIL — `from_cbv_context` not defined.

**Step 3: Implement `from_cbv_context`**

Add to `gadget.rs`:

```rust
use crate::postflop::CbvContext;

impl BlueprintCbvOptOut {
    /// Production constructor. Reads the blueprint's CBV values for
    /// every combo in both players' ranges at the given abstract node,
    /// converts to bcfv units, and caches.
    ///
    /// PANICS:
    /// - If `cbv_context.cbv_table.values` is empty.
    /// - If any hand in `private_cards` lacks a bucket assignment
    ///   (this is a bug — the parent `PostFlopGame` should have already
    ///   filtered blocker-conflicting hands).
    /// - If `half_pot_chips <= 0`.
    pub fn from_cbv_context(
        cbv_context: &CbvContext,
        abstract_node_idx: u32,
        board: &[u8],
        private_cards: &[Vec<(u8, u8)>; 2],
        half_pot_chips: f32,
    ) -> Self {
        assert!(
            !cbv_context.cbv_table.values.is_empty(),
            "CbvTable has no values; cannot construct BlueprintCbvOptOut"
        );
        assert!(half_pot_chips > 0.0, "half_pot must be positive");

        let mut per_hand: [Vec<f32>; 2] = [Vec::new(), Vec::new()];
        for player in 0..2 {
            let hands = &private_cards[player];
            per_hand[player].reserve(hands.len());
            for &(c1, c2) in hands {
                let bucket = cbv_context.all_buckets
                    .lookup_bucket(board, c1, c2)
                    .unwrap_or_else(|| panic!(
                        "no bucket found for hand ({c1}, {c2}) on board {:?} \
                         in player {player}; this hand should have been \
                         filtered by PostFlopGame::assign_zero_weights",
                         board
                    ));
                let chip_cbv = cbv_context.cbv_table.cbv(
                    abstract_node_idx, bucket, player as u8,
                );
                per_hand[player].push(chip_cfv_to_bcfv(chip_cbv, half_pot_chips));
            }
        }
        Self { per_hand_cbv_bcfv: per_hand }
    }
}
```

**Note on `lookup_bucket` / `cbv`:** if the actual API differs (it probably does), adjust. The implementer should cross-reference how `SolveBoundaryEvaluator` does these lookups in `postflop.rs` and match that pattern.

**Step 4: Implement `OptOutProvider` for `BlueprintCbvOptOut`**

Add:

```rust
impl OptOutProvider for BlueprintCbvOptOut {
    fn opt_out_cfvs(
        &self,
        _boundary_ordinal: usize,
        opponent: usize,
        _pot: i32,
        _effective_stack: i32,
        _board: &[u8],
        opponent_private_cards: &[(u8, u8)],
    ) -> Vec<f32> {
        assert_eq!(
            opponent_private_cards.len(),
            self.per_hand_cbv_bcfv[opponent].len(),
            "opt_out_cfvs called with hand list of length {} but \
             constructor registered {} hands for player {opponent}",
            opponent_private_cards.len(),
            self.per_hand_cbv_bcfv[opponent].len(),
        );
        self.per_hand_cbv_bcfv[opponent].clone()
    }
}
```

**Step 5: Run both tests**

Run: `cargo test -p poker-solver-tauri gadget::tests::blueprint_cbv`
Expected: `blueprint_cbv_construct_panics_on_empty_table` PASS, `blueprint_cbv_lookup_panics_on_missing_bucket` PASS.

**Step 6: Add a smoke test that construction succeeds with valid inputs**

Add:

```rust
#[test]
fn blueprint_cbv_opt_out_succeeds_with_valid_inputs() {
    // Build a minimal valid CbvContext fixture. Reuse helpers from
    // crates/tauri-app/src/postflop.rs::tests if available; otherwise
    // construct a one-node CbvTable with known CBV values per bucket.
    //
    // Then call from_cbv_context and verify opt_out_cfvs returns the
    // expected bcfv values (chip_cbv / half_pot).
    todo!("Fill in with the minimal valid CbvContext fixture")
}
```

Note to implementer: this is a fixture-heavy test. Look at the existing `SolveBoundaryEvaluator` unit tests in `postflop.rs` (lines 3161+) for how they build a test `CbvContext` + `CbvTable` and borrow from that.

**Step 7: Commit**

```bash
# Reviews: simplicity, code-reviewer, spec-compliance
touch .reviews-done
git add crates/tauri-app/src/gadget.rs
git commit -m "feat(gadget): BlueprintCbvOptOut with CbvContext lookup"
```

---

## Phase 2 — Coordination: CLI flags + wiring

### Task 5: Add `--gadget-provider` and `--gadget-constant` CLI flags

**Files:**
- Modify: `crates/trainer/src/main.rs` — `Commands::CompareSolve` struct (around line 347)
- Modify: `crates/trainer/src/main.rs` — the arm handling `CompareSolve` (around line 1317)

**Step 1: Add the flag definitions**

In the `CompareSolve` struct (search for `#[command(name = "compare-solve")]`), after the existing `--gadget` field, add:

```rust
/// Opt-out provider when --gadget is set. "blueprint-cbv" reads from
/// the bundle's CbvTable (production). "constant" uses a fixed value
/// from --gadget-constant (diagnostic).
#[arg(long, default_value = "blueprint-cbv")]
gadget_provider: String,

/// Constant opt-out value (pot-normalised bcfv) when
/// --gadget-provider=constant. Ignored otherwise. Use e.g.
/// -999.0 to sanity-check that a dominated opt-out is a no-op.
#[arg(long, default_value_t = 0.0)]
gadget_constant: f32,
```

**Step 2: Add a failing unit test in `main.rs`**

Find the existing `#[cfg(test)] mod tests` section (around line 2759) and add:

```rust
#[test]
fn compare_solve_accepts_gadget_provider_flag() {
    let cli = Cli::try_parse_from([
        "trainer",
        "compare-solve",
        "--bundle", "/tmp/dummy",
        "--spot", "any",
        "--river-boundary", "cfvnet",
        "--river-model", "/tmp/m",
        "--gadget",
        "--gadget-provider", "constant",
        "--gadget-constant", "-0.5",
    ]).expect("should parse");
    if let Commands::CompareSolve { gadget, gadget_provider, gadget_constant, .. } = cli.command {
        assert!(gadget);
        assert_eq!(gadget_provider, "constant");
        assert_eq!(gadget_constant, -0.5);
    } else {
        panic!("expected CompareSolve");
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-trainer compare_solve_accepts_gadget_provider_flag`
Expected: PASS.

**Step 4: Thread the new fields through to `compare_solve::run`**

In the `Commands::CompareSolve { ... }` destructuring arm, add the two fields to the pattern, and pass them to `run(...)` as two new parameters (update `run`'s signature in `crates/trainer/src/compare_solve.rs:714` accordingly — `gadget_provider: String, gadget_constant: f32`).

In the run function, replace the existing `gadget: bool` handling (which constructs `ConstantOptOut(0.0)`) with:

```rust
let opt_out_provider: Option<Arc<dyn OptOutProvider>> = if gadget {
    match gadget_provider.as_str() {
        "constant" => Some(Arc::new(ConstantOptOut(gadget_constant))),
        "blueprint-cbv" => {
            // Build BlueprintCbvOptOut per boundary. Done in Task 6 — for
            // now, return None and eprintln a todo.
            eprintln!("[compare] gadget-provider=blueprint-cbv not yet wired");
            None
        }
        other => {
            return Err(format!(
                "invalid --gadget-provider '{other}': expected 'blueprint-cbv' or 'constant'"
            ));
        }
    }
} else {
    None
};
```

**Step 5: Build**

Run: `cargo build -p poker-solver-trainer`
Expected: clean.

**Step 6: Commit**

```bash
# Reviews
touch .reviews-done
git add crates/trainer/src/main.rs crates/trainer/src/compare_solve.rs
git commit -m "feat(compare-solve): --gadget-provider/--gadget-constant CLI flags"
```

---

### Task 6: Wire BlueprintCbvOptOut into compare-solve's neural path

**Files:**
- Modify: `crates/trainer/src/compare_solve.rs` — `setup_neural_boundaries` (around line 392) and `run` (around line 714).

**Step 1: Thread `Option<Arc<dyn OptOutProvider>>` into `setup_neural_boundaries`**

Change the signature from:

```rust
fn setup_neural_boundaries(game: &mut PostFlopGame, model_path: &Path)
```

to:

```rust
fn setup_neural_boundaries(
    game: &mut PostFlopGame,
    model_path: &Path,
    opt_out_provider: Option<Arc<dyn OptOutProvider>>,
)
```

Inside the function, after constructing each `NeuralBoundaryEvaluator`, wrap in `GadgetEvaluator` when the provider is `Some`:

```rust
let inner: Arc<dyn BoundaryEvaluator> = Arc::new(neural_eval);
let wrapped: Arc<dyn BoundaryEvaluator> = if let Some(ref provider) = opt_out_provider {
    Arc::new(poker_solver_tauri::gadget::GadgetEvaluator::new(
        inner,
        Arc::clone(provider),
        board_4.to_vec(),
        private_cards_pair.clone(),
    ))
} else {
    inner
};
per_boundary.push(wrapped);
```

**Step 2: Build the blueprint provider in `run()` before calling `setup_neural_boundaries`**

Replace the "todo" branch in the `opt_out_provider` match from Task 5:

```rust
"blueprint-cbv" => {
    // Build one BlueprintCbvOptOut per boundary. All boundaries share the
    // same abstract_node_idx (since the compare-solve spot is at one node);
    // lookup via SolveBoundaryEvaluator's logic pattern.
    let half_pot = (pot as f32) / 2.0;
    // cbv_context is available from the session (see line ~730 in run).
    let abstract_node_idx = session.abstract_node_idx_for_current_spot()
        .ok_or("compare-solve: could not map spot to abstract blueprint node")?;
    let private_cards: [Vec<(u8, u8)>; 2] = [
        game.private_cards(0).to_vec(),
        game.private_cards(1).to_vec(),
    ];
    Some(Arc::new(poker_solver_tauri::gadget::BlueprintCbvOptOut::from_cbv_context(
        &ctx,
        abstract_node_idx as u32,
        &board,  // Vec<u8> — convert board strings if needed
        &private_cards,
        half_pot,
    )))
}
```

**Note:** `session.abstract_node_idx_for_current_spot()` is a method on `GameSession` — if it doesn't exist yet, add it (exposing whatever `session.node_idx` returns combined with any needed mapping to the abstract tree). Grep `crates/tauri-app/src/game_session.rs` for how `SolveBoundaryEvaluator::new` gets its `abstract_start_node` and copy that pattern.

**Step 3: Pass provider to both setup fns**

In `run()` where `setup_exact_subtree_boundaries` and `setup_neural_boundaries` are called (around line 840), feed the same provider. The exact-subtree setup already takes it; neural now takes it too.

**Step 4: Test — full build + existing tests**

Run: `cargo test -p poker-solver-trainer` and `cargo test -p poker-solver-tauri`
Expected: all green, including the gadget skeleton tests.

**Step 5: Commit**

```bash
# Reviews
touch .reviews-done
git add crates/trainer/src/compare_solve.rs
git commit -m "feat(compare-solve): wire gadget into neural boundary path"
```

---

### Task 7: Add `enable_gadget` to Tauri `game_solve_core`

**Files:**
- Modify: `crates/tauri-app/src/game_session.rs` — `game_solve_core` signature (around line 1806), `setup_neural_boundaries` (around line 2135).

**Step 1: Add the parameter to `game_solve_core` and the `game_solve` Tauri command**

Search for `game_solve_core` definition (around line 1806) and add `enable_gadget: bool` as the last parameter. Do the same for the `game_solve` Tauri command wrapper in the same file.

**Step 2: Modify `setup_neural_boundaries` (game_session.rs:2135) to accept a provider**

Change:

```rust
fn setup_neural_boundaries(game: &mut PostFlopGame, model_path: &str)
```

to:

```rust
fn setup_neural_boundaries(
    game: &mut PostFlopGame,
    model_path: &str,
    opt_out_provider: Option<Arc<dyn OptOutProvider>>,
)
```

And apply the same wrap-when-Some pattern from Task 6.

**Step 3: In `game_solve_core`, construct the provider when `enable_gadget` is true**

After `let boundary_cut = resolve_street_boundary(...)` (around line 1860), add:

```rust
let opt_out_provider: Option<Arc<dyn OptOutProvider>> = if enable_gadget {
    if boundary_cut.is_none() {
        eprintln!("[solve] enable_gadget=true but no boundary cut resolved — gadget will have no effect");
        None
    } else {
        let cbv_ctx = postflop_state.cbv_context.read().clone().ok_or(
            "enable_gadget=true but no CbvContext loaded (blueprint must include CBV tables)"
        )?;
        let private_cards: [Vec<(u8, u8)>; 2] = [
            oop_private.clone(), ip_private.clone(),
        ];
        let half_pot = (pot as f32) / 2.0;
        let abstract_node_idx = current_node as u32;
        Some(Arc::new(poker_solver_tauri::gadget::BlueprintCbvOptOut::from_cbv_context(
            &cbv_ctx, abstract_node_idx, &board, &private_cards, half_pot,
        )))
    }
} else {
    None
};
```

**Step 4: Pass provider into both setup fns (neural + exact_subtree)**

Where `setup_neural_boundaries(&mut game, model_path)` is called in this file, change to `setup_neural_boundaries(&mut game, model_path, opt_out_provider.clone())`. Same for `setup_exact_subtree_boundaries` (which already has the parameter).

**Step 5: Build**

Run: `cargo build -p poker-solver-tauri`
Expected: clean. Tauri command signature changes may cascade into `game_solve` JS binding — if the build complains about missing callers, add `enable_gadget` to the Tauri IPC payload struct in the frontend in Task 10.

**Step 6: Commit**

```bash
# Reviews
touch .reviews-done
git add crates/tauri-app/src/game_session.rs
git commit -m "feat(tauri): enable_gadget flag in game_solve_core, wired to neural path"
```

---

## Phase 3 — Adapters: Tauri UI

### Task 8: Add `enable_safe_resolving` to `GlobalConfig`

**Files:**
- Modify: `frontend/src/types.ts` — `GlobalConfig` interface
- Modify: `frontend/src/useGlobalConfig.ts` — default

**Step 1: Add to `GlobalConfig`**

In `frontend/src/types.ts`, find `interface GlobalConfig` and add:

```typescript
enable_safe_resolving: boolean;
```

**Step 2: Add to `DEFAULT_CONFIG` in `useGlobalConfig.ts`**

Add `enable_safe_resolving: false,` to the default config object.

**Step 3: Type-check**

Run: `cd frontend && npm run build` (or `npx tsc --noEmit`).
Expected: clean.

**Step 4: Commit**

```bash
touch .reviews-done  # doc-like change; reviews optional but safe
git add frontend/src/types.ts frontend/src/useGlobalConfig.ts
git commit -m "feat(frontend): add enable_safe_resolving to GlobalConfig"
```

---

### Task 9: Add Settings checkbox

**Files:**
- Modify: `frontend/src/Settings.tsx` — boundary-config section

**Step 1: Add the checkbox**

Locate the boundary-config rendering in `Settings.tsx` (search for `boundary_mode` or the existing `flop_boundary_mode` dropdown). Add below the per-street dropdowns:

```tsx
<label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.5rem' }}>
    <input
        type="checkbox"
        checked={config.enable_safe_resolving}
        disabled={!hasAnyCut(config)}
        onChange={(e) => updateConfig({ enable_safe_resolving: e.target.checked })}
    />
    <span>Safe re-solving (Libratus gadget)</span>
</label>
{!hasAnyCut(config) && (
    <div style={{ fontSize: '0.85em', color: '#888', marginLeft: '1.8rem' }}>
        Enable a per-street cut mode (cfvnet or exact_subtree) to use the gadget.
    </div>
)}
```

**Step 2: Add helper `hasAnyCut`**

At the top of the same file (or in a utilities module):

```typescript
function hasAnyCut(cfg: GlobalConfig): boolean {
    return (
        cfg.flop_boundary_mode !== 'exact' ||
        cfg.turn_boundary_mode !== 'exact' ||
        cfg.river_boundary_mode !== 'exact'
    );
}
```

**Step 3: Visual check**

Run: `cargo run -p poker-solver-devserver &` then `cd frontend && npm run dev`. Open localhost:5173 → Settings panel. Verify checkbox appears, greys out when all three per-street modes are "exact", enables when any is cfvnet/exact_subtree.

Manual test — no automated test for UI in this iteration.

**Step 4: Commit**

```bash
touch .reviews-done
git add frontend/src/Settings.tsx
git commit -m "feat(settings): add Safe re-solving checkbox"
```

---

### Task 10: Thread `enableGadget` through `buildSolveParams`

**Files:**
- Modify: `frontend/src/strategy-tabs.ts` — `SolveParams` + `buildSolveParams`
- Check: how `game_solve` is invoked in `GameExplorer.tsx` (the invoke call needs to receive `enableGadget`)

**Step 1: Add `enableGadget` to `SolveParams`**

In `strategy-tabs.ts`:

```typescript
export interface SolveParams {
    mode: SolveMode;
    maxIterations: number;
    targetExploitability: number;
    matrixSnapshotInterval: number;
    rangeClampThreshold: number;
    streetBoundaryConfig: StreetBoundaryConfig;
    traceBoundaries: string;
    traceIters: string;
    enableGadget: boolean;  // NEW
}
```

**Step 2: Set it from the config in `buildSolveParams`**

```typescript
export function buildSolveParams(
    mode: SolveMode,
    config: Record<string, unknown>,
): SolveParams {
    return {
        // ...existing fields...
        enableGadget: (config.enable_safe_resolving as boolean | undefined) ?? false,
    };
}
```

**Step 3: Check the invoke call site**

Grep: `grep -rn 'game_solve' frontend/src`
Look for the `invoke('game_solve', ...)` call (likely `GameExplorer.tsx` or `useSolve.ts`). Include `enableGadget` in the payload so it reaches `game_solve_core` as `enable_gadget` (Tauri does camelCase → snake_case).

**Step 4: Type-check**

Run: `cd frontend && npm run build`
Expected: clean.

**Step 5: End-to-end manual test**

- `cargo build --release -p poker-solver-tauri` (or whatever builds the Tauri app)
- Launch the app
- Load a blueprint bundle that has CbvTable
- Navigate to the `JhTh9h|...|7d` spot, cfvnet boundary mode, run a solve with the checkbox OFF
- Run a solve with the checkbox ON — observe `eprintln` in stderr confirming gadget is wired: `[solve] enable_gadget=true, wrapped N boundaries`
- Compare the resulting matrix cells to verify they differ

**Step 6: Commit**

```bash
touch .reviews-done
git add frontend/src/strategy-tabs.ts frontend/src/GameExplorer.tsx
git commit -m "feat(frontend): thread enable_gadget through game_solve invoke"
```

---

## Phase 4 — Validation: iterations 9–11

### Task 11: Iter 9 baseline — cfvnet without gadget

**Files:**
- Modify: `docs/progress/2026-04-22-subgame-exact-parity.md` (append new entry)

**Step 1: Run the harness**

```bash
./target/release/poker-solver-trainer compare-solve \
    --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
    --snapshot snapshot_0013 \
    --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
    --river-boundary cfvnet \
    --river-model ./local_data/models/cfvnet_river_py_v2/model.onnx \
    --iters 150 \
    --tolerance 0.001 2>&1 | tee /tmp/iter9-cfvnet-baseline.log
```

(Confirm the model path — look at `local_data/models/` for the right filename.)

Dispatch via subagent if this takes >5 min wall time.

**Step 2: Extract metrics**

From the log, capture: `exact_exp`, `subgame_exp`, `mean_mass`, `max_mass`, `worst_delta`, `worst_cell`, action-class bias, top-3 hands.

**Step 3: Append progress entry**

Add to `docs/progress/2026-04-22-subgame-exact-parity.md`:

```markdown
## Iteration 9 — 2026-04-23 (cfvnet baseline, no gadget)

**Approach:** river-boundary=cfvnet (pre-trained), no gadget.

**Result:** exact_exp=X mbb, subgame_exp=Y mbb, worst_delta=Δ, worst_cell=...
Status: FAIL (established baseline for gadget A/B comparison)

**Wall time:** Nm.

**Commits:** none (measurement only).
```

**Step 4: Commit**

```bash
touch .reviews-done
git add docs/progress/2026-04-22-subgame-exact-parity.md
git commit -m "docs(progress): iteration 9 — cfvnet baseline (no gadget)"
```

---

### Task 12: Iter 10 — cfvnet with `--gadget --gadget-provider=blueprint-cbv`

**Step 1: Run**

```bash
./target/release/poker-solver-trainer compare-solve \
    --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
    --snapshot snapshot_0013 \
    --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
    --river-boundary cfvnet \
    --river-model ./local_data/models/cfvnet_river_py_v2/model.onnx \
    --gadget \
    --gadget-provider blueprint-cbv \
    --iters 150 \
    --tolerance 0.001 2>&1 | tee /tmp/iter10-cfvnet-gadget.log
```

**Step 2: Compare to iter 9**

Use `scripts/trace_diff.py` to compute the strategy-delta reduction (if any).

**Step 3: Append progress entry**

```markdown
## Iteration 10 — 2026-04-23 (cfvnet + BlueprintCbvOptOut gadget)

**Approach:** river-boundary=cfvnet + --gadget --gadget-provider=blueprint-cbv.

**Result:** exact_exp=X mbb, subgame_exp=Y mbb, worst_delta=Δ, worst_cell=...
**Delta vs iter 9:** subgame_exp ΔX mbb, worst_delta Δ.
Status: PASS / FAIL

**Verdict:** gadget closes/does-not-close the gap.

**Commits:** feat commits from Phase 1–3.
```

**Step 4: Commit**

```bash
touch .reviews-done
git add docs/progress/2026-04-22-subgame-exact-parity.md
git commit -m "docs(progress): iteration 10 — cfvnet + blueprint-cbv gadget"
```

---

### Task 13: Iter 11 — sanity check (gadget with dominated opt-out)

**Step 1: Run with `--gadget-constant -999.0`**

```bash
./target/release/poker-solver-trainer compare-solve \
    --bundle ./local_data/blueprints/1k_100bb_brdcfr_v2 \
    --snapshot snapshot_0013 \
    --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
    --river-boundary cfvnet \
    --river-model ./local_data/models/cfvnet_river_py_v2/model.onnx \
    --gadget \
    --gadget-provider constant \
    --gadget-constant -999.0 \
    --iters 150 \
    --tolerance 0.001 2>&1 | tee /tmp/iter11-sanity.log
```

**Step 2: Verify match to iter 9**

Values should match iter 9 to within numerical precision (dominated opt-out = no-op). Any non-trivial difference is a bug in the wrapper — investigate before continuing.

**Step 3: Append progress entry and mark MVP bean complete**

```markdown
## Iteration 11 — 2026-04-23 (sanity check, dominated opt-out)

**Approach:** river-boundary=cfvnet + --gadget --gadget-provider=constant --gadget-constant=-999.0.

**Result:** matches iter 9 within ε (expected).
Status: PASS (sanity).
```

Then mark the bean complete:

```bash
beans update --json poker_solver_rust-lay5 -s completed \
    --body-append $'\n## Summary of Changes\n\n- BlueprintCbvOptOut implemented and wired into compare-solve cfvnet path and Tauri game_solve_core.\n- Settings UI checkbox added.\n- Iter 9/10/11 captured in docs/progress/2026-04-22-subgame-exact-parity.md.\n- Gadget closes/does-not-close the gap (see iter 10).\n\nFollow-up: bean akg3 (DeepStack-proper cfvnet retrain).'
```

**Step 4: Commit**

```bash
touch .reviews-done
git add docs/progress/2026-04-22-subgame-exact-parity.md .beans/poker_solver_rust-lay5--*
git commit -m "docs(progress): iteration 11 — sanity check + close MVP bean"
```

---

## Summary

- **Phase 1** (Tasks 1–4): `BlueprintCbvOptOut` with TDD. ~1–2 hours.
- **Phase 2** (Tasks 5–7): CLI + compare-solve wiring + Tauri wiring. ~1 hour.
- **Phase 3** (Tasks 8–10): TS types + Settings UI + invoke params. ~45 min.
- **Phase 4** (Tasks 11–13): 3 harness runs + progress doc. ~30 min (plus wall time for each solve).

Every commit requires `touch .reviews-done` (and strictly a review dispatch for non-trivial changes) per the pre-commit hook.

---

**Plan complete and saved to `docs/plans/2026-04-23-libratus-gadget-impl-plan.md`. Two execution options:**

**1. Subagent-Driven (this session)** — Agent teams with parallel implementer-reviewer streams, fast throughput.

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints.

**Which approach?**
