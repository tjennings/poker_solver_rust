# Multi-SPR Postflop Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a `PostflopAbstraction` per configured SPR and have the preflop solver select the closest SPR model at each showdown terminal.

**Architecture:** `PostflopState` changes from holding one `PostflopAbstraction` to a sorted `Vec<PostflopAbstraction>`. At showdown terminals, `postflop_showdown_value` computes the actual SPR and selects the closest model by absolute distance. Bundle persistence saves one sub-bundle per SPR.

**Tech Stack:** Rust, serde, bincode, rayon (existing dependencies only)

---

### Task 1: Add `select_closest_spr` helper with tests

**Files:**
- Modify: `crates/core/src/preflop/solver.rs`

This is a pure function that can be tested in isolation before any structural changes.

**Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block at the bottom of `solver.rs`:

```rust
#[timed_test]
fn select_closest_spr_picks_nearest() {
    let sprs = [2.0, 6.0, 20.0];
    assert_eq!(super::select_closest_spr(&sprs, 1.0), 0);
    assert_eq!(super::select_closest_spr(&sprs, 3.0), 0);
    assert_eq!(super::select_closest_spr(&sprs, 4.5), 1);
    assert_eq!(super::select_closest_spr(&sprs, 5.0), 1);
    assert_eq!(super::select_closest_spr(&sprs, 12.0), 1);
    assert_eq!(super::select_closest_spr(&sprs, 13.5), 2);
    assert_eq!(super::select_closest_spr(&sprs, 50.0), 2);
}

#[timed_test]
fn select_closest_spr_single_element() {
    assert_eq!(super::select_closest_spr(&[3.5], 0.0), 0);
    assert_eq!(super::select_closest_spr(&[3.5], 100.0), 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core select_closest_spr`
Expected: FAIL — function does not exist

**Step 3: Write minimal implementation**

Add above `postflop_showdown_value` in `solver.rs`:

```rust
/// Select the index of the closest SPR model to `actual_spr`.
///
/// `sprs` must be non-empty. Returns the index into `sprs` with the
/// smallest absolute distance to `actual_spr`.
fn select_closest_spr(sprs: &[f64], actual_spr: f64) -> usize {
    sprs.iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - actual_spr).abs().total_cmp(&(*b - actual_spr).abs())
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core select_closest_spr`
Expected: PASS

**Step 5: Commit**

```
feat: add select_closest_spr helper for multi-SPR model selection
```

---

### Task 2: Change `PostflopState` to hold multiple abstractions

**Files:**
- Modify: `crates/core/src/preflop/solver.rs` (struct + `attach_postflop` + field `postflop`)

**Step 1: Write the failing test**

Add to the test module in `solver.rs`:

```rust
/// Multi-SPR: postflop_showdown_value should select the closest SPR model.
#[timed_test]
fn postflop_showdown_value_selects_closest_spr() {
    use crate::preflop::postflop_abstraction::{PostflopAbstraction, PostflopValues};
    use crate::preflop::postflop_tree::{PostflopNode, PostflopTree, PostflopTerminalType, PotType};

    let n = 2;

    // SPR=2 model: hand 0 vs hand 1 = +0.1 (shallow play)
    let mut avg2 = vec![0.0; 2 * n * n];
    avg2[0 * n * n + 0 * n + 1] = 0.1;
    let abs2 = PostflopAbstraction {
        tree: PostflopTree {
            nodes: vec![PostflopNode::Terminal {
                terminal_type: PostflopTerminalType::Showdown,
                pot_fraction: 1.0,
            }],
            pot_type: PotType::Raised,
            spr: 2.0,
        },
        values: PostflopValues { values: vec![], num_buckets: n, num_flops: 0 },
        hand_avg_values: avg2,
        spr: 2.0,
        flops: vec![],
    };

    // SPR=10 model: hand 0 vs hand 1 = +0.4 (deep play)
    let mut avg10 = vec![0.0; 2 * n * n];
    avg10[0 * n * n + 0 * n + 1] = 0.4;
    let abs10 = PostflopAbstraction {
        tree: PostflopTree {
            nodes: vec![PostflopNode::Terminal {
                terminal_type: PostflopTerminalType::Showdown,
                pot_fraction: 1.0,
            }],
            pot_type: PotType::Raised,
            spr: 10.0,
        },
        values: PostflopValues { values: vec![], num_buckets: n, num_flops: 0 },
        hand_avg_values: avg10,
        spr: 10.0,
        flops: vec![],
    };

    let pf_state = PostflopState {
        abstractions: vec![abs2, abs10],
        raise_counts: vec![1], // raised pot
    };

    let pot = 10;
    let hero_inv = 5.0;
    let equity = 0.5;

    // Stacks = [15, 15]. After investing 5 each, remaining = 10.
    // actual_spr = 10/10 = 1.0 → closest to SPR=2 model → EV from 0.1 fraction
    let stacks_shallow = [15, 15];
    let ev_shallow = postflop_showdown_value(&pf_state, 0, pot, hero_inv, 0, 1, 0, equity, stacks_shallow);
    // model_value = 0.1 * 10 + (10/2 - 5) = 1.0, but actual_spr(1.0) < model_spr(2.0)
    // ratio = 1.0/2.0 = 0.5, eq_value = 0.5*10 - 5 = 0.0
    // interpolated = 0.0 + (1.0 - 0.0) * 0.5 = 0.5
    assert!(
        (ev_shallow - 0.5).abs() < 1e-10,
        "shallow stacks should use SPR=2 model: got {ev_shallow}, expected 0.5"
    );

    // Stacks = [100, 100]. After investing 5 each, remaining = 95.
    // actual_spr = 95/10 = 9.5 → closest to SPR=10 model → EV from 0.4 fraction
    let stacks_deep = [100, 100];
    let ev_deep = postflop_showdown_value(&pf_state, 0, pot, hero_inv, 0, 1, 0, equity, stacks_deep);
    // model_value = 0.4 * 10 + (10/2 - 5) = 4.0, actual_spr(9.5) < model_spr(10)
    // ratio = 9.5/10 = 0.95, eq_value = 0.5*10 - 5 = 0.0
    // interpolated = 0.0 + (4.0 - 0.0) * 0.95 = 3.8
    assert!(
        (ev_deep - 3.8).abs() < 1e-10,
        "deep stacks should use SPR=10 model: got {ev_deep}, expected 3.8"
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core postflop_showdown_value_selects_closest_spr`
Expected: FAIL — `PostflopState` has no field `abstractions`

**Step 3: Make the structural changes**

In `solver.rs`, change `PostflopState`:

```rust
pub(crate) struct PostflopState {
    pub(crate) abstractions: Vec<PostflopAbstraction>,
    pub(crate) raise_counts: Vec<u8>,
}
```

Change `attach_postflop`:

```rust
pub fn attach_postflop(&mut self, abstractions: Vec<PostflopAbstraction>, _config: &PreflopConfig) {
    let raise_counts = precompute_raise_counts(&self.tree);
    self.postflop = Some(PostflopState {
        abstractions,
        raise_counts,
    });
}
```

Change `postflop_showdown_value` to select the closest model. Replace the existing lines that read from `pf_state.abstraction`:

```rust
pub(crate) fn postflop_showdown_value(
    pf_state: &PostflopState,
    preflop_node_idx: u32,
    pot: u32,
    hero_inv: f64,
    hero_hand: u16,
    opp_hand: u16,
    hero_pos: u8,
    equity: f64,
    stacks: [u32; 2],
) -> f64 {
    let pot_f = f64::from(pot);
    let eq_value = equity * pot_f - hero_inv;

    let raise_count = pf_state
        .raise_counts
        .get(preflop_node_idx as usize)
        .copied()
        .unwrap_or(0);
    if raise_count == 0 {
        return eq_value;
    }

    // Compute actual SPR at this terminal.
    let opp_inv = pot_f - hero_inv;
    let hero_remaining = f64::from(stacks[hero_pos as usize]) - hero_inv;
    let opp_remaining = f64::from(stacks[1 - hero_pos as usize]) - opp_inv;
    let effective_remaining = hero_remaining.min(opp_remaining).max(0.0);
    let actual_spr = if pot > 0 { effective_remaining / pot_f } else { 0.0 };

    // Select the closest SPR model.
    let sprs: Vec<f64> = pf_state.abstractions.iter().map(|a| a.spr).collect();
    let idx = select_closest_spr(&sprs, actual_spr);
    let selected = &pf_state.abstractions[idx];

    let pf_ev_frac = selected.avg_ev(
        hero_pos, hero_hand as usize, opp_hand as usize,
    );
    let model_value = pf_ev_frac * pot_f + (pot_f / 2.0 - hero_inv);

    let model_spr = selected.spr;
    if model_spr <= 0.0 || actual_spr >= model_spr {
        return model_value;
    }

    let ratio = actual_spr / model_spr;
    eq_value + (model_value - eq_value) * ratio
}
```

**Step 4: Fix existing tests that construct `PostflopState`**

Update the two existing tests (`postflop_showdown_value_position_mapping` and `postflop_showdown_value_limped_pot_uses_equity`) to use the new field name. In each, change:

```rust
// Old:
let pf_state = PostflopState {
    abstraction,
    raise_counts: vec![...],
};

// New:
let pf_state = PostflopState {
    abstractions: vec![abstraction],
    raise_counts: vec![...],
};
```

**Step 5: Run all tests to verify they pass**

Run: `cargo test -p poker-solver-core postflop_showdown_value`
Expected: all 3 tests PASS

**Step 6: Commit**

```
feat: PostflopState holds Vec<PostflopAbstraction> with closest-SPR selection
```

---

### Task 3: Update `PostflopAbstraction::build` to accept explicit SPR

**Files:**
- Modify: `crates/core/src/preflop/postflop_abstraction.rs`

Currently `build()` always uses `config.primary_spr()`. We need it to accept an explicit SPR so the caller can loop.

**Step 1: Write the failing test**

Add to the test module in `postflop_abstraction.rs`:

```rust
#[timed_test]
fn build_with_explicit_spr() {
    let config = PostflopModelConfig {
        postflop_sprs: vec![3.5, 6.0],
        max_flop_boards: 1,
        postflop_solve_iterations: 10,
        ..PostflopModelConfig::exhaustive_fast()
    };
    // Build with SPR=6.0, NOT primary_spr()=3.5
    let result = PostflopAbstraction::build_for_spr(&config, 6.0, None, |_| {});
    assert!(result.is_ok());
    let abs = result.unwrap();
    assert!((abs.spr - 6.0).abs() < 1e-9, "spr should be 6.0, got {}", abs.spr);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core build_with_explicit_spr`
Expected: FAIL — method `build_for_spr` does not exist

**Step 3: Add `build_for_spr` method**

In `postflop_abstraction.rs`, add alongside the existing `build` method:

```rust
/// Build postflop data for a specific SPR value.
///
/// Like `build()` but uses the given `spr` instead of `config.primary_spr()`.
pub fn build_for_spr(
    config: &PostflopModelConfig,
    spr: f64,
    _equity_table: Option<&super::equity::EquityTable>,
    on_progress: impl Fn(BuildPhase) + Sync,
) -> Result<Self, PostflopAbstractionError> {
    let flops = if let Some(ref names) = config.fixed_flops {
        parse_flops(names).map_err(PostflopAbstractionError::InvalidConfig)?
    } else {
        sample_canonical_flops(config.max_flop_boards)
    };
    let tree = PostflopTree::build_with_spr(config, spr)?;
    let node_streets = annotate_streets(&tree);
    let layout = PostflopLayout::build(&tree, &node_streets, NUM_CANONICAL_HANDS, NUM_CANONICAL_HANDS, NUM_CANONICAL_HANDS);

    let values = match config.solve_type {
        PostflopSolveType::Mccfr => build_mccfr(config, &tree, &layout, &node_streets, &flops, &on_progress),
        PostflopSolveType::Exhaustive => build_exhaustive(config, &tree, &layout, &node_streets, &flops, &on_progress),
    };
    on_progress(BuildPhase::ComputingValues);
    let flop_weights = crate::flops::lookup_flop_weights(&flops);
    let hand_avg_values = compute_hand_avg_values(&values, &flop_weights);
    Ok(Self { tree, values, hand_avg_values, spr, flops })
}
```

Then refactor the existing `build` method to delegate:

```rust
pub fn build(
    config: &PostflopModelConfig,
    equity_table: Option<&super::equity::EquityTable>,
    on_progress: impl Fn(BuildPhase) + Sync,
) -> Result<Self, PostflopAbstractionError> {
    Self::build_for_spr(config, config.primary_spr(), equity_table, on_progress)
}
```

Also add `build_from_cached_spr` for bundle loading with explicit SPR:

```rust
pub fn build_from_cached_spr(
    config: &PostflopModelConfig,
    spr: f64,
    values: PostflopValues,
    hand_avg_values: Vec<f64>,
    flops: Vec<[Card; 3]>,
) -> Result<Self, PostflopAbstractionError> {
    let tree = PostflopTree::build_with_spr(config, spr)?;
    Ok(Self { tree, values, hand_avg_values, spr, flops })
}
```

Update existing `build_from_cached` to delegate:

```rust
pub fn build_from_cached(
    config: &PostflopModelConfig,
    values: PostflopValues,
    hand_avg_values: Vec<f64>,
    flops: Vec<[Card; 3]>,
) -> Result<Self, PostflopAbstractionError> {
    Self::build_from_cached_spr(config, config.primary_spr(), values, hand_avg_values, flops)
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core build_with_explicit_spr`
Expected: PASS

Run: `cargo test -p poker-solver-core postflop_abstraction`
Expected: all existing tests PASS (build delegates to build_for_spr)

**Step 5: Commit**

```
feat: add build_for_spr to PostflopAbstraction for per-SPR construction
```

---

### Task 4: Update `PostflopBundle` for multi-SPR persistence

**Files:**
- Modify: `crates/core/src/preflop/postflop_bundle.rs`

Save/load one sub-bundle per SPR. Backward-compatible: old single-file bundles still load.

**Step 1: Write the failing test**

Add to the test module in `postflop_bundle.rs`:

```rust
#[timed_test]
fn multi_spr_bundle_roundtrip() {
    let config = PostflopModelConfig::fast();
    let values = PostflopValues::from_raw(vec![0.5; 8], 1, 1);
    let flops: Vec<[Card; 3]> = vec![];
    let flop_weights = crate::flops::lookup_flop_weights(&flops);
    let hand_avg = compute_hand_avg_values(&values, &flop_weights);

    let abs1 = PostflopAbstraction::build_from_cached_spr(
        &config, 2.0,
        PostflopValues::from_raw(vec![0.2; 8], 1, 1),
        hand_avg.clone(), vec![],
    ).unwrap();
    let abs2 = PostflopAbstraction::build_from_cached_spr(
        &config, 6.0,
        PostflopValues::from_raw(vec![0.6; 8], 1, 1),
        hand_avg, vec![],
    ).unwrap();

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("multi_spr");

    PostflopBundle::save_multi(&config, &[&abs1, &abs2], &path).unwrap();
    let loaded = PostflopBundle::load_multi(&config, &path).unwrap();

    assert_eq!(loaded.len(), 2);
    assert!((loaded[0].spr - 2.0).abs() < 1e-9);
    assert!((loaded[1].spr - 6.0).abs() < 1e-9);
}

#[timed_test]
fn legacy_single_bundle_loads_as_vec() {
    // Save in old format, load with new multi loader
    let bundle = minimal_bundle();
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("legacy");
    bundle.save(&path).unwrap();

    let config = PostflopModelConfig::fast();
    let loaded = PostflopBundle::load_multi(&config, &path).unwrap();
    assert_eq!(loaded.len(), 1);
    assert!((loaded[0].spr - 3.5).abs() < 1e-9);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core multi_spr_bundle`
Expected: FAIL — methods don't exist

**Step 3: Implement multi-SPR persistence**

Add to `PostflopBundle`:

```rust
/// Save multiple SPR abstractions to a directory.
///
/// Layout:
/// ```text
/// dir/
/// ├── config.yaml
/// ├── spr_2.0/solve.bin
/// ├── spr_6.0/solve.bin
/// └── spr_20.0/solve.bin
/// ```
pub fn save_multi(
    config: &PostflopModelConfig,
    abstractions: &[&PostflopAbstraction],
    dir: &Path,
) -> Result<(), std::io::Error> {
    fs::create_dir_all(dir)?;
    let config_yaml = serde_yaml::to_string(config).map_err(std::io::Error::other)?;
    fs::write(dir.join("config.yaml"), config_yaml)?;

    for abs in abstractions {
        let spr_dir = dir.join(format!("spr_{}", abs.spr));
        fs::create_dir_all(&spr_dir)?;
        let data = PostflopBundleData {
            values: PostflopValues::from_raw(
                abs.values.values.clone(),
                abs.values.num_buckets,
                abs.values.num_flops,
            ),
            hand_avg_values: abs.hand_avg_values.clone(),
            flops: abs.flops.clone(),
            spr: abs.spr,
        };
        let bytes = bincode::serialize(&data).map_err(std::io::Error::other)?;
        fs::write(spr_dir.join("solve.bin"), bytes)?;
    }
    Ok(())
}

/// Load multi-SPR abstractions from a directory.
///
/// Handles both new multi-SPR layout (`spr_*/solve.bin`) and legacy
/// single-file layout (`solve.bin` at root).
pub fn load_multi(
    config: &PostflopModelConfig,
    dir: &Path,
) -> Result<Vec<PostflopAbstraction>, std::io::Error> {
    // Try new layout: look for spr_* subdirectories
    let mut spr_dirs: Vec<_> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name().to_str().map_or(false, |n| n.starts_with("spr_"))
                && e.path().join("solve.bin").exists()
        })
        .collect();

    if !spr_dirs.is_empty() {
        spr_dirs.sort_by_key(|e| e.file_name());
        let mut result = Vec::with_capacity(spr_dirs.len());
        for entry in &spr_dirs {
            let data_bytes = fs::read(entry.path().join("solve.bin"))?;
            let data: PostflopBundleData =
                bincode::deserialize(&data_bytes).map_err(std::io::Error::other)?;
            let flop_weights = crate::flops::lookup_flop_weights(&data.flops);
            let hand_avg = compute_hand_avg_values(&data.values, &flop_weights);
            let abs = PostflopAbstraction::build_from_cached_spr(
                config, data.spr, data.values, hand_avg, data.flops,
            ).map_err(|e| std::io::Error::other(format!("{e}")))?;
            result.push(abs);
        }
        return Ok(result);
    }

    // Legacy: single solve.bin at root
    if dir.join("solve.bin").exists() {
        let bundle = Self::load(dir)?;
        let abs = bundle.into_abstraction()
            .map_err(|e| std::io::Error::other(format!("{e}")))?;
        return Ok(vec![abs]);
    }

    Err(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        format!("no postflop bundle found in {}", dir.display()),
    ))
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core postflop_bundle`
Expected: all tests PASS (including the two new ones and all existing)

**Step 5: Commit**

```
feat: multi-SPR postflop bundle persistence with legacy backward compat
```

---

### Task 5: Update trainer to build and attach multiple SPR models

**Files:**
- Modify: `crates/trainer/src/main.rs`

This is the integration task — wiring everything together.

**Step 1: Update `build_postflop_with_progress` to loop over SPRs**

Change the function signature and body. Currently it calls `PostflopAbstraction::build(pf_config, equity, ...)` once. Change to:

```rust
fn build_postflop_with_progress(
    pf_config: &PostflopModelConfig,
    equity: Option<&EquityTable>,
) -> Result<Vec<PostflopAbstraction>, String> {
```

Inside, loop over `pf_config.postflop_sprs`:

```rust
let mut abstractions = Vec::with_capacity(pf_config.postflop_sprs.len());
for (spr_idx, &spr) in pf_config.postflop_sprs.iter().enumerate() {
    eprintln!("Building postflop model for SPR={spr} ({}/{})...",
        spr_idx + 1, pf_config.postflop_sprs.len());
    // ... existing progress bar setup ...
    let abstraction = PostflopAbstraction::build_for_spr(
        pf_config, spr, equity, |phase| { /* existing progress callback */ },
    ).map_err(|e| format!("postflop abstraction SPR={spr}: {e}"))?;
    abstractions.push(abstraction);
}
Ok(abstractions)
```

**Step 2: Update `run_solve_preflop` postflop loading/saving/attaching**

In the postflop build/load section (~line 1140-1193):

```rust
// Build or load
let postflop: Option<Vec<PostflopAbstraction>> = if let Some(bundle_path) = &postflop_model_path {
    // Load multi-SPR bundle
    let abstractions = PostflopBundle::load_multi(
        config.postflop_model.as_ref().unwrap_or(&PostflopModelConfig::default()),
        bundle_path,
    ).map_err(|e| format!("failed to load postflop bundle: {e}"))?;
    Some(abstractions)
} else if let Some(pf_config) = &config.postflop_model {
    Some(build_postflop_with_progress(pf_config, Some(&equity))?)
} else {
    None
};

// Save multi-SPR bundle
if let Some(ref abstractions) = postflop {
    if let Some(pf_config) = &config.postflop_model {
        let pf_dir = output.join("postflop");
        let refs: Vec<&PostflopAbstraction> = abstractions.iter().collect();
        PostflopBundle::save_multi(pf_config, &refs, &pf_dir)?;
        eprintln!("Postflop bundle saved to {} ({} SPR models)", pf_dir.display(), refs.len());
    }
}

// Clone hand_avg_values from first model (for explorer compatibility)
let hand_avg_values = postflop.as_ref()
    .and_then(|abs| abs.first())
    .map(|a| a.hand_avg_values.clone());

// Diagnostics on first model
if let Some(ref abstractions) = postflop {
    if !ev_diagnostic_hands.is_empty() {
        if let Some(first) = abstractions.first() {
            print_postflop_ev_diagnostics(first, &ev_diagnostic_hands);
        }
    }
}

// Attach all models
if let Some(abstractions) = postflop {
    solver.attach_postflop(abstractions, &config);
}
```

**Step 3: Update `run_solve_postflop` for multi-SPR**

```rust
fn run_solve_postflop(config_path: &Path, output: &Path) -> Result<(), Box<dyn Error>> {
    let yaml = std::fs::read_to_string(config_path)?;
    let training: PreflopTrainingConfig = serde_yaml::from_str(&yaml)?;
    let pf_config = training.game.postflop_model
        .ok_or("config file has no postflop_model section")?;

    let abstractions = build_postflop_with_progress(&pf_config, None)?;

    let refs: Vec<&PostflopAbstraction> = abstractions.iter().collect();
    PostflopBundle::save_multi(&pf_config, &refs, output)?;
    eprintln!(
        "Postflop bundle saved to {} ({} SPR models)",
        output.display(), refs.len(),
    );
    Ok(())
}
```

**Step 4: Build and run tests**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles cleanly

Run: `cargo test -p poker-solver-core`
Expected: all tests PASS

**Step 5: Commit**

```
feat: trainer builds and persists multi-SPR postflop models
```

---

### Task 6: Update exploitability module for multi-SPR

**Files:**
- Modify: `crates/core/src/preflop/exploitability.rs`

The exploitability module passes `postflop: Option<&PostflopState>` through to `postflop_showdown_value`. Since `PostflopState` changed internally but the function signature of `postflop_showdown_value` is the same, this should just compile. Verify.

**Step 1: Build and test**

Run: `cargo test -p poker-solver-core`
Expected: PASS — exploitability module doesn't construct `PostflopState` directly, it receives a reference from the solver.

If compilation fails, fix any field access that references `pf_state.abstraction` (singular). Search for `.abstraction` in `exploitability.rs` and update to `.abstractions`.

**Step 2: Commit (only if changes were needed)**

```
fix: update exploitability module for multi-SPR PostflopState
```

---

### Task 7: Update explorer bundle loading for multi-SPR

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`

The explorer loads `PostflopBundle` to get `hand_avg_values` for display. It only uses the first model's hand-averaged values (for the equity panel). Update the loading code to use `load_multi` and take the first model.

**Step 1: Update the two loading sites**

At line ~260-274 (load_bundle_core):
```rust
let hand_avg_values = preflop.hand_avg_values.clone().or_else(|| {
    let postflop_dir = bp.join("postflop");
    // Try multi-SPR first, then legacy
    if let Ok(abstractions) = poker_solver_core::preflop::PostflopBundle::load_multi(
        &poker_solver_core::preflop::PostflopModelConfig::default(),
        &postflop_dir,
    ) {
        return abstractions.into_iter().next().map(|a| a.hand_avg_values);
    }
    // Legacy: co-located solve.bin
    let solve_bin = bp.join("solve.bin");
    if solve_bin.exists() {
        return poker_solver_core::preflop::PostflopBundle::load_hand_avg_values(&solve_bin).ok();
    }
    None
});
```

At line ~344-352 (the other loading site):
```rust
let hand_avg_values = {
    let postflop_dir = bundle_path.join("postflop");
    if let Ok(abstractions) = poker_solver_core::preflop::PostflopBundle::load_multi(
        &poker_solver_core::preflop::PostflopModelConfig::default(),
        &postflop_dir,
    ) {
        abstractions.into_iter().next().map(|a| a.hand_avg_values)
    } else {
        None
    }
};
```

**Step 2: Build and test**

Run: `cargo build -p poker-solver-tauri-app`
Expected: compiles cleanly

**Step 3: Commit**

```
feat: explorer loads multi-SPR postflop bundles (uses first model for display)
```

---

### Task 8: Full integration test

**Files:**
- Modify: `crates/core/tests/postflop_diagnostics.rs` (update existing tests)

**Step 1: Update existing integration tests**

The tests in `postflop_diagnostics.rs` construct `PostflopAbstraction::build(...)` and `solver.attach_postflop(abstraction, ...)`. Update these to use the new Vec-based API:

```rust
// Old:
solver.attach_postflop(abstraction, &config);

// New:
solver.attach_postflop(vec![abstraction], &config);
```

**Step 2: Add a multi-SPR integration test**

```rust
#[test]
#[ignore = "slow: builds 2 SPR models"]
fn multi_spr_solver_uses_both_models() {
    let mut pf_config = PostflopModelConfig {
        postflop_sprs: vec![2.0, 8.0],
        max_flop_boards: 2,
        postflop_solve_iterations: 20,
        ..PostflopModelConfig::exhaustive_fast()
    };

    let abs_low = PostflopAbstraction::build_for_spr(&pf_config, 2.0, None, |_| {}).unwrap();
    let abs_high = PostflopAbstraction::build_for_spr(&pf_config, 8.0, None, |_| {}).unwrap();

    assert!((abs_low.spr - 2.0).abs() < 1e-9);
    assert!((abs_high.spr - 8.0).abs() < 1e-9);

    // Both should have non-empty value tables
    assert!(!abs_low.hand_avg_values.is_empty());
    assert!(!abs_high.hand_avg_values.is_empty());
}
```

**Step 3: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: all PASS

Run: `cargo test -p poker-solver-core multi_spr_solver -- --ignored`
Expected: PASS (slow)

**Step 4: Commit**

```
test: update integration tests for multi-SPR postflop model
```

---

### Task 9: Update documentation

**Files:**
- Modify: `docs/architecture.md`
- Modify: `docs/training.md`

**Step 1: Update architecture.md**

Add a section about multi-SPR model selection in the postflop abstraction description. Key points:
- Multiple SPR trees are solved independently
- At showdown terminals, closest SPR model is selected by absolute distance
- Within-model interpolation still applies when actual SPR < model SPR

**Step 2: Update training.md**

Document the `postflop_sprs` config parameter:
- `postflop_sprs: [2, 6, 20]` trains three models capturing different stack depths
- Legacy `postflop_spr: 3.5` still works (single scalar)
- Bundle format: `postflop/spr_2.0/solve.bin`, etc.
- Training time scales linearly with number of SPRs

**Step 3: Commit**

```
docs: document multi-SPR postflop model configuration and selection
```
