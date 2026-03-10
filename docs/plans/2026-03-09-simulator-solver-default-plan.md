# Simulator Solver Default Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Make bundle selection in the Simulator tab always use `RealTimeSolvingAgentGenerator`, require CBV files, and remove the `solver:` prefix pattern.

**Architecture:** Change `build_agent_generator()` so the bundle fallback arm calls `build_solver_agent_generator()` instead of creating a `BlueprintAgentGenerator`. Remove the `solver:` prefix branch. Change `load_cbv_or_empty` to `load_cbv` that returns an error if the file is missing.

**Tech Stack:** Rust, Tauri backend only. No frontend changes.

---

### Task 1: Make CBV loading fail on missing files

**Files:**
- Modify: `crates/tauri-app/src/simulation.rs:332-343`

**Step 1: Write the failing test**

No test file — this is a small refactor in adapter code. Verify by compiling.

**Step 2: Change `load_cbv_or_empty` to `load_cbv`**

Replace:
```rust
fn load_cbv_or_empty(path: &std::path::Path) -> CbvTable {
    match CbvTable::load(path) {
        Ok(table) => table,
        Err(_) => {
            eprintln!("Warning: No CBV table found at {}, using empty table", path.display());
            CbvTable {
                values: vec![],
                node_offsets: vec![],
                buckets_per_node: vec![],
            }
        }
    }
}
```

With:
```rust
fn load_cbv(path: &std::path::Path) -> Result<CbvTable, String> {
    CbvTable::load(path)
        .map_err(|e| format!("Failed to load CBV table from {}: {e}", path.display()))
}
```

**Step 3: Update call sites in `build_solver_agent_generator`**

Replace:
```rust
let cbv_p0 = load_cbv_or_empty(&dir.join("cbv_p0.bin"));
let _cbv_p1 = load_cbv_or_empty(&dir.join("cbv_p1.bin"));
```

With:
```rust
let cbv_p0 = load_cbv(&dir.join("cbv_p0.bin"))?;
let _cbv_p1 = load_cbv(&dir.join("cbv_p1.bin"))?;
```

**Step 4: Compile**

Run: `cargo build -p poker-solver-tauri`
Expected: Compiles cleanly.

**Step 5: Commit**

```bash
git add crates/tauri-app/src/simulation.rs
git commit -m "fix: require CBV files when loading bundles for simulation"
```

---

### Task 2: Route bundle paths through solver agent generator

**Files:**
- Modify: `crates/tauri-app/src/simulation.rs:351-384`

**Step 1: Remove the `solver:` prefix branch**

Delete lines 362-364:
```rust
if let Some(bundle_path) = path.strip_prefix("solver:") {
    return build_solver_agent_generator(bundle_path);
}
```

**Step 2: Change the bundle fallback to use `build_solver_agent_generator`**

Replace the else arm (lines 372-383):
```rust
    } else {
        let bundle = StrategyBundle::load(&path_buf)
            .map_err(|e| format!("Failed to load bundle: {e}"))?;
        let bet_sizes = bundle.config.game.bet_sizes.clone();
        Ok((
            Box::new(BlueprintAgentGenerator::new(
                Arc::new(bundle.blueprint),
                bundle.config,
            )),
            bet_sizes,
        ))
    }
```

With:
```rust
    } else {
        build_solver_agent_generator(path)
    }
```

**Step 3: Remove unused `BlueprintAgentGenerator` import if no longer referenced**

Check if `BlueprintAgentGenerator` is used anywhere else in this file. If not, remove it from the import at line 20.

**Step 4: Compile and test**

Run: `cargo build -p poker-solver-tauri`
Expected: Compiles cleanly. If `BlueprintAgentGenerator` import is unused, clippy will warn.

Run: `cargo clippy -p poker-solver-tauri`
Expected: No warnings.

**Step 5: Commit**

```bash
git add crates/tauri-app/src/simulation.rs
git commit -m "feat: use RealTimeSolvingAgent for all bundle selections in simulator"
```
