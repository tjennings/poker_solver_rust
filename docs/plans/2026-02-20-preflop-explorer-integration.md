# Preflop Explorer Integration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Unify dataset loading so a single "Load Dataset..." menu item auto-detects preflop vs postflop bundles, and block street progression past preflop for preflop-only solves.

**Architecture:** Add `preflop_only: bool` to `BundleInfo`. Modify `load_bundle` backend command to detect `strategy.bin` (preflop) vs `blueprint.bin` (postflop). Frontend reads the flag to guard street transitions.

**Tech Stack:** Rust/Tauri backend, React/TypeScript frontend.

---

### Task 1: Add `preflop_only` field to `BundleInfo` (backend)

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:103-109` (BundleInfo struct)
- Modify: `crates/tauri-app/src/exploration.rs:233-239` (load_bundle postflop info)
- Modify: `crates/tauri-app/src/exploration.rs:259-265` (load_agent info)
- Modify: `crates/tauri-app/src/exploration.rs:284-290` (load_preflop_solve info)
- Modify: `crates/tauri-app/src/exploration.rs:802-835` (get_bundle_info match arms)

**Step 1: Add field to struct**

In `BundleInfo` struct (~line 103), add `preflop_only`:

```rust
#[derive(Debug, Clone, Serialize)]
pub struct BundleInfo {
    pub name: Option<String>,
    pub stack_depth: u32,
    pub bet_sizes: Vec<f32>,
    pub info_sets: usize,
    pub iterations: u64,
    pub preflop_only: bool,
}
```

**Step 2: Add `preflop_only: false` to all existing BundleInfo constructions**

There are 5 places that construct `BundleInfo`. Add `preflop_only: false` to all except the `PreflopSolve` variant which gets `preflop_only: true`:

- `load_bundle` postflop path (~line 233): `preflop_only: false`
- `load_agent` (~line 259): `preflop_only: false`
- `load_preflop_solve` (~line 284): `preflop_only: true`
- `get_bundle_info` `Bundle` arm (~line 803): `preflop_only: false`
- `get_bundle_info` `Agent` arm (~line 810): `preflop_only: false`
- `get_bundle_info` `PreflopSolve` arm (~line 817): `preflop_only: true`
- `get_bundle_info` `SubgameSolve` arm (~line 824): `preflop_only: false`

**Step 3: Build to verify compilation**

Run: `cargo build -p poker-solver-tauri 2>&1 | head -20`
Expected: Compiles successfully (or warnings only).

**Step 4: Commit**

```bash
git add crates/tauri-app/src/exploration.rs
git commit -m "feat(explorer): add preflop_only field to BundleInfo"
```

---

### Task 2: Auto-detect preflop vs postflop in `load_bundle` (backend)

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs:215-253` (load_bundle function)

**Step 1: Replace the directory branch of `load_bundle`**

Currently the `else` branch (line 224-246) unconditionally calls `StrategyBundle::load`. Replace it with detection logic:

```rust
pub async fn load_bundle(
    state: State<'_, ExplorationState>,
    path: String,
) -> Result<BundleInfo, String> {
    let bundle_path = PathBuf::from(&path);

    let (info, source, boundaries) = if path.ends_with(".toml") {
        let (info, source) = load_agent(&bundle_path)?;
        (info, source, None)
    } else {
        let bp = bundle_path.clone();
        let is_preflop = !bp.join("blueprint.bin").exists() && bp.join("strategy.bin").exists();

        if is_preflop {
            let bundle = tauri::async_runtime::spawn_blocking(move || {
                poker_solver_core::preflop::PreflopBundle::load(&bundle_path)
                    .map_err(|e| format!("Failed to load preflop bundle: {e}"))
            })
            .await
            .map_err(|e| format!("Load task panicked: {e}"))??;

            let info = BundleInfo {
                name: Some("Preflop Solve".into()),
                stack_depth: bundle.config.stacks.first().copied().unwrap_or(0) / 2,
                bet_sizes: vec![],
                info_sets: bundle.strategy.len(),
                iterations: 0,
                preflop_only: true,
            };
            let source = StrategySource::PreflopSolve {
                config: bundle.config,
                strategy: bundle.strategy,
            };
            (info, source, None)
        } else {
            let bundle = tauri::async_runtime::spawn_blocking(move || {
                StrategyBundle::load(&bundle_path)
                    .map_err(|e| format!("Failed to load bundle: {e}"))
            })
            .await
            .map_err(|e| format!("Load task panicked: {e}"))??;

            let info = BundleInfo {
                name: Some("Trained Bundle".to_string()),
                stack_depth: bundle.config.game.stack_depth,
                bet_sizes: bundle.config.game.bet_sizes.clone(),
                info_sets: bundle.blueprint.len(),
                iterations: bundle.blueprint.iterations_trained(),
                preflop_only: false,
            };
            let boundaries = bundle.boundaries;
            let source = StrategySource::Bundle {
                config: bundle.config,
                blueprint: bundle.blueprint,
            };
            (info, source, boundaries)
        }
    };

    *state.abstraction_boundaries.write() = boundaries;
    *state.source.write() = Some(source);
    state.bucket_cache.write().clear();
    *state.suit_mapping.write() = None;

    Ok(info)
}
```

**Step 2: Build to verify**

Run: `cargo build -p poker-solver-tauri 2>&1 | head -20`
Expected: Compiles successfully.

**Step 3: Commit**

```bash
git add crates/tauri-app/src/exploration.rs
git commit -m "feat(explorer): auto-detect preflop vs postflop bundles in load_bundle"
```

---

### Task 3: Add `preflop_only` to frontend types and rename menu

**Files:**
- Modify: `frontend/src/types.ts:1-7` (BundleInfo interface)
- Modify: `frontend/src/Explorer.tsx:15-76` (HamburgerMenu component)
- Modify: `frontend/src/Explorer.tsx:640-652` (handleLoadBundle → handleLoadDataset)
- Modify: `frontend/src/Explorer.tsx:1000-1006` (HamburgerMenu usage)

**Step 1: Add field to TypeScript interface**

In `frontend/src/types.ts`, add `preflop_only`:

```typescript
export interface BundleInfo {
  name: string | null;
  stack_depth: number;
  bet_sizes: number[];
  info_sets: number;
  iterations: number;
  preflop_only: boolean;
}
```

**Step 2: Rename menu item text**

In `Explorer.tsx` HamburgerMenu (~line 64-70), change the button text:

```tsx
<button
  className="menu-item"
  disabled={loading}
  onClick={() => { onLoadBundle(); setOpen(false); }}
>
  Load Dataset...
</button>
```

**Step 3: Rename the callback for clarity**

Rename `handleLoadBundle` → `handleLoadDataset` (~line 640) and update the dialog title:

```tsx
const handleLoadDataset = useCallback(async () => {
  try {
    const path = await open({
      directory: true,
      title: 'Select Dataset Directory',
    });
    if (path) {
      loadSource(path);
    }
  } catch (e) {
    setError(String(e));
  }
}, [loadSource]);
```

Update the HamburgerMenu prop (~line 1005):

```tsx
onLoadBundle={handleLoadDataset}
```

**Step 4: Verify frontend builds**

Run: `cd frontend && npm run build 2>&1 | tail -5`
Expected: Build succeeds.

**Step 5: Commit**

```bash
git add frontend/src/types.ts frontend/src/Explorer.tsx
git commit -m "feat(explorer): rename Load Bundle to Load Dataset, add preflop_only type"
```

---

### Task 4: Block street progression for preflop-only solves (frontend)

**Files:**
- Modify: `frontend/src/Explorer.tsx:655-682` (checkStreetTransition)
- Modify: `frontend/src/Explorer.tsx:734-762` (handleActionSelect transition handling)

**Step 1: Update `checkStreetTransition` to accept `preflopOnly` flag**

The function currently takes `(history, currentStreet)`. Add a third parameter:

```tsx
const checkStreetTransition = useCallback(
  (history: string[], currentStreet: string, preflopOnly: boolean): { needsTransition: boolean; nextStreet: string } => {
    if (history.length < 2) return { needsTransition: false, nextStreet: '' };

    const lastTwo = history.slice(-2);

    const isCallAfterBetOrRaise =
      lastTwo[1] === 'c' && (lastTwo[0].startsWith('r:') || lastTwo[0].startsWith('b:'));

    const isBothCheck = lastTwo[0] === 'x' && lastTwo[1] === 'x';

    const isPreflopLimp = currentStreet === 'Preflop' && lastTwo[0] === 'c' && lastTwo[1] === 'x';

    if (isCallAfterBetOrRaise || isBothCheck || isPreflopLimp) {
      if (currentStreet === 'Preflop') {
        if (preflopOnly) {
          // Preflop-only solve: round ends as showdown, no flop transition
          return { needsTransition: true, nextStreet: '' };
        }
        return { needsTransition: true, nextStreet: 'FLOP' };
      }
      if (currentStreet === 'Flop') return { needsTransition: true, nextStreet: 'TURN' };
      if (currentStreet === 'Turn') return { needsTransition: true, nextStreet: 'RIVER' };
      if (currentStreet === 'River') return { needsTransition: true, nextStreet: '' };
    }

    return { needsTransition: false, nextStreet: '' };
  },
  []
);
```

**Step 2: Update the call site in `handleActionSelect`**

At ~line 735, pass the flag:

```tsx
const { needsTransition, nextStreet } = checkStreetTransition(
  newHistory,
  matrix.street,
  bundleInfo?.preflop_only ?? false
);
```

This means when preflop betting completes (call after raise, or limp), `nextStreet` is `''` which triggers the `needsTransition && !nextStreet` branch at line 755, setting `handResult` to `'showdown'`. The fold path (line 724) already works correctly.

**Step 3: Verify frontend builds**

Run: `cd frontend && npm run build 2>&1 | tail -5`
Expected: Build succeeds.

**Step 4: Manual test**

1. Run the Tauri app: `cargo tauri dev`
2. Click menu → "Load Dataset..." → select `preflop_hu_25bb/`
3. Verify: 13x13 matrix shows preflop actions (fold/call/raise sizes)
4. Click an action → verify navigation works within preflop tree
5. Navigate to a call-after-raise → verify it shows "showdown" instead of prompting for flop cards
6. Click menu → "Load Dataset..." → select a postflop bundle directory
7. Verify: postflop still transitions to flop normally

**Step 5: Commit**

```bash
git add frontend/src/Explorer.tsx
git commit -m "feat(explorer): block street progression past preflop for preflop-only solves"
```
