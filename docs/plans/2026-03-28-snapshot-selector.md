# Snapshot Selector Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a two-step blueprint loading flow: pick blueprint → select snapshot (defaulting to latest) → load.

**Architecture:** New `list_snapshots` command scans a blueprint directory for `snapshot_*` dirs and reads `metadata.json` from each. `load_blueprint_v2` gains an optional `snapshot` param to load a specific snapshot. Frontend shows a snapshot dropdown after blueprint selection.

**Tech Stack:** Rust (Tauri commands, serde), TypeScript/React (Explorer.tsx), Axum (devserver)

---

### Task 1: Backend — `SnapshotEntry` type and `list_snapshots` command

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`

**Step 1: Add the `SnapshotEntry` struct near `BlueprintListEntry` (after line 57)**

```rust
/// A single snapshot within a blueprint directory.
#[derive(Debug, Clone, serde::Serialize)]
pub struct SnapshotEntry {
    /// Directory name, e.g. "snapshot_0005".
    pub name: String,
    /// Iteration count from metadata.json, if available.
    pub iterations: Option<u64>,
    /// Elapsed training minutes from metadata.json, if available.
    pub elapsed_minutes: Option<u64>,
    /// Whether this snapshot contains a strategy.bin file.
    pub has_strategy: bool,
}
```

**Step 2: Add `list_snapshots_core` function (after `list_blueprints_core`, ~line 706)**

```rust
/// List all snapshots in a blueprint directory, sorted newest-first.
pub fn list_snapshots_core(blueprint_path: String) -> Result<Vec<SnapshotEntry>, String> {
    let dir = PathBuf::from(&blueprint_path);
    let entries = std::fs::read_dir(&dir)
        .map_err(|e| format!("Failed to read directory {blueprint_path}: {e}"))?;

    let mut snapshots: Vec<SnapshotEntry> = entries
        .filter_map(Result::ok)
        .filter_map(|e| {
            let name = e.file_name().to_str()?.to_string();
            if !name.starts_with("snapshot_") {
                return None;
            }
            let snap_dir = e.path();
            let has_strategy = snap_dir.join("strategy.bin").exists();

            // Try to read metadata.json for iteration count and elapsed time.
            let (iterations, elapsed_minutes) = snap_dir
                .join("metadata.json")
                .exists()
                .then(|| {
                    let data = std::fs::read_to_string(snap_dir.join("metadata.json")).ok()?;
                    let json: serde_json::Value = serde_json::from_str(&data).ok()?;
                    let iters = json.get("iteration").and_then(|v| v.as_u64());
                    let mins = json.get("elapsed_minutes").and_then(|v| v.as_u64());
                    Some((iters, mins))
                })
                .flatten()
                .unwrap_or((None, None));

            Some(SnapshotEntry {
                name,
                iterations,
                elapsed_minutes,
                has_strategy,
            })
        })
        .collect();

    // Sort newest-first (highest snapshot number first).
    snapshots.sort_by(|a, b| b.name.cmp(&a.name));
    Ok(snapshots)
}
```

**Step 3: Add the Tauri command wrapper (after `list_blueprints` command)**

```rust
#[tauri::command]
pub async fn list_snapshots(path: String) -> Result<Vec<SnapshotEntry>, String> {
    list_snapshots_core(path)
}
```

**Step 4: Verify it compiles**

Run: `cargo build -p poker-solver-tauri`

**Step 5: Commit**

```bash
git add crates/tauri-app/src/exploration.rs
git commit -m "feat: add list_snapshots command to enumerate blueprint snapshots"
```

---

### Task 2: Backend — Add `snapshot` param to `load_blueprint_v2`

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs`

**Step 1: Add `snapshot` parameter to `load_blueprint_v2` and `load_blueprint_v2_core`**

Change the Tauri command signature at line 506:

```rust
#[tauri::command]
pub async fn load_blueprint_v2(
    state: State<'_, ExplorationState>,
    postflop_state: State<'_, Arc<crate::postflop::PostflopState>>,
    path: String,
    snapshot: Option<String>,
) -> Result<BundleInfo, String> {
    let info = load_blueprint_v2_core(&state, path, snapshot).await?;
    populate_cbv_context(&state, &postflop_state);
    Ok(info)
}
```

Change `load_blueprint_v2_core` signature at line 338:

```rust
pub async fn load_blueprint_v2_core(
    state: &ExplorationState,
    dir_path: String,
    snapshot: Option<String>,
) -> Result<BundleInfo, String> {
```

**Step 2: Update the strategy path logic inside `load_blueprint_v2_core` (~line 347-372)**

Replace the existing snapshot auto-detection with:

```rust
        // Determine strategy path. If a specific snapshot is requested, use it.
        // Otherwise: final/ → root → latest snapshot.
        let strat_path = if let Some(ref snap_name) = snapshot {
            let snap_dir = dir.join(snap_name);
            if !snap_dir.join("strategy.bin").exists() {
                return Err(format!("No strategy.bin in {snap_name}"));
            }
            snap_dir.join("strategy.bin")
        } else if dir.join("final/strategy.bin").exists() {
            dir.join("final/strategy.bin")
        } else if dir.join("strategy.bin").exists() {
            dir.join("strategy.bin")
        } else {
            // Auto-detect latest snapshot.
            let mut snapshots: Vec<_> = std::fs::read_dir(&dir)
                .map_err(|e| format!("Cannot read directory: {e}"))?
                .filter_map(Result::ok)
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .is_some_and(|n| n.starts_with("snapshot_"))
                })
                .collect();
            snapshots.sort_by_key(|e| e.file_name());
            match snapshots.last() {
                Some(entry) => entry.path().join("strategy.bin"),
                None => {
                    return Err(
                        "No strategy.bin found in bundle directory".to_string()
                    )
                }
            }
        };
```

**Step 3: Fix ALL callers of `load_blueprint_v2_core`**

Search for callers:

```bash
grep -rn "load_blueprint_v2_core" crates/tauri-app/src/ crates/devserver/src/ --include="*.rs"
```

Each existing call site passes `None` for the new `snapshot` param. For example in `exploration.rs` where `load_bundle` calls it:

```rust
// In load_bundle or wherever load_blueprint_v2_core is called without snapshot:
load_blueprint_v2_core(state, path, None).await
```

Also update the devserver handler at `devserver/main.rs:193`:

```rust
let info = poker_solver_tauri::load_blueprint_v2_core(&state, params.path, params.snapshot).await;
```

And update `PathParams` in devserver to include snapshot:

```rust
#[derive(Deserialize)]
struct LoadBlueprintParams {
    path: String,
    snapshot: Option<String>,
}
```

Use `LoadBlueprintParams` instead of `PathParams` for the `handle_load_blueprint_v2` handler.

**Step 4: Verify it compiles**

Run: `cargo build -p poker-solver-tauri -p poker-solver-devserver`

**Step 5: Commit**

```bash
git add crates/tauri-app/src/exploration.rs crates/devserver/src/main.rs
git commit -m "feat: add optional snapshot param to load_blueprint_v2"
```

---

### Task 3: Backend — Register `list_snapshots` in Tauri and devserver

**Files:**
- Modify: `crates/tauri-app/src/main.rs`
- Modify: `crates/devserver/src/main.rs`

**Step 1: Register Tauri command in `main.rs`**

Add `poker_solver_tauri::list_snapshots,` to the `generate_handler![]` macro (near line 66 where `list_blueprints` is registered).

**Step 2: Add devserver handler and route**

In `devserver/main.rs`, add a params struct:

```rust
#[derive(Deserialize)]
struct ListSnapshotsParams {
    path: String,
}
```

Add handler:

```rust
async fn handle_list_snapshots(
    Json(params): Json<ListSnapshotsParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::list_snapshots_core(params.path))
}
```

Add route (near the `list_blueprints` route, ~line 688):

```rust
.route("/api/list_snapshots", post(handle_list_snapshots))
```

**Step 3: Verify both compile**

Run: `cargo build -p poker-solver-tauri -p poker-solver-devserver`

**Step 4: Commit**

```bash
git add crates/tauri-app/src/main.rs crates/devserver/src/main.rs
git commit -m "feat: register list_snapshots in Tauri and devserver"
```

---

### Task 4: Frontend — Add `SnapshotEntry` type

**Files:**
- Modify: `frontend/src/types.ts`

**Step 1: Add the type after `BlueprintListEntry` (~line 167)**

```typescript
export interface SnapshotEntry {
  name: string;
  iterations: number | null;
  elapsed_minutes: number | null;
  has_strategy: boolean;
}
```

**Step 2: Commit**

```bash
git add frontend/src/types.ts
git commit -m "feat: add SnapshotEntry type"
```

---

### Task 5: Frontend — Snapshot picker in Explorer.tsx

**Files:**
- Modify: `frontend/src/Explorer.tsx`

**Step 1: Add imports and state**

Add `SnapshotEntry` to the imports from `./types` at line 15:

```typescript
import {
  BlueprintListEntry,
  SnapshotEntry,
  // ... existing imports
} from './types';
```

Add state variables near the blueprint picker state (~line 591-592):

```typescript
const [selectedBlueprint, setSelectedBlueprint] = useState<BlueprintListEntry | null>(null);
const [snapshots, setSnapshots] = useState<SnapshotEntry[]>([]);
const [selectedSnapshot, setSelectedSnapshot] = useState<string | null>(null);
const [showSnapshotPicker, setShowSnapshotPicker] = useState(false);
```

**Step 2: Change the blueprint click handler**

In the blueprint list `onClick` handler (~line 1249-1282), instead of immediately calling `load_blueprint_v2`, fetch snapshots first:

```typescript
onClick={async () => {
  try {
    const snaps = await invoke<SnapshotEntry[]>('list_snapshots', { path: bp.path });
    const strategySnaps = snaps.filter(s => s.has_strategy);
    if (strategySnaps.length <= 1) {
      // 0 or 1 snapshot — load directly (no picker needed)
      setShowBlueprintPicker(false);
      setLoading(true);
      setError(null);
      try {
        const info = await invoke<BundleInfo>('load_blueprint_v2', {
          path: bp.path,
          snapshot: strategySnaps.length === 1 ? strategySnaps[0].name : null,
        });
        // ... existing post-load logic (setBundleInfo, setPosition, etc.)
      } catch (e) {
        setError(String(e));
      } finally {
        setLoading(false);
      }
    } else {
      // Multiple snapshots — show snapshot picker
      setSelectedBlueprint(bp);
      setSnapshots(strategySnaps);
      setSelectedSnapshot(strategySnaps[0].name); // default to latest (sorted newest-first)
      setShowSnapshotPicker(true);
    }
  } catch (e) {
    setError(String(e));
  }
}}
```

**Step 3: Add the snapshot picker modal**

After the blueprint picker modal closing `</div>` (~line 1292), add:

```tsx
{showSnapshotPicker && selectedBlueprint && (
  <div className="dataset-picker-overlay" onClick={() => setShowSnapshotPicker(false)}>
    <div className="dataset-picker" onClick={e => e.stopPropagation()}>
      <div className="dataset-picker-header">
        <h3>{selectedBlueprint.name} — Select Snapshot</h3>
        <button className="dataset-picker-close" onClick={() => setShowSnapshotPicker(false)}>×</button>
      </div>
      <div className="dataset-picker-list">
        {snapshots.map((snap) => (
          <div
            key={snap.name}
            className={`dataset-picker-item ${selectedSnapshot === snap.name ? 'selected' : ''}`}
            onClick={() => setSelectedSnapshot(snap.name)}
            onDoubleClick={async () => {
              setShowSnapshotPicker(false);
              setShowBlueprintPicker(false);
              setLoading(true);
              setError(null);
              try {
                const info = await invoke<BundleInfo>('load_blueprint_v2', {
                  path: selectedBlueprint.path,
                  snapshot: snap.name,
                });
                setBundleInfo(info);
                const sp1 = info.stack_depth - 1;
                const sp2 = info.stack_depth - 2;
                const initialPosition: ExplorationPosition = {
                  board: [],
                  history: [],
                  pot: 3,
                  stacks: [sp1, sp2],
                  to_act: 0,
                  num_players: 2,
                  active_players: [true, true],
                };
                setPosition(initialPosition);
                setHistoryItems([]);
                setPendingStreet(null);
                setHandResult(null);
                setSelectedCell(null);
                const initialMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
                  position: initialPosition,
                });
                setMatrix(initialMatrix);
                updateRangesFromMatrix(initialMatrix);
              } catch (e) {
                setError(String(e));
              } finally {
                setLoading(false);
              }
            }}
          >
            <span className="dataset-kind-badge preflop">
              {snap.name.replace('snapshot_', '#')}
            </span>
            <span className="dataset-name">
              {snap.iterations != null
                ? `${(snap.iterations / 1_000_000).toFixed(1)}M iterations`
                : 'unknown iterations'}
              {snap.elapsed_minutes != null
                ? ` · ${Math.round(snap.elapsed_minutes)}min`
                : ''}
            </span>
          </div>
        ))}
      </div>
      <div style={{ padding: '8px 12px', display: 'flex', justifyContent: 'flex-end', gap: '8px' }}>
        <button
          className="action-button"
          onClick={() => setShowSnapshotPicker(false)}
        >
          Cancel
        </button>
        <button
          className="action-button"
          onClick={async () => {
            if (!selectedSnapshot) return;
            setShowSnapshotPicker(false);
            setShowBlueprintPicker(false);
            setLoading(true);
            setError(null);
            try {
              const info = await invoke<BundleInfo>('load_blueprint_v2', {
                path: selectedBlueprint.path,
                snapshot: selectedSnapshot,
              });
              setBundleInfo(info);
              const sp1 = info.stack_depth - 1;
              const sp2 = info.stack_depth - 2;
              const initialPosition: ExplorationPosition = {
                board: [],
                history: [],
                pot: 3,
                stacks: [sp1, sp2],
                to_act: 0,
                num_players: 2,
                active_players: [true, true],
              };
              setPosition(initialPosition);
              setHistoryItems([]);
              setPendingStreet(null);
              setHandResult(null);
              setSelectedCell(null);
              const initialMatrix = await invoke<StrategyMatrix>('get_strategy_matrix', {
                position: initialPosition,
              });
              setMatrix(initialMatrix);
              updateRangesFromMatrix(initialMatrix);
            } catch (e) {
              setError(String(e));
            } finally {
              setLoading(false);
            }
          }}
        >
          Load
        </button>
      </div>
    </div>
  </div>
)}
```

Note: The post-load logic (setBundleInfo → setPosition → getStrategyMatrix) is duplicated from the original blueprint click handler. Consider extracting it into a helper function `loadBlueprint(path, snapshot)` to DRY it up. This is recommended but not strictly required.

**Step 4: Verify it builds**

Run: `cd frontend && npm run build`

**Step 5: Commit**

```bash
git add frontend/src/Explorer.tsx
git commit -m "feat: snapshot picker UI — two-step blueprint loading flow"
```

---

### Task 6: Smoke test and final verification

**Step 1: Build full workspace**

Run: `cargo build`

**Step 2: Run all Rust tests**

Run: `cargo test -p poker-solver-core -p poker-solver-trainer`

**Step 3: Manual smoke test**

Start the dev server and frontend:

```bash
cargo run -p poker-solver-devserver &
cd frontend && npm run dev
```

1. Open browser, go to Settings, set blueprint directory
2. Click "Load Strategy"
3. Pick a blueprint with multiple snapshots
4. Verify: snapshot list appears, newest first, with iteration counts
5. Select an older snapshot, click Load
6. Verify: strategy loads from the correct snapshot (check BundleInfo.snapshot_name in the response)

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: snapshot selector polish"
```
