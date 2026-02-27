# EV Breakdown Panel Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Show EV breakdown (vs range + vs specific hand) in the Explorer when clicking a hand in the grid.

**Architecture:** Embed `hand_avg_values` (169×169 pot-fraction EV table) directly into PreflopBundle as `ev_table.bin`. Extend `get_hand_equity` to accept optional villain hand for per-matchup lookup. Add frontend EvPanel with text input defaulting to "AA".

**Tech Stack:** Rust (bincode, serde), React/TypeScript, existing showdown_equity module

---

### Task 1: Add `hand_avg_values` to PreflopBundle

**Files:**
- Modify: `crates/core/src/preflop/bundle.rs`

**Step 1: Write failing test**

Add to existing test module in `bundle.rs`:

```rust
#[timed_test]
fn preflop_bundle_ev_table_roundtrip() {
    let config = tiny_config();
    let mut solver = PreflopSolver::new(&config);
    solver.train(1);
    let strategy = solver.strategy();
    let ev_data = vec![0.1_f64, 0.2, 0.3];
    let bundle = PreflopBundle::with_ev_table(config, strategy, Some(ev_data.clone()));

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("ev_test");
    bundle.save(&path).unwrap();
    assert!(path.join("ev_table.bin").exists());

    let loaded = PreflopBundle::load(&path).unwrap();
    assert_eq!(loaded.hand_avg_values.as_ref().unwrap(), &ev_data);
}

#[timed_test]
fn preflop_bundle_no_ev_table_backward_compat() {
    let config = tiny_config();
    let mut solver = PreflopSolver::new(&config);
    solver.train(1);
    let bundle = PreflopBundle::new(config, solver.strategy());

    let dir = TempDir::new().unwrap();
    let path = dir.path().join("no_ev_test");
    bundle.save(&path).unwrap();
    assert!(!path.join("ev_table.bin").exists());

    let loaded = PreflopBundle::load(&path).unwrap();
    assert!(loaded.hand_avg_values.is_none());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p poker-solver-core preflop_bundle_ev_table -- --nocapture`
Expected: FAIL — `with_ev_table` method and `hand_avg_values` field don't exist

**Step 3: Implement**

In `bundle.rs`, modify `PreflopBundle`:

```rust
pub struct PreflopBundle {
    pub config: PreflopConfig,
    pub strategy: PreflopStrategy,
    /// Optional hand-averaged EV table from postflop model.
    /// Layout: `[pos0: 169×169, pos1: 169×169]` — pot-fraction units.
    pub hand_avg_values: Option<Vec<f64>>,
}

impl PreflopBundle {
    #[must_use]
    pub fn new(config: PreflopConfig, strategy: PreflopStrategy) -> Self {
        Self { config, strategy, hand_avg_values: None }
    }

    #[must_use]
    pub fn with_ev_table(
        config: PreflopConfig,
        strategy: PreflopStrategy,
        hand_avg_values: Option<Vec<f64>>,
    ) -> Self {
        Self { config, strategy, hand_avg_values }
    }

    pub fn save(&self, dir: &Path) -> Result<(), std::io::Error> {
        fs::create_dir_all(dir)?;

        let config_yaml = serde_yaml::to_string(&self.config).map_err(std::io::Error::other)?;
        fs::write(dir.join("config.yaml"), config_yaml)?;

        let strategy_bytes = bincode::serialize(&self.strategy).map_err(std::io::Error::other)?;
        fs::write(dir.join("strategy.bin"), strategy_bytes)?;

        if let Some(ref ev) = self.hand_avg_values {
            let ev_bytes = bincode::serialize(ev).map_err(std::io::Error::other)?;
            fs::write(dir.join("ev_table.bin"), ev_bytes)?;
        }

        Ok(())
    }

    pub fn load(dir: &Path) -> Result<Self, std::io::Error> {
        let config_yaml = fs::read_to_string(dir.join("config.yaml"))?;
        let config: PreflopConfig =
            serde_yaml::from_str(&config_yaml).map_err(std::io::Error::other)?;

        let strategy_bytes = fs::read(dir.join("strategy.bin"))?;
        let strategy: PreflopStrategy =
            bincode::deserialize(&strategy_bytes).map_err(std::io::Error::other)?;

        let hand_avg_values = {
            let ev_path = dir.join("ev_table.bin");
            if ev_path.exists() {
                let ev_bytes = fs::read(&ev_path)?;
                Some(bincode::deserialize(&ev_bytes).map_err(std::io::Error::other)?)
            } else {
                None
            }
        };

        Ok(Self { config, strategy, hand_avg_values })
    }

    #[must_use]
    pub fn exists(dir: &Path) -> bool {
        dir.join("config.yaml").exists() && dir.join("strategy.bin").exists()
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core preflop_bundle -- --nocapture`
Expected: ALL PASS (including existing `preflop_bundle_roundtrip` and `preflop_bundle_files_on_disk`)

**Step 5: Commit**

```
feat: add optional ev_table.bin to PreflopBundle
```

---

### Task 2: Save hand_avg_values during preflop training

**Files:**
- Modify: `crates/trainer/src/main.rs` — `run_solve_preflop` function (~line 1184-1270)

**Step 1: Extract hand_avg_values before postflop abstraction is consumed**

At line 1184 (before `attach_postflop` consumes the abstraction), clone `hand_avg_values`:

```rust
    let hand_avg_values = postflop.as_ref().map(|abs| abs.hand_avg_values.clone());

    if let Some(ref abstraction) = postflop {
        if !ev_diagnostic_hands.is_empty() {
            print_postflop_ev_diagnostics(abstraction, &ev_diagnostic_hands);
        }
    }
    if let Some(abstraction) = postflop {
        solver.attach_postflop(abstraction, &config);
    }
```

**Step 2: Use `with_ev_table` when saving the bundle**

Replace line 1266:
```rust
    let bundle = PreflopBundle::with_ev_table(config, strategy, hand_avg_values);
```

Also update the checkpoint save at line 1244 to include `None` for ev (checkpoints don't need it):
```rust
    let bundle = PreflopBundle::new(config.clone(), solver.strategy());
```
(This line is unchanged — `new` already sets `hand_avg_values: None`.)

**Step 3: Run build**

Run: `cargo build -p poker-solver-trainer`
Expected: Compiles clean

**Step 4: Commit**

```
feat: save postflop EV table into preflop bundle during training
```

---

### Task 3: Simplify explorer loader — read hand_avg_values from PreflopBundle

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` — `load_bundle_core` (~line 251-288)

**Step 1: Replace the postflop/ directory lookup with direct PreflopBundle field**

Replace lines 258-267:
```rust
            // Try loading companion postflop bundle for hand equity data.
            let postflop_dir = bp.join("postflop");
            let hand_avg_values =
                if poker_solver_core::preflop::PostflopBundle::exists(&postflop_dir) {
                    poker_solver_core::preflop::PostflopBundle::load(&postflop_dir)
                        .ok()
                        .map(|b| b.hand_avg_values().to_vec())
                } else {
                    None
                };
```

With:
```rust
            let hand_avg_values = preflop.hand_avg_values.clone();
```

Note: Keep backward compat — if `hand_avg_values` is None (old bundle) AND `postflop/` exists, fall back to loading from there:

```rust
            let hand_avg_values = preflop.hand_avg_values.clone().or_else(|| {
                let postflop_dir = bp.join("postflop");
                if poker_solver_core::preflop::PostflopBundle::exists(&postflop_dir) {
                    poker_solver_core::preflop::PostflopBundle::load(&postflop_dir)
                        .ok()
                        .map(|b| b.hand_avg_values().to_vec())
                } else {
                    None
                }
            });
```

**Step 2: Run build**

Run: `cargo build -p poker-solver-tauri`
Expected: Compiles clean

**Step 3: Commit**

```
refactor: read hand_avg_values from PreflopBundle, fallback to postflop/ dir
```

---

### Task 4: Extend get_hand_equity with villain_hand parameter

**Files:**
- Modify: `crates/tauri-app/src/exploration.rs` — `HandEquity`, `get_hand_equity_core`, `get_hand_equity`
- Modify: `crates/devserver/src/main.rs` — `HandEquityParams`, `handle_get_hand_equity`

**Step 1: Add `ev_vs_hand` to HandEquity struct**

At line 1057:
```rust
pub struct HandEquity {
    pub ev_pos0: f64,
    pub ev_pos1: f64,
    pub ev_avg: f64,
    /// EV vs a specific opponent hand (if requested).
    pub ev_vs_hand: Option<MatchupEquity>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MatchupEquity {
    pub villain_hand: String,
    pub ev_pos0: f64,
    pub ev_pos1: f64,
    pub ev_avg: f64,
}
```

**Step 2: Extend `get_hand_equity_core` to accept villain_hand**

```rust
pub fn get_hand_equity_core(
    state: &ExplorationState,
    hand: &str,
    villain_hand: Option<&str>,
) -> Result<Option<HandEquity>, String> {
    let hand_index = match canonical_hand_index_from_str(hand) {
        Some(idx) => idx as usize,
        None => return Ok(None),
    };

    let source_guard = state.source.read();
    let source = source_guard
        .as_ref()
        .ok_or_else(|| "No bundle loaded".to_string())?;

    let hand_avg = match source {
        StrategySource::PreflopSolve { hand_avg_values: Some(v), .. } => v,
        _ => return Ok(None),
    };

    let half = hand_avg.len() / 2;
    if half == 0 { return Ok(None); }
    let n = (half as f64).sqrt() as usize;
    if n == 0 || n * n != half || hand_index >= n { return Ok(None); }

    // EV vs all opponents (average across all 169 hands).
    let avg_for_pos = |pos: usize| -> f64 {
        let base = pos * n * n + hand_index * n;
        let slice = &hand_avg[base..base + n];
        let sum: f64 = slice.iter().sum();
        sum / n as f64
    };
    let ev_pos0 = avg_for_pos(0);
    let ev_pos1 = avg_for_pos(1);
    let ev_avg = (ev_pos0 + ev_pos1) / 2.0;

    // EV vs specific opponent hand.
    let ev_vs_hand = villain_hand.and_then(|vh| {
        let v_idx = canonical_hand_index_from_str(vh)? as usize;
        if v_idx >= n { return None; }
        let vp0 = hand_avg[0 * n * n + hand_index * n + v_idx];
        let vp1 = hand_avg[1 * n * n + hand_index * n + v_idx];
        Some(MatchupEquity {
            villain_hand: vh.to_string(),
            ev_pos0: vp0,
            ev_pos1: vp1,
            ev_avg: (vp0 + vp1) / 2.0,
        })
    });

    Ok(Some(HandEquity { ev_pos0, ev_pos1, ev_avg, ev_vs_hand }))
}
```

**Step 3: Update Tauri wrapper**

```rust
#[tauri::command]
pub fn get_hand_equity(
    state: State<'_, ExplorationState>,
    hand: String,
    villain_hand: Option<String>,
) -> Result<Option<HandEquity>, String> {
    get_hand_equity_core(&state, &hand, villain_hand.as_deref())
}
```

**Step 4: Update devserver**

In `crates/devserver/src/main.rs`, update `HandEquityParams`:
```rust
#[derive(Deserialize)]
struct HandEquityParams {
    hand: String,
    villain_hand: Option<String>,
}
```

Update `handle_get_hand_equity`:
```rust
async fn handle_get_hand_equity(
    AxumState(state): AxumState<AppState>,
    Json(params): Json<HandEquityParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::get_hand_equity_core(
        &state,
        &params.hand,
        params.villain_hand.as_deref(),
    ))
}
```

**Step 5: Run build**

Run: `cargo build -p poker-solver-devserver`
Expected: Compiles clean

**Step 6: Commit**

```
feat: extend get_hand_equity with villain_hand matchup lookup
```

---

### Task 5: Add MatchupEquity type and villain_hand to frontend types

**Files:**
- Modify: `frontend/src/types.ts`

**Step 1: Update HandEquity interface**

```typescript
export interface MatchupEquity {
  villain_hand: string;
  ev_pos0: number;
  ev_pos1: number;
  ev_avg: number;
}

export interface HandEquity {
  ev_pos0: number;
  ev_pos1: number;
  ev_avg: number;
  ev_vs_hand: MatchupEquity | null;
}
```

**Step 2: Commit**

```
feat: add MatchupEquity type for villain-specific EV lookup
```

---

### Task 6: Add EV breakdown panel to Explorer

**Files:**
- Modify: `frontend/src/Explorer.tsx`

**Step 1: Add villainHand state and update the handEquity fetch**

Near line 586 (after `handEquity` state), add:
```typescript
const [villainHand, setVillainHand] = useState('AA');
```

Update the `useEffect` that fetches hand equity (lines 635-649) to pass `villain_hand` and re-fetch when `villainHand` changes:

```typescript
useEffect(() => {
    if (!matrix || !selectedCell || !bundleInfo?.preflop_only) {
      setHandEquity(null);
      return;
    }
    const cell = matrix.cells[selectedCell.row]?.[selectedCell.col];
    if (!cell || cell.filtered) {
      setHandEquity(null);
      return;
    }

    invoke<HandEquity | null>('get_hand_equity', {
      hand: cell.hand,
      villain_hand: villainHand || null,
    })
      .then(setHandEquity)
      .catch(() => setHandEquity(null));
  }, [matrix, selectedCell, bundleInfo, villainHand]);
```

**Step 2: Replace the existing handEquity panel (lines 1190-1208) with EvPanel**

Replace the existing `handEquity &&` block with a new component that includes the villain input:

```tsx
{handEquity && (
  <div className="hand-equity-panel">
    <div className="cell-detail-header">Postflop EV (% of pot)</div>
    <div className="hand-equity-rows">
      <div className="hand-equity-subheader">vs Range</div>
      <div className="hand-equity-row">
        <span className="hand-equity-label">As SB</span>
        <span className="hand-equity-value">{formatEV(handEquity.ev_pos0)}</span>
      </div>
      <div className="hand-equity-row">
        <span className="hand-equity-label">As BB</span>
        <span className="hand-equity-value">{formatEV(handEquity.ev_pos1)}</span>
      </div>
      <div className="hand-equity-row hand-equity-avg">
        <span className="hand-equity-label">Average</span>
        <span className="hand-equity-value">{formatEV(handEquity.ev_avg)}</span>
      </div>
    </div>
    <div className="hand-equity-vs-hand">
      <div className="hand-equity-subheader">
        vs
        <input
          type="text"
          className="villain-hand-input"
          value={villainHand}
          onChange={(e) => setVillainHand(e.target.value.toUpperCase())}
          placeholder="AA"
          maxLength={3}
        />
      </div>
      {handEquity.ev_vs_hand ? (
        <div className="hand-equity-rows">
          <div className="hand-equity-row">
            <span className="hand-equity-label">As SB</span>
            <span className="hand-equity-value">{formatEV(handEquity.ev_vs_hand.ev_pos0)}</span>
          </div>
          <div className="hand-equity-row">
            <span className="hand-equity-label">As BB</span>
            <span className="hand-equity-value">{formatEV(handEquity.ev_vs_hand.ev_pos1)}</span>
          </div>
          <div className="hand-equity-row hand-equity-avg">
            <span className="hand-equity-label">Average</span>
            <span className="hand-equity-value">{formatEV(handEquity.ev_vs_hand.ev_avg)}</span>
          </div>
        </div>
      ) : (
        <div className="hand-equity-row">
          <span className="hand-equity-label" style={{ color: '#666' }}>Invalid hand</span>
        </div>
      )}
    </div>
  </div>
)}
```

Also update the `import` at the top to include `MatchupEquity` (though it's not directly used — just `HandEquity` which now contains it).

**Step 3: Commit**

```
feat: add EV vs specific hand input to Explorer equity panel
```

---

### Task 7: Add CSS for the villain hand input and subheaders

**Files:**
- Modify: `frontend/src/App.css`

**Step 1: Add styles after the existing `.hand-equity-value` block (~line 916)**

```css
.hand-equity-subheader {
  font-size: 0.75rem;
  color: #888;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 0.5rem;
  margin-bottom: 0.15rem;
  display: flex;
  align-items: center;
  gap: 0.4rem;
}

.hand-equity-subheader:first-child {
  margin-top: 0;
}

.hand-equity-vs-hand {
  margin-top: 0.25rem;
}

.villain-hand-input {
  width: 3.5em;
  padding: 0.15rem 0.3rem;
  background: #0d1117;
  border: 1px solid #333;
  border-radius: 4px;
  color: #eee;
  font-size: 0.8rem;
  font-family: inherit;
  text-align: center;
}

.villain-hand-input:focus {
  outline: none;
  border-color: #00d9ff;
}
```

**Step 2: Commit**

```
feat: style villain hand input and EV subheaders
```

---

### Task 8: Backfill ev_table.bin for existing full_postflop bundle

This is a one-time data migration. The `full_postflop` bundle has `solve.bin` (PostflopBundle data) at the root level. We need to extract `hand_avg_values` from it and save as `ev_table.bin`.

**Files:**
- No code changes — CLI one-liner using existing tools

**Step 1: Add a trainer subcommand or use a script**

Actually, the simplest approach: update the explorer loader fallback (Task 3) to ALSO check for `solve.bin` at the root level (not just `postflop/` dir). When found, load the PostflopBundle from the root to get hand_avg_values.

Modify the fallback in `load_bundle_core` (from Task 3):

```rust
let hand_avg_values = preflop.hand_avg_values.clone().or_else(|| {
    // Fallback 1: postflop/ subdirectory (older training layout)
    let postflop_dir = bp.join("postflop");
    if poker_solver_core::preflop::PostflopBundle::exists(&postflop_dir) {
        return poker_solver_core::preflop::PostflopBundle::load(&postflop_dir)
            .ok()
            .map(|b| b.hand_avg_values().to_vec());
    }
    // Fallback 2: solve.bin at root (solve-postflop output co-located with preflop)
    let solve_bin = bp.join("solve.bin");
    if solve_bin.exists() {
        // The PostflopBundle config.yaml was overwritten by PreflopBundle.
        // Read solve.bin directly and deserialize the PostflopBundleData.
        // We only need hand_avg_values, but bincode needs the full struct.
        // Use PostflopBundle::load won't work (wrong config.yaml), so
        // deserialize the data portion only.
        if let Ok(data_bytes) = std::fs::read(&solve_bin) {
            #[derive(serde::Deserialize)]
            struct PostflopData {
                _values: poker_solver_core::preflop::postflop_abstraction::PostflopValues,
                hand_avg_values: Vec<f64>,
                _flops: Vec<poker_solver_core::poker::Card>,
                _spr: f64,
            }
            if let Ok(data) = bincode::deserialize::<PostflopData>(&data_bytes) {
                return Some(data.hand_avg_values);
            }
        }
    }
    None
});
```

Wait — `PostflopBundleData` is private and `PostflopValues` has a non-trivial layout. Deserializing a private struct is fragile. Better approach: make `PostflopBundle` able to load from data bytes alone (without config.yaml), OR add a public method to extract hand_avg_values from raw solve.bin bytes.

**Simpler approach**: Add a `PostflopBundle::load_hand_avg_values(solve_bin_path)` method:

In `crates/core/src/preflop/postflop_bundle.rs`:
```rust
/// Load just the hand-averaged EV table from a solve.bin file.
///
/// This is useful when the companion config.yaml has been overwritten
/// (e.g., by a co-located PreflopBundle).
pub fn load_hand_avg_values(solve_bin: &Path) -> Result<Vec<f64>, std::io::Error> {
    let data_bytes = fs::read(solve_bin)?;
    let data: PostflopBundleData =
        bincode::deserialize(&data_bytes).map_err(std::io::Error::other)?;
    Ok(data.hand_avg_values)
}
```

Then the fallback in `load_bundle_core` becomes:
```rust
// Fallback 2: solve.bin at root (solve-postflop output co-located with preflop)
let solve_bin = bp.join("solve.bin");
if solve_bin.exists() {
    return poker_solver_core::preflop::PostflopBundle::load_hand_avg_values(&solve_bin).ok();
}
```

**Step 2: Run build and test manually with full_postflop bundle**

Run: `cargo build -p poker-solver-devserver --release && cargo run -p poker-solver-devserver --release`

Then: `curl -s -X POST http://localhost:3001/api/load_bundle -H 'Content-Type: application/json' -d '{"path":"./local_data/full_postflop"}'`

Expected: loads successfully (after fixing config compat in Task 9)

**Step 3: Commit**

```
feat: add PostflopBundle::load_hand_avg_values for co-located solve.bin fallback
```

---

### Task 9: Fix RaiseSize backward compatibility

The old bundles have plain float raise sizes (e.g. `0.3`, `1.0`) in config.yaml. The new `RaiseSize` deserializer requires "bb" or "p" suffix. Add backward compat: plain floats deserialize as `PotFraction`.

**Files:**
- Modify: `crates/core/src/preflop/config.rs` — `RaiseSize::Deserialize` impl (~line 55)

**Step 1: Write failing test**

```rust
#[test]
fn raise_size_deserialize_plain_float_as_pot_fraction() {
    let parsed: RaiseSize = serde_yaml::from_str("0.75").unwrap();
    assert_eq!(parsed, RaiseSize::PotFraction(0.75));
}

#[test]
fn raise_size_deserialize_plain_integer_as_pot_fraction() {
    let parsed: RaiseSize = serde_yaml::from_str("2.0").unwrap();
    assert_eq!(parsed, RaiseSize::PotFraction(2.0));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core raise_size_deserialize_plain -- --nocapture`
Expected: FAIL — "invalid raise size '0.75'"

**Step 3: Update deserializer**

In the `Deserialize` impl, add fallback before the error:

```rust
impl<'de> Deserialize<'de> for RaiseSize {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        if let Some(num) = s.strip_suffix("bb") {
            let val: f64 = num.parse().map_err(serde::de::Error::custom)?;
            Ok(Self::Bb(val))
        } else if let Some(num) = s.strip_suffix('p') {
            let val: f64 = num.parse().map_err(serde::de::Error::custom)?;
            Ok(Self::PotFraction(val))
        } else if let Ok(val) = s.parse::<f64>() {
            // Backward compat: plain float → pot fraction
            Ok(Self::PotFraction(val))
        } else {
            Err(serde::de::Error::custom(format!(
                "invalid raise size '{s}': must end with 'bb' (e.g. \"2.5bb\") or 'p' (e.g. \"0.75p\")"
            )))
        }
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core raise_size -- --nocapture`
Expected: ALL PASS

**Step 5: Commit**

```
fix: accept plain float raise sizes as pot fractions for backward compat
```

---

### Task 10: End-to-end test with full_postflop bundle

**No code changes — manual verification.**

**Step 1: Rebuild devserver**

Run: `cargo build -p poker-solver-devserver --release`

**Step 2: Start devserver and load bundle**

```bash
cargo run -p poker-solver-devserver --release &
curl -s -X POST http://localhost:3001/api/load_bundle \
  -H 'Content-Type: application/json' \
  -d '{"path":"./local_data/full_postflop"}'
```

Expected: `{"name":"Preflop Solve","stack_depth":100,"preflop_only":true,...}`

**Step 3: Test EV endpoint**

```bash
# EV vs range only
curl -s -X POST http://localhost:3001/api/get_hand_equity \
  -H 'Content-Type: application/json' \
  -d '{"hand":"AKs"}'

# EV vs range + vs AA
curl -s -X POST http://localhost:3001/api/get_hand_equity \
  -H 'Content-Type: application/json' \
  -d '{"hand":"AKs","villain_hand":"AA"}'
```

Expected: Both return non-null with ev_pos0/ev_pos1/ev_avg values. Second includes `ev_vs_hand` object.

**Step 4: Test frontend**

Open `http://localhost:5173`, load `./local_data/full_postflop`. Click a hand. Verify:
- EV vs Range section shows with SB/BB/Avg values
- Input field shows "AA" by default
- EV vs AA section shows values
- Typing a different hand (e.g. "72o") updates the matchup EV

**Step 5: Commit**

```
test: verify EV breakdown panel with full_postflop bundle
```
