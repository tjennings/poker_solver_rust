# Remote Backend Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Allow the Tauri desktop app to optionally route solver commands to a remote devserver backend over HTTP, enabling GPU-accelerated solving on a separate machine.

**Architecture:** Evolve the existing devserver into a production remote backend. The frontend's `invoke.ts` gains a configurable URL — when set, HTTP replaces Tauri IPC. Events stream over WebSocket instead of Tauri's event system. The Tauri shell always runs locally for window management.

**Tech Stack:** Rust (Axum, tokio, tokio-tungstenite), TypeScript (React), WebSocket, HTTP

**Design doc:** `docs/plans/2026-03-16-remote-backend-design.md`

---

### Task 1: Extract `_core` Variants for Simulation Commands

The simulation commands (`list_strategy_sources`, `start_simulation`, `stop_simulation`, `get_simulation_result`) currently use Tauri's `AppHandle` and `State<'_>` directly. They need `_core` variants (like exploration commands already have) so the devserver can call them.

**Files:**
- Modify: `crates/tauri-app/src/simulation.rs`
- Modify: `crates/tauri-app/src/lib.rs`

**Step 1: Write the failing test**

No unit test needed — the existing Tauri commands are the test. Instead, verify the project compiles after refactoring.

**Step 2: Extract `list_strategy_sources_core`**

In `crates/tauri-app/src/simulation.rs`, the existing `list_strategy_sources` function doesn't use `AppHandle` or `State` — it's already a pure function. Just rename it to `list_strategy_sources_core` and add a thin Tauri wrapper:

```rust
/// Core: list all strategy sources (no Tauri dependency).
pub fn list_strategy_sources_core(dir: Option<String>) -> Result<Vec<StrategySourceInfo>, String> {
    // ... existing body unchanged ...
}

#[tauri::command]
pub fn list_strategy_sources(dir: Option<String>) -> Result<Vec<StrategySourceInfo>, String> {
    list_strategy_sources_core(dir)
}
```

**Step 3: Extract `stop_simulation_core` and `get_simulation_result_core`**

These two are simple — they only touch `SimulationState`:

```rust
pub fn stop_simulation_core(state: &SimulationState) {
    state.running.store(false, Ordering::SeqCst);
}

#[tauri::command]
pub async fn stop_simulation(state: State<'_, SimulationState>) -> Result<(), String> {
    stop_simulation_core(&state);
    Ok(())
}

pub fn get_simulation_result_core(state: &SimulationState) -> Option<SimResultResponse> {
    state.result.read().as_ref().map(|r| SimResultResponse {
        hands_played: r.hands_played,
        p1_profit_bb: r.p1_profit_bb,
        mbbh: r.mbbh,
        equity_curve: r.equity_curve.clone(),
        elapsed_ms: r.elapsed_ms,
    })
}

#[tauri::command]
pub fn get_simulation_result(state: State<'_, SimulationState>) -> Result<Option<SimResultResponse>, String> {
    Ok(get_simulation_result_core(&state))
}
```

**Step 4: Extract `start_simulation_core` with an event sink**

This is the most involved. `start_simulation` currently uses `app.emit(...)` to send events. Replace with a trait-based event sink so both Tauri and broadcast channel can emit:

```rust
use std::fmt::Debug;

/// Trait for emitting simulation events (Tauri AppHandle or broadcast channel).
pub trait SimEventSink: Send + 'static {
    fn emit_progress(&self, event: SimProgressEvent);
    fn emit_complete(&self, event: SimResultResponse);
    fn emit_error(&self, error: String);
}

/// Tauri implementation.
impl SimEventSink for AppHandle {
    fn emit_progress(&self, event: SimProgressEvent) {
        let _ = self.emit("simulation-progress", event);
    }
    fn emit_complete(&self, event: SimResultResponse) {
        let _ = self.emit("simulation-complete", event);
    }
    fn emit_error(&self, error: String) {
        let _ = self.emit("simulation-error", error);
    }
}

pub fn start_simulation_core(
    sink: impl SimEventSink,
    state: &SimulationState,
    p1_path: String,
    p2_path: String,
    num_hands: u64,
    stack_depth: u32,
) -> Result<(), String> {
    // Stop any running simulation
    state.running.store(false, Ordering::SeqCst);
    std::thread::sleep(std::time::Duration::from_millis(50));

    let running = Arc::clone(&state.running);
    let result_store = Arc::clone(&state.result);

    running.store(true, Ordering::SeqCst);
    *result_store.write() = None;

    std::thread::spawn(move || {
        // ... existing body, replacing app.emit("simulation-progress", ...) with sink.emit_progress(...)
        // ... app.emit("simulation-complete", ...) with sink.emit_complete(...)
        // ... app.emit("simulation-error", ...) with sink.emit_error(...)
    });

    Ok(())
}

#[tauri::command]
pub async fn start_simulation(
    app: AppHandle,
    state: State<'_, SimulationState>,
    p1_path: String,
    p2_path: String,
    num_hands: u64,
    stack_depth: u32,
) -> Result<(), String> {
    start_simulation_core(app, &state, p1_path, p2_path, num_hands, stack_depth)
}
```

**Step 5: Update `lib.rs` exports**

Add the new `_core` functions and `SimEventSink` to the public exports in `crates/tauri-app/src/lib.rs`:

```rust
pub use simulation::{
    get_simulation_result, list_strategy_sources, start_simulation, stop_simulation,
    // Core functions
    get_simulation_result_core, list_strategy_sources_core, start_simulation_core,
    stop_simulation_core,
    // Types
    SimEventSink, SimProgressEvent, SimResultResponse, SimulationState, StrategySourceInfo,
};
```

**Step 6: Verify compilation**

Run: `cargo build -p poker-solver-tauri`
Expected: compiles with no errors

**Step 7: Run tests**

Run: `cargo test`
Expected: all tests pass

**Step 8: Commit**

```bash
git add crates/tauri-app/src/simulation.rs crates/tauri-app/src/lib.rs
git commit -m "refactor: extract _core variants for simulation commands

Separate Tauri-specific wrappers from core logic so the devserver
can call simulation functions without Tauri dependencies.
Introduce SimEventSink trait for event emission abstraction."
```

---

### Task 2: Add Health Endpoint and Simulation Routes to Devserver

**Files:**
- Modify: `crates/devserver/src/main.rs`

**Step 1: Add request param structs for simulation commands**

```rust
#[derive(Deserialize)]
struct StartSimulationParams {
    p1_path: String,
    p2_path: String,
    num_hands: u64,
    stack_depth: u32,
}

#[derive(Deserialize)]
struct ListStrategySourcesParams {
    dir: Option<String>,
}
```

**Step 2: Add health endpoint handler**

```rust
async fn handle_health() -> Json<serde_json::Value> {
    to_json_value(serde_json::json!({ "ok": true }))
}
```

**Step 3: Add simulation command handlers**

```rust
async fn handle_list_strategy_sources(
    Json(params): Json<ListStrategySourcesParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::list_strategy_sources_core(params.dir))
}

async fn handle_stop_simulation(
    Extension(state): Extension<Arc<SimulationState>>,
) -> Json<serde_json::Value> {
    poker_solver_tauri::stop_simulation_core(&state);
    to_json_value("ok")
}

async fn handle_get_simulation_result(
    Extension(state): Extension<Arc<SimulationState>>,
) -> Json<serde_json::Value> {
    to_json_value(poker_solver_tauri::get_simulation_result_core(&state))
}
```

Note: `start_simulation` handler will be added in Task 3 (needs the broadcast event sink).

**Step 4: Add SimulationState and routes in main()**

```rust
let sim_state: Arc<SimulationState> = Arc::new(SimulationState::default());

// Add to router:
.route("/health", get(handle_health))
.route("/api/list_strategy_sources", post(handle_list_strategy_sources))
.route("/api/stop_simulation", post(handle_stop_simulation))
.route("/api/get_simulation_result", post(handle_get_simulation_result))
.layer(Extension(sim_state))
```

Add `use axum::routing::get;` to imports, and `use poker_solver_tauri::SimulationState;` to imports.

**Step 5: Verify compilation**

Run: `cargo build -p poker-solver-devserver`
Expected: compiles

**Step 6: Commit**

```bash
git add crates/devserver/src/main.rs
git commit -m "feat: add health endpoint and simulation routes to devserver

Adds GET /health for connection testing and POST endpoints for
list_strategy_sources, stop_simulation, get_simulation_result."
```

---

### Task 3: Add WebSocket Event Streaming to Devserver

**Files:**
- Modify: `crates/devserver/Cargo.toml`
- Modify: `crates/devserver/src/main.rs`

**Step 1: Add dependencies**

In `crates/devserver/Cargo.toml`, add:

```toml
futures-util = "0.3"
```

Note: `axum` already supports WebSocket upgrade natively via its `ws` feature. Add that feature:

```toml
axum = { version = "0.8", features = ["ws"] }
```

**Step 2: Implement BroadcastEventSink**

In `crates/devserver/src/main.rs`:

```rust
use tokio::sync::broadcast;
use poker_solver_tauri::{SimEventSink, SimProgressEvent, SimResultResponse};

/// Event envelope sent over WebSocket.
#[derive(Clone, Serialize)]
struct WsEvent {
    event: String,
    payload: serde_json::Value,
}

/// Broadcast-based event sink for the devserver.
#[derive(Clone)]
struct BroadcastEventSink {
    tx: broadcast::Sender<WsEvent>,
}

impl SimEventSink for BroadcastEventSink {
    fn emit_progress(&self, event: SimProgressEvent) {
        let _ = self.tx.send(WsEvent {
            event: "simulation-progress".to_string(),
            payload: serde_json::to_value(event).unwrap(),
        });
    }
    fn emit_complete(&self, event: SimResultResponse) {
        let _ = self.tx.send(WsEvent {
            event: "simulation-complete".to_string(),
            payload: serde_json::to_value(event).unwrap(),
        });
    }
    fn emit_error(&self, error: String) {
        let _ = self.tx.send(WsEvent {
            event: "simulation-error".to_string(),
            payload: serde_json::to_value(error).unwrap(),
        });
    }
}
```

**Step 3: Add start_simulation handler using the broadcast sink**

```rust
async fn handle_start_simulation(
    Extension(state): Extension<Arc<SimulationState>>,
    Extension(sink): Extension<BroadcastEventSink>,
    Json(params): Json<StartSimulationParams>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, String)> {
    result_to_response(poker_solver_tauri::start_simulation_core(
        sink,
        &state,
        params.p1_path,
        params.p2_path,
        params.num_hands,
        params.stack_depth,
    ))
}
```

**Step 4: Add WebSocket handler**

```rust
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use futures_util::SinkExt;

async fn handle_ws_events(
    ws: WebSocketUpgrade,
    Extension(tx): Extension<BroadcastEventSink>,
) -> impl axum::response::IntoResponse {
    ws.on_upgrade(move |socket| async move {
        let mut rx = tx.tx.subscribe();
        let (mut sender, _receiver) = socket.split();

        while let Ok(event) = rx.recv().await {
            let json = serde_json::to_string(&event).unwrap();
            if sender.send(Message::Text(json.into())).await.is_err() {
                break;
            }
        }
    })
}
```

**Step 5: Wire up in main()**

```rust
let (event_tx, _) = broadcast::channel::<WsEvent>(256);
let event_sink = BroadcastEventSink { tx: event_tx };

// Add to router:
.route("/api/start_simulation", post(handle_start_simulation))
.route("/ws/events", get(handle_ws_events))
.layer(Extension(event_sink))
```

Add `use futures_util::StreamExt;` to imports.

**Step 6: Verify compilation**

Run: `cargo build -p poker-solver-devserver`
Expected: compiles

**Step 7: Commit**

```bash
git add crates/devserver/
git commit -m "feat: add WebSocket event streaming and start_simulation to devserver

Implements /ws/events WebSocket endpoint that streams simulation
events. Uses broadcast channel as event sink instead of Tauri AppHandle."
```

---

### Task 4: Add `backend_url` to Frontend GlobalConfig

**Files:**
- Modify: `frontend/src/types.ts`
- Modify: `frontend/src/useGlobalConfig.ts`

**Step 1: Add `backend_url` to `GlobalConfig` type**

In `frontend/src/types.ts`, update the `GlobalConfig` interface:

```typescript
export interface GlobalConfig {
  blueprint_dir: string;
  target_exploitability: number;
  stub_range_solver?: boolean;
  flop_combo_threshold: number;
  turn_combo_threshold: number;
  backend_url: string;  // empty string = local mode
}
```

**Step 2: Update DEFAULT_CONFIG**

In `frontend/src/useGlobalConfig.ts`:

```typescript
const DEFAULT_CONFIG: GlobalConfig = {
  blueprint_dir: '',
  target_exploitability: 3.0,
  flop_combo_threshold: 200,
  turn_combo_threshold: 300,
  backend_url: '',
};
```

**Step 3: Verify frontend compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

**Step 4: Commit**

```bash
git add frontend/src/types.ts frontend/src/useGlobalConfig.ts
git commit -m "feat: add backend_url to GlobalConfig

Adds backend_url field for configuring remote backend connection.
Empty string means local Tauri mode (default)."
```

---

### Task 5: Update `invoke.ts` for Remote Routing

**Files:**
- Modify: `frontend/src/invoke.ts`

**Step 1: Update invoke.ts**

Replace the entire file:

```typescript
const DEV_SERVER_URL = 'http://localhost:3001';
const STORAGE_KEY = 'global_config';

export function isTauri(): boolean {
  return '__TAURI__' in window || '__TAURI_INTERNALS__' in window;
}

/** Read backend_url from localStorage (avoids React dependency). */
function getBackendUrl(): string {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return '';
    return JSON.parse(raw).backend_url || '';
  } catch {
    return '';
  }
}

export function isRemoteMode(): boolean {
  return getBackendUrl() !== '';
}

export async function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const backendUrl = getBackendUrl();

  // Remote mode: always use HTTP to the configured backend
  if (backendUrl) {
    return httpInvoke<T>(backendUrl, cmd, args);
  }

  // Local Tauri mode
  if (isTauri()) {
    const { invoke: tauriInvoke } = await import('@tauri-apps/api/core');
    return tauriInvoke<T>(cmd, args);
  }

  // Browser dev fallback
  return httpInvoke<T>(DEV_SERVER_URL, cmd, args);
}

async function httpInvoke<T>(baseUrl: string, cmd: string, args?: Record<string, unknown>): Promise<T> {
  const res = await fetch(`${baseUrl}/api/${cmd}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(args ?? {}),
  });

  if (!res.ok) {
    const text = await res.text();
    throw new Error(text);
  }

  return res.json();
}
```

**Step 2: Verify frontend compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

**Step 3: Commit**

```bash
git add frontend/src/invoke.ts
git commit -m "feat: route invoke calls to remote backend when configured

When backend_url is set in global config, invoke() sends HTTP POST
to the remote server instead of using Tauri IPC."
```

---

### Task 6: Create `events.ts` Event Abstraction

**Files:**
- Create: `frontend/src/events.ts`

**Step 1: Create events.ts**

```typescript
import { isRemoteMode } from './invoke';

type UnlistenFn = () => void;

const STORAGE_KEY = 'global_config';

function getBackendUrl(): string {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return '';
    return JSON.parse(raw).backend_url || '';
  } catch {
    return '';
  }
}

/** Convert HTTP URL to WebSocket URL (http→ws, https→wss). */
function toWsUrl(httpUrl: string): string {
  return httpUrl.replace(/^http/, 'ws') + '/ws/events';
}

// Shared WebSocket singleton for remote mode
let sharedWs: WebSocket | null = null;
let wsListeners: Map<string, Set<(payload: unknown) => void>> = new Map();
let wsRefCount = 0;

function ensureWebSocket(): WebSocket {
  if (sharedWs && sharedWs.readyState === WebSocket.OPEN) return sharedWs;

  const url = toWsUrl(getBackendUrl());
  const ws = new WebSocket(url);

  ws.onmessage = (msg) => {
    try {
      const data = JSON.parse(msg.data);
      const listeners = wsListeners.get(data.event);
      if (listeners) {
        listeners.forEach(cb => cb(data.payload));
      }
    } catch {
      // ignore malformed messages
    }
  };

  ws.onclose = () => {
    sharedWs = null;
    // Reconnect if there are still active listeners
    if (wsRefCount > 0) {
      setTimeout(() => {
        if (wsRefCount > 0) ensureWebSocket();
      }, 1000);
    }
  };

  sharedWs = ws;
  return ws;
}

/**
 * Listen for events — uses Tauri listen() locally, WebSocket remotely.
 * Returns an unlisten function.
 */
export async function listen<T>(
  event: string,
  handler: (payload: T) => void,
): Promise<UnlistenFn> {
  if (!isRemoteMode()) {
    // Local Tauri mode
    const { listen: tauriListen } = await import('@tauri-apps/api/event');
    return tauriListen<T>(event, (e) => handler(e.payload));
  }

  // Remote WebSocket mode
  const cb = handler as (payload: unknown) => void;
  if (!wsListeners.has(event)) {
    wsListeners.set(event, new Set());
  }
  wsListeners.get(event)!.add(cb);
  wsRefCount++;

  ensureWebSocket();

  return () => {
    const set = wsListeners.get(event);
    if (set) {
      set.delete(cb);
      if (set.size === 0) wsListeners.delete(event);
    }
    wsRefCount--;
    if (wsRefCount === 0 && sharedWs) {
      sharedWs.close();
      sharedWs = null;
    }
  };
}
```

**Step 2: Verify frontend compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

**Step 3: Commit**

```bash
git add frontend/src/events.ts
git commit -m "feat: add events.ts abstraction for Tauri/WebSocket event listening

Provides a unified listen() API that delegates to Tauri's event
system in local mode and WebSocket in remote mode. Uses a shared
WebSocket singleton with auto-reconnect."
```

---

### Task 7: Update Simulator.tsx to Use events.ts

**Files:**
- Modify: `frontend/src/Simulator.tsx`

**Step 1: Replace Tauri event imports with events.ts**

In `Simulator.tsx`, replace the event listener `useEffect` (lines 24-53):

Change from:
```typescript
useEffect(() => {
    if (!isTauri()) return;

    let cancelled = false;
    const unlisteners: (() => void)[] = [];

    (async () => {
      const { listen } = await import('@tauri-apps/api/event');
      if (cancelled) return;

      const u1 = await listen<SimulationProgress>('simulation-progress', (event) => {
        setProgress(event.payload);
      });
      // ...
    })();
    // ...
  }, []);
```

To:
```typescript
useEffect(() => {
    let cancelled = false;
    const unlisteners: (() => void)[] = [];

    (async () => {
      const { listen } = await import('./events');
      if (cancelled) return;

      const u1 = await listen<SimulationProgress>('simulation-progress', (payload) => {
        setProgress(payload);
      });
      const u2 = await listen<SimulationResult>('simulation-complete', (payload) => {
        setResult(payload);
        setRunning(false);
      });
      const u3 = await listen<string>('simulation-error', (payload) => {
        setError(payload);
        setRunning(false);
        setProgress(null);
      });
      unlisteners.push(u1, u2, u3);
    })();

    return () => {
      cancelled = true;
      unlisteners.forEach(f => f());
    };
  }, []);
```

Key changes:
- Remove `if (!isTauri()) return;` guard — events.ts handles both modes
- Import `listen` from `./events` instead of `@tauri-apps/api/event`
- Handler receives `payload` directly (not `event.payload`) — events.ts unwraps it

Also remove the `isTauri` import if no longer used elsewhere in the file. Check: `isTauri` is imported on line 2 but only used in the event listener. Remove it from the import.

**Step 2: Verify frontend compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

**Step 3: Commit**

```bash
git add frontend/src/Simulator.tsx
git commit -m "refactor: use events.ts abstraction in Simulator

Replaces direct @tauri-apps/api/event usage with the unified
events.ts listen() that works in both local and remote modes."
```

---

### Task 8: Add Remote Backend Settings to Settings.tsx

**Files:**
- Modify: `frontend/src/Settings.tsx`

**Step 1: Add connection status state and health check**

Add state and a health-check function at the top of the `Settings` component:

```typescript
const [connectionStatus, setConnectionStatus] = useState<'idle' | 'checking' | 'connected' | 'error'>('idle');

useEffect(() => {
  if (!config.backend_url) {
    setConnectionStatus('idle');
    return;
  }
  setConnectionStatus('checking');
  fetch(`${config.backend_url}/health`)
    .then(res => {
      setConnectionStatus(res.ok ? 'connected' : 'error');
    })
    .catch(() => setConnectionStatus('error'));
}, [config.backend_url]);
```

Add `useEffect` to the existing React import.

**Step 2: Add Remote Backend section to the JSX**

Insert before the Blueprint Directory section (before the `{/* Blueprint Directory */}` comment):

```tsx
{/* Remote Backend */}
<div style={{ marginBottom: '1.5rem' }}>
  <label style={{ display: 'block', fontSize: '0.8rem', color: '#888', marginBottom: '0.4rem' }}>
    Remote Backend URL
  </label>
  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
    <input
      type="text"
      value={config.backend_url}
      onChange={e => setConfig({ backend_url: e.target.value })}
      placeholder="http://192.168.1.50:3001"
      style={{
        flex: 1,
        padding: '0.45rem 0.6rem',
        background: 'rgba(255,255,255,0.06)',
        border: '1px solid rgba(255,255,255,0.12)',
        borderRadius: 6,
        color: '#eee',
        fontSize: '0.85rem',
        fontFamily: 'inherit',
      }}
    />
    {config.backend_url && (
      <div style={{
        width: 10,
        height: 10,
        borderRadius: '50%',
        flexShrink: 0,
        background: connectionStatus === 'connected' ? '#22c55e'
          : connectionStatus === 'error' ? '#ef4444'
          : connectionStatus === 'checking' ? '#eab308'
          : '#555',
      }} title={connectionStatus} />
    )}
  </div>
  <p style={{ fontSize: '0.7rem', color: '#555', marginTop: '0.3rem' }}>
    Connect to a remote solver backend. Leave empty for local mode.
  </p>
</div>
```

**Step 3: Verify frontend compiles**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors

**Step 4: Commit**

```bash
git add frontend/src/Settings.tsx
git commit -m "feat: add Remote Backend URL setting with connection indicator

Shows a text field for the backend URL and a colored dot
(green=connected, red=error, yellow=checking) from /health ping."
```

---

### Task 9: Integration Test — Manual Verification

**Files:** None (manual testing)

**Step 1: Start the devserver**

Run: `cargo run -p poker-solver-devserver`
Expected: `devserver listening on http://0.0.0.0:3001`

**Step 2: Test health endpoint**

Run: `curl http://localhost:3001/health`
Expected: `{"ok":true}`

**Step 3: Test WebSocket connection**

Run: `websocat ws://localhost:3001/ws/events` (or use browser devtools)
Expected: connection stays open, no errors

**Step 4: Start Tauri app**

Run: `cd frontend && npm run dev` (or `cargo tauri dev`)

**Step 5: Test local mode**

- Open app, verify everything works as before with no `backend_url` set

**Step 6: Test remote mode**

- Go to Settings, enter `http://localhost:3001` as backend URL
- Verify green dot appears
- Switch to Explorer, load a bundle (enter path manually)
- Switch to Arena, run a simulation
- Verify progress events stream correctly

**Step 7: Test switching back to local mode**

- Clear the backend URL field
- Verify app returns to Tauri IPC mode

**Step 8: Commit any fixes**

```bash
git commit -m "fix: integration test fixes for remote backend"
```

---

### Task 10: Update Documentation

**Files:**
- Modify: `docs/explorer.md`

**Step 1: Add Remote Backend section to explorer.md**

Add a section documenting the remote backend feature:

```markdown
## Remote Backend

WarpGTO can connect to a remote backend for GPU-accelerated solving. This is useful
when the solver machine has a powerful GPU but you want to use the desktop UI on
another machine.

### Setup

1. On the remote machine, start the devserver:
   ```bash
   cargo run -p poker-solver-devserver --release
   ```
   The server listens on `http://0.0.0.0:3001`.

2. On your local machine, open WarpGTO and go to **Settings**.

3. Enter the remote machine's URL (e.g., `http://192.168.1.50:3001`) in the
   **Remote Backend URL** field.

4. A green dot indicates a successful connection. Leave the field empty to
   return to local mode.

### Notes

- File paths (bundle loading, cache directory) refer to the **remote machine's**
  filesystem. Type paths manually when in remote mode.
- All solver commands (exploration, postflop, simulation) are routed to the remote
  backend. Window management stays local.
- No authentication is required — the server is intended for trusted LAN use.
```

**Step 2: Commit**

```bash
git add docs/explorer.md
git commit -m "docs: add remote backend usage guide to explorer.md"
```
