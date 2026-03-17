# Remote Backend Design

**Date:** 2026-03-16
**Goal:** Allow the Tauri desktop app to optionally connect to a remote backend for GPU-accelerated solving.

## Motivation

The remote machine has a powerful GPU. The user wants to run the solver backend there while keeping the Tauri frontend on their local machine. LAN first, cloud later.

## Architecture

The Tauri app always runs as the desktop shell. A "Remote Backend URL" setting in the UI lets the user optionally redirect solver commands to a remote devserver instance over HTTP, instead of using the local in-process Rust backend via Tauri IPC.

### Modes

1. **Local mode (default)** — Tauri IPC to in-process Rust backend. No change from today.
2. **Remote mode** — When `backendUrl` is set in settings, `invoke.ts` routes commands via HTTP POST to `{backendUrl}/api/{cmd}` instead of Tauri IPC. Events stream over WebSocket from `{backendUrl}/ws/events`.

Tauri still handles window management, file dialogs (local mode), and app lifecycle. Only solver/data commands are redirected.

## Components

### 1. Frontend: `invoke.ts` Changes

Current logic:
```
if (isTauri()) → Tauri IPC
else → HTTP to localhost:3001
```

New logic:
```
if (remoteUrl) → HTTP POST to remoteUrl/api/{cmd}
else if (isTauri()) → Tauri IPC
else → HTTP POST to localhost:3001
```

`remoteUrl` is read from app settings (persisted in localStorage or Tauri store).

### 2. Frontend: `events.ts` (New File)

A thin abstraction mirroring Tauri's `listen()` API:

- **Local mode:** Delegates to `@tauri-apps/api/event` `listen()`
- **Remote mode:** Opens a WebSocket to `{backendUrl}/ws/events`, dispatches incoming JSON messages to registered listeners keyed by event name

Returns an unlisten function in both modes so callers don't change.

### 3. Frontend: Settings UI

- Add "Remote Backend URL" text field to the settings panel
- Connection status indicator (green/red dot) that pings `GET /health`
- Empty field = local mode

### 4. Frontend: `Simulator.tsx` Changes

Replace direct `@tauri-apps/api/event` import with the new `events.ts` abstraction. Same API surface, transparent local/remote switch.

### 5. Devserver: Missing Simulation Commands

Add the 4 missing simulation endpoints:
- `POST /api/list_strategy_sources`
- `POST /api/start_simulation`
- `POST /api/stop_simulation`
- `POST /api/get_simulation_result`

These call the existing `_core` functions from `poker_solver_tauri::simulation`.

### 6. Devserver: WebSocket Events

Add `GET /ws/events` endpoint using `axum`'s WebSocket support.

Replace the Tauri `AppHandle` event emission with a `tokio::sync::broadcast` channel. The `_core` functions that currently take an `AppHandle` for emitting events will accept an alternative event sink (the broadcast sender). WebSocket clients subscribe to the broadcast receiver and forward messages as JSON:

```json
{ "event": "simulation-progress", "payload": { ... } }
```

Events to support:
- `simulation-progress`
- `simulation-complete`
- `simulation-error`
- `bucket-computation`
- `bucket-computation-done`

### 7. Devserver: Health Endpoint

Add `GET /health` returning `{ "ok": true }`. Used by the frontend to test connectivity and show connection status.

### 8. File Path Handling

Commands that take file paths (`load_bundle`, `load_blueprint_v2`, `list_blueprints`, `postflop_set_cache_dir`) refer to the backend machine's filesystem.

- **Local mode:** Tauri file picker works as today
- **Remote mode:** File picker is replaced with a text input for manually entering paths on the remote machine

No remote file browser for now — manual path entry is sufficient.

## Error Handling & UX

- On app launch with `backendUrl` set, ping `/health` and show connection status
- Failed remote calls show an error toast with retry option
- No automatic fallback from remote to local — user explicitly chooses the mode
- HTTP over LAN adds ~1ms overhead — negligible for all commands

## Security

- No authentication for now (LAN-only use case)
- Devserver already binds `0.0.0.0:3001` — accessible on LAN

## What Does NOT Change

- Rust `_core` functions (solver logic, game tree, abstractions)
- `core` crate, `range-solver`, `trainer`
- Tauri app lifecycle, window management
- Any solver algorithms or data structures

## Scope Summary

| Component | Change |
|-----------|--------|
| `frontend/src/invoke.ts` | Add remote URL conditional |
| `frontend/src/events.ts` | New: Tauri listen / WebSocket abstraction |
| Settings UI | New: backend URL field + connection indicator |
| `frontend/src/Simulator.tsx` | Use `events.ts` instead of direct Tauri import |
| `crates/devserver/src/main.rs` | Add simulation commands, WebSocket events, health endpoint |
| `crates/tauri-app/src/simulation.rs` | Refactor event emission to support broadcast channel |
