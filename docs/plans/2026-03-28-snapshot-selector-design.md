# Snapshot Selector for Blueprint Loading

**Date**: 2026-03-28
**Status**: Approved

## Problem

`load_blueprint_v2` always auto-picks the latest snapshot. No way to load earlier snapshots for comparison or debugging convergence at different iteration counts.

## Solution

Two-step loading flow: pick blueprint → select snapshot (defaulting to latest) → load.

## Changes

### 1. Backend — `exploration.rs`
- Add `list_snapshots` Tauri command: takes blueprint path, scans for `snapshot_*` dirs, reads `metadata.json` from each (iteration count, elapsed minutes), returns `Vec<SnapshotEntry>` sorted newest-first.
- Add optional `snapshot: Option<String>` param to `load_blueprint_v2`. When provided, load strategy from that specific snapshot dir. When `None`, existing auto-detect behavior (latest snapshot).

### 2. Frontend — `Explorer.tsx`
- After user clicks a blueprint in the list, call `invoke('list_snapshots', { path })`.
- Show a dropdown/selector with snapshot entries (e.g. "snapshot_0005 — 10M iterations"). Latest pre-selected.
- "Load" button calls `load_blueprint_v2` with the chosen snapshot name.

### 3. Types — `types.ts`
- Add `SnapshotEntry { name: string, iterations: number | null, elapsed_minutes: number | null, path: string }`.

### 4. Dev server — `devserver/main.rs`
- Mirror `list_snapshots` as `POST /api/list_snapshots`.

## Not changing
- `list_blueprints` return type (stays lightweight, no snapshot list)
- Auto-detection when `snapshot` is `None` (backwards compatible)
- `load_bundle` command (existing generic loader unchanged)
