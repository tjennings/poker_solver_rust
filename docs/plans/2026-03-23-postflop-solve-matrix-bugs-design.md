# Postflop Solve Matrix Bugs — Design

**Date:** 2026-03-23

## Problem

Three bugs in the postflop explorer's transition from blueprint mode to solver mode:

1. **Position label flips from BB to SB** — `build_matrix_from_snapshot` hardcodes `dealer: 1` (range-solver convention: player 1 = IP = SB = dealer). The depth-limited solver uses V2 tree convention (dealer = 0, OOP = seat 1). When the V2 tree's `player` values are fed into `build_matrix_from_snapshot` with the wrong `dealer`, the frontend formula `player === dealer ? 'SB' : 'BB'` produces incorrect labels.

2. **EV data lost** — Blueprint EVs from `hand_ev.bin` disappear when the solver matrix replaces the blueprint matrix. Full-depth solver computes EVs after `finalize()`, but depth-limited solver never computes them.

3. **Stale matrix on solve start** — `matrix_snapshot` isn't cleared when starting a new solve. For the depth-limited path (where the initial matrix is created inside a spawned thread), the first poll can return a stale matrix from a previous solve.

## Fix 1: Add `dealer` to `MatrixSnapshot`

### Changes

**`MatrixSnapshot` struct** — add `dealer: u8` field.

**`capture_matrix_snapshot`** (range-solver path):
- Set `dealer: 1` (IP is always the dealer in range-solver convention).

**`snapshot_from_subgame`** — add `dealer: u8` parameter, store in snapshot.

**`build_matrix_from_snapshot`** — use `snap.dealer` instead of hardcoded `1`.

**`solve_depth_limited` `make_matrix` closure:**
- Get root player from `tree.nodes[tree.root]` (V2 convention) instead of hardcoding `player = 0`.
- Pass `tree.dealer` to `snapshot_from_subgame`.

**`postflop_play_action_subgame`:**
- Pass `result.tree.dealer` to `snapshot_from_subgame`.

### Convention Summary

| Source | `player` at root | `dealer` | `player === dealer` at root |
|--------|------------------|----------|-----------------------------|
| Range-solver | 0 (OOP/BB) | 1 (IP/SB) | false → "BB" |
| V2 subgame tree | 1 (BB/OOP) | 0 (SB) | false → "BB" |

Both produce correct labels because `dealer` matches the source's own convention.

## Fix 2: EV After Depth-Limited Solve

After training completes in `solve_depth_limited`, compute per-hand EVs from the `SubgameStrategy` and include them in the final snapshot.

The solver can produce expected values per combo at the root node. Add these to the `MatrixSnapshot.hand_evs` field for the final snapshot only (not during iteration snapshots).

## Fix 3: Clear Stale Snapshot

**`postflop_solve_street_impl`** — add `*state.matrix_snapshot.write() = None;` before dispatching to either solver path.

The depth-limited path already writes matrix snapshots every 10 iterations. The gap between thread spawn and first snapshot (during `build_subgame_solver`) shows no matrix (acceptable — solver is initializing).

## Files to Change

- `crates/tauri-app/src/postflop.rs` — all three fixes
- No frontend changes needed (display logic already uses `matrix.player === matrix.dealer`)
