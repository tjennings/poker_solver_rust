# TUI Spot Resolution Error Indicator

**Date**: 2026-03-27
**Status**: Approved

## Problem

When a scenario's spot notation doesn't match the game tree (e.g. action sizes changed), `resolve_spot` returns `None` and the code silently falls back to the root node, showing "Preflop" with no indication of failure. This masks config errors.

## Solution

Add `error_message: Option<String>` to `HandGridState`. When set, the grid renderer shows the error text centered in the grid area instead of the 13×13 matrix.

## Changes

1. **`blueprint_tui_widgets.rs`** — Add `pub error_message: Option<String>` to `HandGridState`, default `None`.

2. **`main.rs` scenario resolution (~line 286)** — When `resolve_spot` returns `None`, set `error_message = Some(format!("Spot failed to resolve: {}", sc.spot))`. Still use root as `node_idx` so the struct is valid, but the error flag prevents rendering the grid.

3. **Grid render function** — Before drawing cells, check `error_message`. If `Some`, render the message centered in the grid area with a warning style (yellow/dim) and return early.

## Non-goals

- No change to random scenario carousel (it generates spots from the tree, always resolves)
- No change to strategy refresh callback
- No retry/recovery mechanism
