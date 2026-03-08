# Blueprint Explorer: Load Strategy + Range Solver Redesign

## Overview

Redesign the Explorer to support two distinct modes:

1. **Load Strategy** — browse a trained blueprint's preflop strategy, then transition into postflop range solving with blueprint-derived ranges
2. **Range Solve** — standalone postflop solver with user-specified ranges (existing flow, unchanged)

## Global Configuration (Gear Icon)

The gear icon opens a modal with global settings:

- **Blueprint Directory** — OS native directory picker (Tauri dialog API). This is the root directory containing one or more trained blueprint bundles.
- **Target Exploitability** — moved here from the per-game config. Applies to all postflop range solves.

These persist across sessions (stored in app settings / local storage).

## Main Screen Layout

The initial screen shows two entry points (replaces the current icon-only split button):

- **Load Strategy** — opens blueprint selection flow
- **Range Solve** — opens the existing Game Configuration dialog (unchanged except target exploitability is removed, now in global config)

## Load Strategy Flow

### Step 1: Blueprint Selection

When the user clicks Load Strategy, a config modal appears with:

- **Blueprint dropdown** — lists available blueprints found in the configured Blueprint Directory. Each entry is a subdirectory containing `config.yaml` + `strategy.bin` (or `final/strategy.bin` or latest `snapshot_NNNN/strategy.bin`). Display format: directory name + stack depth from config.
- **Read-only fields** populated from the selected blueprint's `config.yaml`:
  - Stack Depth (BB)
  - Bet Sizes (per street, from `action_abstraction`)
- **No range fields** — ranges are implicit from the blueprint and preflop navigation path.

Click **Apply** to load the blueprint and enter preflop navigation.

### Step 2: Preflop Navigation

After applying, the explorer shows the preflop strategy from the blueprint:

- 13x13 matrix displays action frequencies for the current decision node (starts at SB's first action)
- Action strip shows available actions at the current node
- User clicks actions to navigate the preflop tree
- **Fold terminal**: matrix stays showing the last decision point, no new action card added. Dead end — user must navigate back.
- **Non-fold terminal** (call after a raise): flop picker appears automatically

Strategy data comes from `BlueprintV2Strategy` — the existing `load_blueprint_v2_core` / `get_strategy_matrix` flow handles this.

### Step 3: Preflop → Postflop Transition

When the user reaches a non-fold preflop terminal:

1. **Derive ranges**: Walk the preflop action path. For each decision node, multiply each canonical hand's weight by the strategy probability for the chosen action. Expand 169 canonical hands → 1326 combos with uniform suit distribution.
2. **Derive pot/stack**: Computed from the preflop terminal node's bet amounts and the blueprint's stack depth.
3. **Show flop picker**: User selects 3 flop cards.

### Step 4: Postflop Solving

Once the user picks a flop:

1. Build a `PostflopConfig` from the derived ranges, pot, stack, and the blueprint's postflop bet sizes.
2. Compute cache key: `hash(config fields + flop cards)`.
3. Check cache at `{blueprint_dir}/spots/{hash}.bin`.
4. **Cache hit** → load cached solution, display immediately with metadata ("Cached — exploitability: X% pot").
5. **Cache miss** → show solve button. After solve completes, write to cache.
6. Postflop navigation proceeds as today (play actions, close street → turn/river).

For turn/river streets, the cache key extends to include the action history and new board cards (see [cache design doc](2026-03-08-postflop-solve-cache-design.md)). These are each distinct spots and written to their own cache files.  

### Bet Size Mapping: Blueprint → Range Solver

The blueprint's `action_abstraction` defines postflop bet sizes as pot fractions per street and raise depth (e.g., flop: `[[0.33, 0.67, 1.0], [0.5, 1.0], [1.0]]`). These need to be translated into the range solver's `BetSizeOptions` format (e.g., `"33%,67%,100%"`).

Use the first raise depth's sizes for bet sizes and the second depth's sizes for raise sizes (matching the range solver's bet/raise split). If only one depth exists, use the same for both.

## Range Solve Flow (Unchanged)

Clicking Range Solve opens the existing Game Configuration dialog:

- OOP Range, IP Range (editable)
- Pot, Effective Stack (editable)
- OOP/IP Bet Sizes, Raise Sizes (editable)
- Target Exploitability removed (now in global config)

Apply → flop picker → solve → navigate. No caching in this mode (for now).

## Blueprint Directory Discovery

To populate the blueprint dropdown, scan the configured Blueprint Directory for subdirectories containing a `config.yaml` file. For each:

1. Parse `config.yaml` to extract display metadata (stack depth, player count)
2. Check for strategy availability (`final/strategy.bin`, `strategy.bin`, or `snapshot_NNNN/strategy.bin`)
3. Show in dropdown as: `{dir_name} ({stack_depth}BB)`

Directories without a valid `config.yaml` or strategy file are skipped silently.
For blueprint directories with multiple snapshots, only load the latest snapshot (largest sequence number)

## Backend Changes

### New Tauri Commands

- `get_global_config` / `set_global_config` — read/write global settings (blueprint dir, target exploitability)
- `list_blueprints` — scan blueprint directory, return list of `{ name, path, stack_depth, has_strategy }`
- `get_blueprint_preflop_ranges` — given a blueprint path + preflop action history, return the derived 1326-combo ranges for OOP and IP

### Modified Commands

- `postflop_solve_street_core` — accept optional `cache_dir` parameter; after solve, write cache file
- `postflop_set_config_core` — accept config without ranges (populated externally from blueprint)

### New State

- `PostflopState` gains an optional `blueprint_dir: Option<PathBuf>` for cache lookups
- Global config stored via Tauri's app settings or a config file

## Frontend Changes

- `App.tsx`: Add gear icon → global config modal (blueprint dir picker + target exploitability)
- `Explorer.tsx`: Load Strategy button → blueprint selection modal → preflop navigation (reuse existing matrix/action strip components) → automatic postflop transition
- `PostflopExplorer.tsx`: Remove target exploitability from game config. Accept optional pre-populated config (from blueprint flow).

## Open Questions

- Should the global config persist via Tauri's `app_data_dir` config file, or `localStorage` in the frontend?
- When listing blueprints, should we show snapshot count or training progress metadata?
