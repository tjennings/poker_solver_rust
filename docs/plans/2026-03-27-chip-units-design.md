# Unify on Chips as Core Unit — Design

**Date:** 2026-03-27

## Problem

The V2 tree uses BB as its unit (small_blind=0.5, big_blind=1.0) while the range-solver uses chips (small_blind=1, big_blind=2). This causes `/ 2` and `* 2` conversions scattered throughout the codebase, leading to unit mismatch bugs (e.g., EV display showing half the correct value).

## The Rule

**1 chip = 1 chip. 1 BB = 2 chips. Every number in the system is in chips unless explicitly labeled "BB".**

No more `/ 2`, `* 2`, `bb_scale`, or "half-BB" anywhere in internal code. The only place BB appears is at the display boundary (UI and CLI output).

## Config Changes

All YAML configs change from BB to chips:

```yaml
# Before (BB)                    # After (chips)
game:                            game:
  stack_depth: 100                 stack_depth: 200
  small_blind: 0.5                 small_blind: 1
  big_blind: 1.0                   big_blind: 2
```

Bet sizes in action_abstraction stay as pot fractions — unitless, unaffected.

## Changes by Component

### V2 Tree Builder (`game_tree.rs`)
- Receives chip values directly from config
- Terminal payoffs already relative to starting_stack — no formula change
- Blind posting uses chip values (sb=1, bb=2)
- `pot_at_v2_node` returns chips

### MCCFR (`mccfr.rs`)
- EV tracker stores chip values — no change to accumulation logic
- `hand_ev.bin` values in chips
- Terminal payoff computation unchanged (already uses tree's own units)

### Display Layer — Remove `* 2` Conversions
- `game_session.rs:778-790` — remove `* 2.0` on pot/stacks (tree already in chips)
- `exploration.rs:1276` — remove `* 2.0` on pot
- `exploration.rs:1280-1281` — remove `* 2.0` on stacks
- `exploration.rs:977-987` — remove `bb_scale` parameter, always 1.0

### Action Labels
- Remove `bb_scale` concept entirely
- Preflop and postflop both use the same conversion: `amount_in_chips / 2` → "Xbb"
- This is a DISPLAY conversion, not internal

### Frontend (`GameExplorer.tsx`)
- `cell.ev / 2` → display conversion to BB, labeled "BB"
- Pot/stack display: `value / 2` → "XBB", done once at the UI boundary
- `ActionBlock` header: `stack / 2` and `pot / 2` for BB display

### CLI (`inspect_spot.rs`)
- Remove existing `/ 2` conversions (currently compensating for the double conversion)
- Add single `/ 2` at output formatting, labeled "BB"

### Range-Solver Interface
- Already in chips — no change
- `TreeConfig.starting_pot`, `effective_stack` stay as-is

### Simulation Harness (`simulation.rs:201-203`)
- Remove `stack_depth * 2` — config already in chips

### CFVnet Datagen
- Uses range-solver (chips) — minimal changes
- `half_pot = pot / 2.0` idiom stays (that's pot math, not unit conversion)

### Existing Snapshots / Trained Models
- **Breaking change**: existing `hand_ev.bin` files have EVs in old BB units
- Need migration or version flag
- Strategy files (`strategy.bin`) store action probabilities — unitless, unaffected
- Regret files store scaled integers — the scale factor doesn't depend on units

## Display Boundary

The ONLY place BB appears:

| Location | Conversion | Label |
|----------|-----------|-------|
| UI matrix EV overlay | `chips / 2` | "+X.Y" (implicit BB) |
| UI pot/stack display | `chips / 2` | "XBB" |
| UI action labels | `chips / 2` | "Xbb" |
| CLI inspect-spot | `chips / 2` | "XBB" |
| CLI training TUI | `chips / 2` | "BB" |

## Files to Change

### Config files (update values)
- `sample_configurations/*.yaml` — all stack_depth, small_blind, big_blind values

### Core crate
- `crates/core/src/blueprint_v2/config.rs` — default values
- `crates/core/src/blueprint_v2/game_tree.rs` — verify construction uses config directly
- `crates/core/src/blueprint_v2/mccfr.rs` — verify EV tracker, remove any conversions
- `crates/core/src/simulation.rs` — remove `* 2` conversion
- `crates/core/src/info_key.rs` — SPR computation

### Tauri app
- `crates/tauri-app/src/game_session.rs` — remove `* 2.0` in pot/stack display
- `crates/tauri-app/src/exploration.rs` — remove `* 2.0`, remove `bb_scale`
- `crates/tauri-app/src/postflop.rs` — verify subgame construction

### Frontend
- `frontend/src/GameExplorer.tsx` — verify EV/pot/stack display conversions
- `frontend/src/Explorer.tsx` — `ActionBlock` header formatting

### CLI
- `crates/trainer/src/inspect_spot.rs` — update display formatting

## Migration

- Update all YAML configs: multiply stack_depth by 2, small_blind from 0.5→1, big_blind from 1.0→2
- Existing trained models: EVs in `hand_ev.bin` will be off by 2× — retrain or add a version marker
- Document the change in `docs/training.md`
