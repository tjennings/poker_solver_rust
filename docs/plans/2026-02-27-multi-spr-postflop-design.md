# Multi-SPR Postflop Model Design

## Problem

`postflop_sprs: [2, 6, 20]` is parsed correctly but only the first SPR is ever used. `PostflopAbstraction::build()` calls `config.primary_spr()` (first element), builds one tree, one value table, and stores a single `spr: f64`. The preflop solver has no way to select between multiple SPR models.

## Goal

Build a `PostflopAbstraction` per configured SPR. At showdown terminals, the preflop solver selects the closest SPR model to the terminal's actual SPR and uses that model's EV table.

## Design

### Data Flow

1. **Build phase**: For each SPR in `config.postflop_sprs`, build a separate `PostflopAbstraction` (tree + values + hand_avg_values). Each gets its own postflop tree sized to that SPR.
2. **Attach phase**: Store all abstractions in `PostflopState` as a sorted vec.
3. **Lookup phase**: At each showdown terminal, compute `actual_spr`, find the closest model by absolute distance, use that model's `avg_ev()`.

### Key Changes

**`PostflopState` (solver.rs):**
- `abstraction: PostflopAbstraction` → `abstractions: Vec<PostflopAbstraction>` (sorted by spr)

**`postflop_showdown_value` (solver.rs):**
- Compute `actual_spr` first (same formula as now)
- Select closest abstraction by `|model.spr - actual_spr|`
- Use that model's `avg_ev()` and `spr` for the existing interpolation logic

**`PreflopSolver::attach_postflop` (solver.rs):**
- Takes `Vec<PostflopAbstraction>` instead of a single one

**Trainer (main.rs):**
- Loop over `config.postflop_sprs`, build an abstraction per SPR
- Pass the vec to `attach_postflop`

**Bundle (postflop_bundle.rs):**
- Save/load one bundle per SPR (subdirectories like `postflop/spr_2.0/`, `postflop/spr_6.0/`)
- Backward compat: if loading a single-SPR bundle from old format, wrap in a vec

### Unchanged

- `PostflopAbstraction` struct — still represents one SPR
- Per-flop solve logic (mccfr/exhaustive)
- Interpolation within `postflop_showdown_value` for `actual_spr < model_spr`
- Limped-pot fallback to raw equity

### SPR Selection

Closest model by absolute distance. No interpolation between models. The existing within-model interpolation (`ratio = actual_spr / model_spr`) handles gaps between actual and selected model SPR.

### Training Time

N SPRs = ~Nx postflop solve time. Each SPR builds its own tree and solves all flops independently.
