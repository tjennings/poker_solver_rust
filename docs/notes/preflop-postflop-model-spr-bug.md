# Preflop Solver: Postflop Model SPR Interpolation Bug

## Date: 2026-03-05

## Problem Statement

The preflop LCFR solver produces degenerate strategies in two scenarios:
1. **BB overfolding strong hands** (Q8s, K5s) facing SB 2bb open raise
2. **AA/KK/QQ/JJ calling instead of raising** when SB faces BB raise after limp (node 155)

Both are caused by the same root issue: the postflop model's terminal values create a structural convergence trap in the preflop solver.

## Investigation Summary

### Tools Built (in `crates/trainer/src/main.rs`)

- **`inspect-preflop`** — dump average strategy at any node for all 169 hands
- **`trace-regrets`** — trace per-iteration regret/strategy evolution for specific hands at a node
- **`decompose-regrets`** — per-opponent breakdown of regret contributions at a target node
- **`trace-terminals`** — walk subtree from a decision node, print pot/investments/SPR/model-value at every terminal

### Key Method Added (in `crates/core/src/preflop/solver.rs`)

- `decompose_regrets_at()` — single-iteration CFR traversal that returns per-opponent regret deltas
- `dump_terminal_values()` / `walk_terminals()` — recursive tree walker printing terminal diagnostics

### Bugs Fixed Along the Way

1. `decompose_regrets_at()` referenced `total_slots()` (doesn't exist) — changed to `total_size` field
2. Hand swap bug in `decompose_regrets_at()` — was passing `(opp_hand, hero_hand)` to `cfr_traverse` when hero_pos=1, but should always pass `(hero_hand, opp_hand)` since `cfr_traverse` uses `hero_pos` to determine which is which

## Root Cause Analysis

### The Formula

In `postflop_showdown_value()` (solver.rs ~line 801):

```rust
let model_value = pf_ev_frac * pot_f + (pot_f / 2.0 - hero_inv);
// At showdown terminals hero_inv = pot_f/2, so this simplifies to:
// model_value = pf_ev_frac * pot_f

if model_spr <= 0.0 || actual_spr >= model_spr {
    return model_value;  // No interpolation when SPR >= model
}
let ratio = actual_spr / model_spr;
eq_value + (model_value - eq_value) * ratio  // Interpolate toward equity
```

### What `pf_ev_frac` Means

The postflop model stores EV in "starting pot fraction" units. The postflop terminal payoff is `eq * pot_fraction - pot_fraction/2`, where `pot_fraction` is how much the pot grew relative to the starting pot.

For AA vs 72o at SPR=3.5: `pf_ev_frac = 2.96`, meaning AA profits ~3x the starting pot through optimal postflop play (betting multiple streets, getting stacks in). This is correct — at SPR=3.5, max pot = 8x starting pot, and 0.88 * 8 - 4 = 3.04.

### The Structural Asymmetry

At node 155 (SB facing BB raise after limp), for AA vs 72o:

| Action | Terminal Pot | SPR | Final Value | How Computed |
|-|-|-|-|-|
| Call | 24 | 3.67 | **71.08** | Full model (SPR > 3.5) |
| R(0.33)→call | 38 | 2.13 | **74.24** | Interpolated (ratio=0.61) |
| R(1.0)→call | 72 | 0.89 | **74.75** | Interpolated (ratio=0.25) |
| AllIn→call | 200 | 0.00 | **76.69** | Pure equity (ratio=0) |

**Call is an immediate leaf node** — gets 71.08 directly, no opponent decisions.

**All raise paths go through opponent decision trees** (7 actions per node, raise_cap=4). Under uniform strategies at iteration 1:
- BB folds 1/7 of the time → hero gets only 12 (dead money)
- At deeper levels, hero also folds uniformly, further diluting value
- R(0.33) under uniform play ≈ 44 chips (vs Call = 71)

### Why DCFR Can't Recover

1. Iteration 1: Call (71) >> all raises (~44) due to fold dilution in deep tree
2. Regret matching locks in Call as dominant action
3. DCFR discounting amplifies early bias
4. Self-reinforcing: hero always calls → opponent raise subtrees never develop correct strategies

### The Fundamental Issue

The postflop model value at Call (71 chips in a 24-chip pot) **already accounts for aggressive postflop play** where AA bets multiple streets and gets most of the stack committed. This is the same value that the preflop raise subtree is trying to capture. There's double-counting: the raise subtree says "build a bigger pot preflop" while the model says "I'll build a big pot postflop anyway."

## Terminal SPR Distribution

Terminals from the preflop tree span SPR 0 to 12:

| SPR Range | Example | Current Behavior | Problem |
|-|-|-|-|
| 0 | All-in | Raw equity | Correct |
| 0.08–0.89 | 3bet/4bet calls | 70-97% interpolated toward equity | Model underweighted |
| 1.5–2.1 | Small 3bet calls | 40-60% interpolated | Model underweighted |
| 3.5–3.7 | Single raise call | Full model (no interp) | Model at wrong SPR |
| 12.0 | Open-call | Full model (no interp) | Model trained at 3.5, not 12 |

## Recommended Fix: Multi-SPR Postflop Models

Build postflop models at multiple SPR values (e.g., `postflop_sprs: [1.0, 3.5, 8.0]` or `[0.5, 1.5, 3.5, 6.0, 12.0]`). The infrastructure already exists:

- `select_closest_spr()` picks the nearest model
- `load_multi()` scans `spr_*` subdirectories
- `PostflopModelConfig.postflop_sprs` accepts a vec
- `PostflopAbstraction::build_for_spr()` builds at explicit SPR

This eliminates interpolation artifacts by having a model close to every terminal's actual SPR. Key benefits:
- SPR=12 terminals use a model trained at SPR=12 (where postflop play is deeper, less "all money goes in")
- SPR=1 terminals use a model trained at SPR=1 (shallower postflop, closer to raw equity)
- Minimal interpolation needed between adjacent models

## Blocker Hypothesis (Debunked)

Investigated whether card removal (blocker effects) caused Q8s overfolding. Analysis showed:
- Q-blocked hands: per-weight fold-call diff = 9.33
- 8-blocked hands: per-weight fold-call diff = 9.26
- Unblocked hands: per-weight fold-call diff = 5.38

Blockers actually **help** Q8s (reduce bad matchups). The unblocked group (53% of total diff) dominates. Conclusion: not a blocker issue.

## Iteration 1 Regret Data (AA at node 155)

Even at iteration 1 with uniform strategies everywhere:
- Fold: -58K
- **Call: +41K** (best)
- **AllIn: +19K** (second)
- R(0.33): +1.5K
- R(0.5): +1K
- **R(1.0): -1.5K** (negative!)
- **R(1.3): -2.6K** (negative!)

Bigger raises produce MORE NEGATIVE regret — the opposite of correct for AA. This is the "dead on arrival" signal that moderate raises never recover from.

## Files Modified

- `crates/core/src/preflop/solver.rs` — added `decompose_regrets_at()`, `dump_terminal_values()`, `walk_terminals()`, fixed 2 bugs
- `crates/trainer/src/main.rs` — added `InspectPreflop`, `TraceRegrets`, `DecomposeRegrets`, `TraceTerminals` CLI subcommands with handlers
