# Dead Money EV Model — Design

## Problem

The MCCFR trainer treats blind postings as player investments. This means:
- SB folding preflop shows EV = -0.5 instead of 0
- No SB open hand should have negative EV (fold = 0, blinds are sunk cost)
- The `invested` field conflates forced blind postings with voluntary betting decisions

GTO Wizard and standard poker theory treat blinds as dead money — they're in the pot but owned by neither player. Voluntary investment starts at 0.

## Solution

### Terminal value formula (dead money model)

```
vol[p] = invested[p] - blinds[p]    // voluntary investment only
initial_pot = blinds[0] + blinds[1]  // + future antes/straddles

Winner:  initial_pot + vol[opponent] - rake
Loser:   -vol[self]
Tie:     initial_pot / 2 - rake / 2
```

This is a constant per-position offset from the current model. Regret differences are unchanged, so strategy convergence is identical.

### Changes

1. **`GameTree`** — add `blinds: [f64; 2]` field, set at build time.

2. **`GameNode::Terminal`** — rename `invested` to `initial_pot` (the total pot at this terminal node). The per-player invested amounts are derived during traversal, not stored.

   Actually, we still need per-player invested amounts at terminals for the dead money formula. Rename field to `voluntary: [f64; 2]` and add `initial_pot: f64`, OR keep `invested` (total per player including blinds) and compute voluntary at terminal_value time from `tree.blinds`.

   **Decision:** Keep `invested: [f64; 2]` in Terminal nodes (total per player including blinds — needed for correct pot math). Add `blinds: [f64; 2]` to `GameTree`. Compute `vol = invested - blinds` in `terminal_value`.

3. **`terminal_value()`** — accept `blinds: [f64; 2]`, use dead money formula.

4. **`traverse_external()`** — pass `tree.blinds` to `terminal_value`.

5. **`ev_sum` in trainer** — automatically picks up new values, no change needed.

6. **Tree construction** — `BuildState.invested` stays the same (needed for bet sizing, remaining stack, call amounts). No change to tree building logic.

### Naming

- `GameNode::Terminal.invested` → rename to `initial_pot` to reflect the dead money concept (stores the total pot at terminal, which includes initial dead money + voluntary investments)
- `BuildState.invested` → internal bookkeeping, can stay as-is

### Future: Antes & Straddles

The `blinds: [f64; 2]` field generalizes to include antes. For antes posted by all players, add them to each player's blind amount. For a single ante (e.g., BB ante), add to that player's blind. The initial_pot calculation remains `blinds[0] + blinds[1]`.
