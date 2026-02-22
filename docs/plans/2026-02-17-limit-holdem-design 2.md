# Limit Hold'em Game Design

## Overview

A configurable Limit Hold'em (LHE) game for training and evaluating an SD-CFR model. Supports both full Heads-Up Limit Hold'em (HULH, 4 streets) and Flop Hold'em (2 streets) via a single configurable game struct.

## Game Rules

- **Players:** 2 (heads-up, zero-sum)
- **Blinds:** SB posts 1 unit, BB posts 2 units
- **Betting:** Fixed bet sizes — small bet (2 units) on preflop/flop, big bet (4 units) on turn/river
- **Raise caps:** Configurable per phase — default 3 raises on early streets (preflop/flop), 4 on late streets (turn/river)
- **Position:** P1 (SB) acts first preflop; P2 (BB) acts first on all postflop streets
- **Showdown:** Best 5-card hand from 2 hole cards + 5 community cards wins. Ties split the pot.
- **Flop Hold'em variant:** Identical but only preflop + flop (2 streets)

## Architecture: Standalone Game (Approach 1)

New file `crates/core/src/game/limit_holdem.rs` implementing the `Game` trait independently. Shares utilities (hand evaluation, card canonicalization) with existing HUNL code but has its own action logic, state transitions, and config.

### Configuration

```rust
pub struct LimitHoldemConfig {
    pub stack_depth: u32,        // In BBs (stacks = stack_depth * 2 in SB units)
    pub num_streets: u8,         // 2 = Flop HE, 4 = full HULH
    pub max_raises_early: u8,    // Preflop/flop raise cap (default 3)
    pub max_raises_late: u8,     // Turn/river raise cap (default 4)
    pub small_bet: u32,          // SB units: 2 (= 1 BB)
    pub big_bet: u32,            // SB units: 4 (= 2 BB)
}
```

### State

```rust
pub struct LimitHoldemState {
    pub p1_holding: [Card; 2],
    pub p2_holding: [Card; 2],
    pub full_board: [Card; 5],   // Pre-dealt, revealed progressively
    pub board_len: u8,           // 0, 3, 4, or 5
    pub street: Street,          // Preflop, Flop, Turn, River
    pub pot: u32,                // SB units
    pub stacks: [u32; 2],
    pub to_call: u32,            // Amount to match current bet
    pub to_act: Option<Player>,  // None if terminal
    pub street_raises: u8,       // Raises so far this street
    pub history: ArrayVec<(Street, Action), 48>,
    pub terminal: Option<TerminalType>,
}
```

### Unit System

- SB = 1, BB = 2 (matches existing HUNL convention)
- Small bet = 2 units (preflop/flop)
- Big bet = 4 units (turn/river)
- Stacks start at `stack_depth * 2` SB units

### Action Model

- `Action::Bet(0)` / `Action::Raise(0)` — single fixed size, game resolves amount by street
- No bet sizing index needed since there's only one possible size per street

**Available actions:**

| Situation | Actions |
|-----------|---------|
| Facing bet/raise (`to_call > 0`) | Fold, Call, Raise(0) (if under raise cap) |
| No bet facing (`to_call == 0`) | Check, Bet(0) (if under raise cap) |

**Raise cap per street:**
- Early (preflop, flop): `max_raises_early` (default 3)
- Late (turn, river): `max_raises_late` (default 4)

### Street Transitions

- Street ends when both players have acted and last action was Call or both Checked
- On transition: advance street, reveal board cards, reset `street_raises = 0`, set `to_act` to P2 (OOP postflop)
- Board reveal: preflop → 0 cards, flop → 3, turn → 4, river → 5
- If `num_streets == 2`: game ends after flop betting round (showdown)

### Terminal Conditions

- Fold → opponent wins pot
- Final street's betting completes → showdown
- Both players all-in → run out remaining streets, showdown

## SD-CFR Integration

### State Encoder

New `LimitHoldemEncoder` implementing `StateEncoder<LimitHoldemState>`:

- **Cards:** Reuse `card_features::canonicalize(hole, board)` → `[i32; 7]`
- **Bets:** Reuse HUNL 32-float bets array format. Fixed bets expressed as pot fractions for NN compatibility.

### Network Config

- `num_actions: 3` (fold, check/call, bet/raise) — much smaller than NLHE's 8
- `hidden_dim: 128` (smaller than NLHE's 256, game is simpler)

### Training YAML

```yaml
game_type: limit_holdem
game:
  stack_depth: 50
  num_streets: 4
  max_raises_early: 3
  max_raises_late: 4

deals:
  count: 50000
  seed: 42

training:
  iterations: 500
  traversals_per_iter: 1000
  output_dir: "./lhe_sdcfr"

network:
  hidden_dim: 128
  num_actions: 3

sgd:
  steps: 4000
  batch_size: 4096
  learning_rate: 0.001

memory:
  advantage_cap: 5000000

checkpoint:
  interval: 100
```

### Trainer CLI

Add `game_type` field to YAML config (defaults to `hunl_postflop` for backward compatibility). Trainer dispatches to the right game + encoder based on this field.

## Evaluation: `eval-lhe` Command

```bash
cargo run -p poker-solver-trainer -- eval-lhe -m ./lhe_sdcfr/checkpoint.bin --eval-deals 10000
```

### Sampled Exploitability

- Sample N deals (e.g., 10K)
- For each deal, walk the full game tree
- At opponent nodes: query strategy NN for action probabilities
- At BR player nodes: pick max-EV action
- Report `(BR_p1_EV + BR_p2_EV) / 2` in milli-big-bets per hand (mbb/h)

### Loss Tracking

- Log advantage network loss per iteration during training
- Print summary: initial loss, final loss, convergence trend

### Strategy Visualization (Terminal ASCII)

Color-coded 13×13 hand matrices with stacked action distributions per cell:

1. **Preflop RFI** — SB opening strategy: fold / call / raise frequencies
2. **Preflop RFI Response** — BB facing SB raise: fold / call / 3-bet frequencies
3. **Flop strategies** — ~20-30 representative flops sampled by texture category:
   - Monotone / two-tone / rainbow
   - Paired / unpaired
   - High / medium / low
   - OOP and IP action frequencies on each

Each cell renders a stacked ANSI color bar:
```
AKs: ███░░  (60% raise, 30% call, 10% fold)
```

## Testing (TDD)

Thorough test-driven test suite covering:

- **Action generation:** Correct actions at every decision point, raise cap enforcement
- **Street transitions:** Board reveal, position switch, state reset
- **Terminal detection:** Fold, showdown, all-in run-out
- **Utility calculation:** Pot splitting, fold equity, showdown comparison
- **Edge cases:** All-in before raise cap, min stack scenarios, tie pots
- **Raise cap variants:** 3-raise early / 4-raise late enforcement
- **Flop HE variant:** 2-street game terminates after flop
- **Full hand playthrough:** Complete hands from preflop to showdown, ensure correct actions available in each betting round and street. 
- **Info set key encoding:** Consistent keys for equivalent states
- **Encoder correctness:** Card canonicalization, bet history encoding
