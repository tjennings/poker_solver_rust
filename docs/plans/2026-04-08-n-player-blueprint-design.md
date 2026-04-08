# N-Player Blueprint Trainer — Design Document

**Date:** 2026-04-08
**Status:** Approved
**Approach:** Clean-room module (`blueprint_mp`) alongside existing `blueprint_v2`

## Overview

Extend the blueprint MCCFR trainer from hardcoded 2-player to configurable 2–8 players. Built as a new `blueprint_mp` module within `crates/core/` using strong domain types, with the existing `blueprint_v2` as a reference implementation. Shares abstraction, CFR utilities, and hand evaluation with `blueprint_v2` but owns its own game tree, traversal, storage, config, and info set encoding.

## Scope

**In scope (this design):**
- `crates/core/src/blueprint_mp/` — new module
- `crates/trainer/` — CLI integration for the new module
- 2–8 player support with configurable blind/ante/straddle structure

**Out of scope (future work):**
- Range solver N-player support
- cfvnet N-player support
- Tauri explorer N-player support (next phase)

## Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Player count | 2–8, `MAX_PLAYERS = 8` | Full ring support |
| Bucketing | Player-agnostic, shared | Matches Pluribus; equity vs 1 random hand preserves relative ordering |
| Blind structure | Fully configurable per-seat | SB, BB, ante, BB-ante, straddle via config list |
| Side pots | Full resolution | Correctness requirement; approximations produce wrong payoffs |
| Info key | 128-bit fixed | 22 action slots, Copy type, fast hashing |
| Info key overflow | Panic | No silent degradation; expand key width if real configs hit the limit |
| Storage | Pre-allocated | Keep it simple; lazy allocation is a future optimization |
| Exploitability | Per-player best-response for all N | Expensive diagnostic, not a training signal |
| Action abstraction | Shared per-street, lead/raise split | Lead sizes separate from raise depth sizes |
| Strategy averaging | Simple (Pluribus-style) | Biased for N>2 but empirically sufficient |

## Module Structure

```
crates/core/src/blueprint_mp/
├── mod.rs              # Public API
├── types.rs            # Domain types (Seat, Chips, PlayerSet, Street, Bucket, etc.)
├── config.rs           # BlueprintMpConfig, TableConfig, BlindConfig
├── game_tree.rs        # N-player game tree builder
├── info_key.rs         # 128-bit InfoKey
├── mccfr.rs            # External-sampling MCCFR traversal
├── terminal.rs         # Side pot resolution, showdown, fold payoffs
├── trainer.rs          # Training loop (deal sampling, iteration cycling)
├── storage.rs          # Strategy/regret storage (may share with blueprint_v2)
└── exploitability.rs   # Per-player best-response diagnostic
```

### Shared with `blueprint_v2`

- `abstraction/` — clustering pipeline, equity computation
- `cfr/dcfr.rs` — regret matching, DCFR/LCFR discount factors
- `hand_eval` — hand ranking
- `Card`, `Action` types

## Domain Types

### Core Newtypes

```rust
/// Player seat index (0-based). Prevents mixing with bucket/action indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Seat(u8); // 0..num_players

/// Chip amount. Newtype over f64 for type safety.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Chips(pub f64);

/// Bucket index for card abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bucket(pub u16);

/// Street enum (replaces raw u8 indexing).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Street { Preflop, Flop, Turn, River }

/// Compact bitfield for player sets. Supports up to 8 players.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlayerSet(u8);

pub const MAX_PLAYERS: usize = 8;
```

### PlayerSet Operations

```rust
impl PlayerSet {
    pub fn contains(self, seat: Seat) -> bool;
    pub fn insert(&mut self, seat: Seat);
    pub fn remove(&mut self, seat: Seat);
    pub fn count(self) -> u8;
    pub fn iter(self) -> impl Iterator<Item = Seat>;
    pub fn next_after(self, seat: Seat) -> Option<Seat>; // clockwise
}
```

### Deal

```rust
pub struct Deal {
    pub hole_cards: [[Card; 2]; MAX_PLAYERS],
    pub board: [Card; 5],
    pub num_players: u8,
}

pub struct DealWithBuckets {
    pub deal: Deal,
    pub buckets: [[Bucket; 4]; MAX_PLAYERS], // per-seat, per-street
}
```

## Game Tree

### Node Types

```rust
pub enum GameNode {
    Decision {
        seat: Seat,
        actions: Actions,
        children: Vec<usize>,  // indices into node array
    },
    Chance {
        street: Street,
        child: usize,
    },
    Terminal {
        kind: TerminalKind,
    },
}

pub enum TerminalKind {
    /// All but one player folded.
    LastStanding { winner: Seat },
    /// Two or more players reach showdown.
    Showdown { pots: Vec<SidePot> },
}

pub struct SidePot {
    pub amount: Chips,
    pub eligible: PlayerSet,
}
```

### Build State

```rust
pub struct BuildState {
    pub stacks: [Chips; MAX_PLAYERS],
    pub street_bets: [Chips; MAX_PLAYERS],
    pub to_act: Seat,
    pub active: PlayerSet,                // hasn't folded
    pub all_in: PlayerSet,                // is all-in
    pub acted_since_aggression: PlayerSet, // for closing-action detection
    pub street: Street,
    pub pot: Chips,
    pub num_players: u8,
    pub raise_count: u8,                  // for raise cap enforcement
}
```

### Action Sequencing

- **Preflop**: Action starts left of BB (or straddle). Blinds/straddle act last in first orbit. Proceeds clockwise, skipping folded and all-in players.
- **Postflop**: Starts at first active player left of BTN. Proceeds clockwise.
- **Closing condition**: Round closes when all active non-all-in players have acted since the last aggression AND all have equal investment or are all-in.

### Fold Continuation

```
Fold action → remove seat from active set
  If active.count() == 1 → Terminal::LastStanding
  If active.count() > 1  → continue to next seat's decision
```

A fold is NOT a terminal leaf unless it creates a last-standing situation.

## Terminal Payoffs

### Side Pot Resolution

```rust
pub fn resolve_showdown(
    contributions: &[Chips; MAX_PLAYERS],
    hand_ranks: &[HandRank; MAX_PLAYERS],
    active: PlayerSet,
    num_players: u8,
    rake_rate: f64,
    rake_cap: Chips,
) -> [Chips; MAX_PLAYERS]  // net payoff per seat
```

Algorithm:
1. Sort active players by total contribution (ascending)
2. Build pots layer by layer:
   - First pot: `min_contribution × num_eligible`
   - Each subsequent: `(next_contribution - prev) × remaining_eligible`
   - Final pot: remaining bets among non-all-in players
3. Apply rake to each pot (capped at `rake_cap` total)
4. Award each pot to the best hand among its eligible players
5. Return net payoffs (winnings minus contributions)

### Fold Terminal

Winner receives sum of all contributions. No hand evaluation, no side pots.

### MCCFR Terminal Value

```rust
fn terminal_value(&self, node: &GameNode, deal: &DealWithBuckets, traverser: Seat) -> f64
```

Returns the traverser's net payoff from the resolved terminal.

## 128-bit InfoKey

### Layout

```
Bits 127-125: seat position       (3 bits, 0-7)
Bits 124-97:  hand/bucket         (28 bits, up to 268M buckets)
Bits 96-95:   street              (2 bits, 0-3)
Bits 94-90:   spr_bucket          (5 bits, 0-31)
Bits 89-0:    action history      (90 bits = 22 slots × 4 bits each)
```

Action slot encoding (4 bits): fold=0, check=1, call=2, bet_size_0..N=3+, empty=0xF.

### Overflow

Panic if action history exceeds 22 slots. No silent truncation or fallback.

## Action Abstraction Config

### Lead / Raise Split

```yaml
action_abstraction:
  preflop:
    lead: ["5bb", "6bb"]
    raise:
      - ["3.0x"]          # first raise
      - ["2.5x"]          # second raise+
  flop:
    lead: [0.33, 0.67, 1.0]
    raise:
      - [0.5, 1.0, 2.0]
      - [0.75, 1.5]
  turn:
    lead: [0.5, 1.0]
    raise:
      - [0.67, 1.0]
  river:
    lead: [0.5, 1.0]
    raise:
      - [1.0]
```

Lead is a flat list (one depth). Raise is depth-indexed; last entry repeats for further depths.

## Game Config

```yaml
game:
  name: "6-max 100bb ante"
  num_players: 6
  stack_depth: 200
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2
    - seat: 1
      type: bb_ante
      amount: 2
  rake_rate: 0.0
  rake_cap: 0.0
```

Supported blind types: `small_blind`, `big_blind`, `ante`, `bb_ante`, `straddle`.

## MCCFR Traversal

### External Sampling (Pluribus-style)

```rust
fn traverse_external(&self, node_idx: usize, traverser: Seat, deal: &DealWithBuckets) -> f64 {
    match &self.tree[node_idx] {
        GameNode::Decision { seat, actions, .. } => {
            if *seat == traverser {
                // Explore ALL actions, accumulate regrets
            } else {
                // Sample ONE action from opponent's current strategy
            }
        }
        GameNode::Chance { .. } => {
            // Sample one board card runout
        }
        GameNode::Terminal { .. } => {
            self.terminal_value(node_idx, deal, traverser)
        }
    }
}
```

### Training Loop

```rust
for meta_iteration in 0.. {
    for traverser in 0..num_players {
        let deal = sample_deal(num_players, &mut rng);
        let buckets = compute_buckets(&deal, &abstraction);
        self.traverse_external(root, Seat(traverser), &deal_with_buckets);
    }
    // DCFR discount, pruning checks, snapshot saves
    // All keyed off meta_iterations, not individual traversals
}
```

### Strategy Averaging

Simple approach (Pluribus): update average strategy at opponent nodes during the traverser's pass. Biased for N>2 but empirically sufficient and much cheaper than unbiased alternative.

### Deal Sampling

```rust
fn sample_deal(num_players: u8, rng: &mut impl Rng) -> Deal {
    // Shuffle deck, deal 2 cards per player + 5 board
    // Needs 2*num_players + 5 cards (max 21 for 8 players)
}
```

## Exploitability (Diagnostic)

Per-player best-response computation. For each seat, compute BR value assuming all other seats play current average strategy. Sum = total exploitability upper bound.

- Expensive: N full-tree traversals per measurement
- Intended for final validation, not training loop
- Available for all player counts (including 2-player, where it matches the standard definition)

## Validation Strategy

### Cross-validation against `blueprint_v2`

Run both modules on identical 2-player configs. Verify:
- Same game tree structure (node count, action sets per node)
- Same bucket assignments (shared abstraction)
- Strategy frequencies converge to within tolerance after N iterations

### N-player validation

- **Unit tests**: side pot resolution, fold continuation, action sequencing, closing condition, deal sampling, info key encode/decode roundtrip
- **Invariant checks**: probabilities sum to 1.0, payoffs zero-sum across all seats, every chip accounted for
- **Convergence smoke test**: toy 3-player config, verify strategy delta decreases
- **Exploitability diagnostic**: small configs, verify BR values decrease over training
