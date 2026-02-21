# Blueprint + Subgame Solving Architecture

**Status:** Reference / Discussion
**Date:** 2026-02-12
**Context:** Overview of how modern poker solvers (Libratus, Pluribus, PioSolver) use a two-phase blueprint + subgame solving design.

## The fundamental problem

HUNL poker has ~10^161 information sets. Even with abstraction, solving the full game at high precision is impractical. The key insight behind modern solvers is: **you don't need a precise strategy everywhere — only where you actually play.**

## Two-phase architecture

```
OFFLINE (hours to weeks)                    ONLINE (seconds per decision)
========================                    ============================

Full game + coarse abstraction              Observe opponent action + board
         |                                           |
    MCCFR (billions of iters)               Identify subgame root
         |                                           |
    Blueprint Strategy                      Solve subgame with:
    (imprecise but complete)                  - finer abstraction
                                              - blueprint leaf values
                                              - opponent reach probs
                                                     |
                                              Play refined strategy
```

## Phase 1: Blueprint (offline)

The blueprint is a strategy for the entire game, solved with coarse abstraction to make it tractable.

### Abstraction dimensions

- **Hand abstraction** — bucket similar hands together (e.g., EHS2 clusters, or hand classes). Instead of tracking every specific hand, you might have 200 buckets per street.
- **Action abstraction** — limit bet sizes to a few options (e.g., 0.5x, 1x, 2x pot) rather than every possible chip amount.
- **Board abstraction** — group similar boards (suit isomorphism, texture classes).

### Solving method

Typically MCCFR (external sampling) with DCFR discounting. Runs for millions to billions of iterations. Libratus ran for ~15 million core-hours. Our existing trainer does exactly this.

### What you get

A strategy table mapping `(info_set_key -> action_probabilities)` for every reachable game state in the abstracted game. This strategy is "good enough" everywhere but not precise anywhere — like a low-resolution map of the entire territory.

## Phase 2: Subgame solving (online)

When you actually play a hand, you know the specific board cards and action history. You can now solve just the relevant subtree at much higher precision.

**Example:** The flop comes Ah Kd 7c. Instead of using the blueprint's coarse strategy for "AK-high flop texture bucket," you solve this specific flop with:
- More hand buckets (or no bucketing at all for river subgames)
- More bet sizes (add 0.33x, 0.75x, 1.5x, overbet sizes)
- Exact equity calculations for this specific board

The subgame is orders of magnitude smaller than the full game, so you can solve it in seconds.

## The hard part: connecting subgames to the rest of the tree

You can't just solve a subgame in isolation. The critical constraint is: **what happens at the subgame boundary?**

There are two boundaries to handle:

### 1. The root (top) — opponent reaching probabilities

When you solve a flop subgame, you need to know: "what hands does my opponent arrive at this flop with, and with what probability?" This comes from the blueprint. The opponent's range at the subgame root constrains the solution.

### 2. The leaves (bottom) — continuation values

If you're solving just the flop action (not all the way to showdown), you need values for "what happens after the flop betting round ends?" These leaf values come from the blueprint strategy for the turn/river.

## Evolution of subgame solving techniques

### Unsafe subgame solving (early approach)

Just re-solve the subgame assuming the opponent plays according to the blueprint range. Problem: you might find an "improvement" that actually makes you exploitable — you shift probability away from some lines, and an adversary who knows this can exploit you.

### Safe subgame solving (Burch et al. 2014, refined by Brown & Sandholm 2017)

Ensures the refined strategy is provably no worse than the blueprint. The key idea: at the subgame root, give the opponent the option of "taking the blueprint value" as an alternative to entering the subgame. This means:

- For every opponent hand, the opponent can either play into your new subgame strategy, or "cash out" at the blueprint value
- Your new strategy must be at least as good against every opponent hand
- This is implemented as a modified CFR with "gadget" nodes at the root

### Depth-limited subgame solving (Brown & Sandholm 2018, used in Libratus)

Don't solve to the end of the game — solve just one or two streets ahead, using blueprint values as leaf estimates. This dramatically reduces subgame size:

```
Full subgame (flop to river):  ~10^9 info sets
Depth-limited (flop only):     ~10^5 info sets
```

The tradeoff: leaf value estimates from the blueprint introduce approximation error. But the finer abstraction within the solved portion more than compensates.

### Nested subgame solving (Libratus's real-time approach)

Re-solve at every decision point, not just once per street:

```
Flop dealt -> solve flop subgame -> play action
Opponent acts -> re-solve with updated beliefs -> play action
Turn dealt -> solve turn subgame -> play action
...
```

Each re-solve incorporates the actual actions taken, giving increasingly precise strategies as the hand progresses. Libratus did this at every decision point in real-time.

## How Pluribus handled multiplayer (6-max)

Pluribus (Brown & Sandholm 2019) extended this to 6 players with modifications:

- **Blueprint:** MCCFR with linear CFR weighting, ~8 days on 64 cores, ~12.5K core-hours (vastly less than Libratus, due to algorithmic improvements)
- **Real-time search:** depth-limited to 1-4 actions ahead (not full streets)
- **Opponent modeling:** assumed all opponents play the blueprint strategy during search (no safe subgame solving — theoretical guarantees don't exist for 3+ players anyway)
- **Key insight:** for multiplayer, you don't need safety guarantees. A strategy that does well against the blueprint-playing opponents is good enough in practice.

## Practical architecture for this codebase

Given what already exists, the architecture maps naturally:

```
What exists now                      What's needed
===============                      =============

HunlPostflop (Game trait)            Already works as blueprint game
MCCFR / SequenceCfr                  Blueprint training (done)
BlueprintStrategy                    Stores blueprint (done)
InfoKey encoding                     Used for lookup (done)
SubgameCache (stub)                  Needs real implementation

Missing pieces:
  1. Subgame tree builder — given a specific board + action history,
     build a finer-grained subtree
  2. Opponent reach calculator — walk the blueprint to compute
     P(opponent has hand X | actions so far)
  3. Leaf value estimator — query blueprint for continuation values
     at depth-limited leaves
  4. Gadget game construction — for safe subgame solving, add
     alternative-payoff nodes at the subgame root
  5. Real-time CFR loop — fast CFR+ on the subgame (seconds, not hours)
```

## What makes this fast enough for real-time

The numbers work out because subgames are tiny relative to the full game:

| Scope | Info sets | Solve time |
|-------|-----------|------------|
| Full HUNL game (abstracted) | ~10^8 | Hours-days |
| Flop subgame (one board) | ~10^5-10^6 | 1-10 seconds |
| Turn subgame (one board) | ~10^4-10^5 | <1 second |
| River subgame (one board) | ~10^3-10^4 | milliseconds |

The subgame solver only needs ~1000 CFR iterations to converge on these small trees, and each iteration is fast because the tree fits in L2/L3 cache.

## Key tradeoffs in implementation

### Blueprint precision vs. subgame quality

A better blueprint gives better leaf values and opponent ranges for subgame solving. But diminishing returns — going from 1M to 10M blueprint iterations matters; going from 100M to 1B matters less because subgame solving corrects errors.

### Subgame depth vs. solve time

Solving deeper (flop through river) gives more precise answers but takes longer. Depth-limited solving (one street) is faster but depends more on blueprint quality at the leaves.

### Abstraction in subgames

You can use finer abstraction in subgames (more buckets, more bet sizes) because the tree is smaller. The extreme case is "no abstraction" for river subgames — you can track every specific hand and use exact equity.

## Key references

- Burch, Johanson, Bowling (2014) — "Solving Imperfect Information Games Using Decomposition"
- Brown, Sandholm (2017) — "Safe and Nested Subgame Solving for Imperfect-Information Games"
- Brown, Sandholm (2018) — "Depth-Limited Solving for Imperfect-Information Games" (Libratus)
- Brown, Sandholm (2019) — "Superhuman AI for multiplayer poker" (Pluribus)
- Tammelin (2014) — CFR+
- Brown, Sandholm (2019) — "Solving Imperfect-Information Games via Discounted Regret Minimization" (DCFR)
