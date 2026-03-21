# ReBeL Analysis & Comparison

**Paper:** "Combining Deep Reinforcement Learning and Search for Imperfect-Information Games" (Brown, Bakhtin, Lerer, Gong — NeurIPS 2020)

## ReBeL Summary

ReBeL (Recursive Belief-based Learning) is an RL+Search framework that achieves superhuman HUNL performance with far less domain knowledge than Libratus or Pluribus.

### Core Idea

Instead of operating on the game tree directly, ReBeL converts the imperfect-information game into a perfect-information game over **Public Belief States (PBSs)**. A PBS is:
- The public game state (pot, board, action history)
- Plus a probability distribution over each player's possible private cards (beliefs)

This is a continuous-state perfect-information game, so techniques from AlphaZero (RL + search) become applicable.

### Algorithm (3 interlocking parts)

**1. Search (CFR-D in a depth-limited subgame):**
- At the current PBS, build a small depth-limited subgame (a few actions deep)
- Run T iterations of CFR-D to approximately solve it
- Leaf values come from the value network — not from rollouts or a blueprint
- Produces: infostate values at the root PBS, and average policies at every PBS in the subgame

**2. Value Network (v-hat):**
- Input: a PBS (beliefs + public state)
- Output: a vector of infostate values — one value per possible private hand for each player
- Trained via self-play: the values computed by search become training targets

**3. Self-Play Training Loop:**
```
while game not terminal:
    construct subgame at current PBS
    run CFR-D for T iterations (leaf values from value net)
    add (PBS, computed infostate values) to value net training data
    add (PBS, average policy) to policy net training data (optional)
    sample a leaf PBS -> make it the new root
    repeat
```

### Key Innovations

- **Safe search without extra constraints:** At test time, sampling a random CFR iteration and using that iteration's beliefs produces a Nash equilibrium in expectation. No need for the "safe subgame solving" gadgets Libratus used.
- **No blueprint needed:** The value network replaces the blueprint entirely. No tabular strategy storage, no hand abstraction, no pre-computed buckets.
- **Provable convergence:** With perfect function approximation, converges to Nash at rate O(1/sqrt(T)).

## Comparison: ReBeL vs Our Current Systems

| Dimension | Blueprint V2 (our MCCFR) | CFVnet (our depth-limited) | ReBeL |
|-----------|--------------------------|---------------------------|-------|
| **Training** | Tabular MCCFR over full game tree | Range-solver generates river training data | Self-play search generates training data at every game state |
| **Abstraction** | 500 buckets/street, lossy | None (exact 1326 combos at river) | None — value net generalizes across states |
| **Storage** | ~500M i32 regret slots | Neural net weights (~few MB) | Neural net weights (~few MB) |
| **Play-time search** | No search (just lookup) | Subgame re-solve at river only | Search at every decision point |
| **Leaf values** | N/A | CFVnet predicts river values | Value net predicts values at any depth |
| **Domain knowledge** | Heavy (hand abstraction, bet sizing grid, action abstraction) | Moderate (bet sizing grid, range-solver) | Minimal (game rules + a few bet sizes) |
| **Scalability** | Memory-bound (regret tables) | Compute-bound (training data gen) | Compute-bound (search during training) |
| **Theoretical guarantee** | Converges to Nash of the abstracted game | Exact Nash for individual spots | Converges to Nash of the full game |

### What ReBeL Does Better

1. **No abstraction loss.** Our blueprint groups 178M river entries into 500 buckets. ReBeL's value net operates on raw beliefs — no information is thrown away.

2. **Unified training + search.** We train offline (blueprint) then optionally search online (cfvnet subgames). ReBeL uses search during training, so the value net is trained on states it will actually encounter during play.

3. **Adapts to arbitrary bet sizes and stack depths.** Our blueprint is trained on a fixed action abstraction. ReBeL can handle any bet size at test time by running search.

4. **Smaller footprint.** A few MB of neural net weights vs hundreds of MB of regret tables.

### What ReBeL Costs

1. **Compute at play time.** Every decision requires running CFR for T iterations with neural net evaluations at every leaf. Our blueprint is a single table lookup.

2. **Training compute.** Self-play with search at every step is massively more expensive than tabular MCCFR. The paper used up to 128 machines with 8 GPUs each.

3. **Function approximation risk.** If the value net is inaccurate for some PBS, search produces bad strategies. Tabular CFR has no approximation error (for the abstracted game).

### How ReBeL Relates to What We Have

Our cfvnet is the closest piece to ReBeL — it trains a neural net to predict counterfactual values for river subgames. ReBeL generalizes this to all streets and integrates search into training. A ReBeL implementation would essentially:
- Replace the blueprint with a value network that works at all streets (not just river)
- Replace the MCCFR training loop with the self-play search loop
- Use CFR-D subgame solving (which we already have pieces of in subgame_cfr.rs) as the search algorithm
- Add a policy network for warm-starting search

### New Components Needed

1. **PBS representation** — public state + belief vectors (1326-dim per player)
2. **Value network architecture** — MLP over belief vectors (paper uses GeLU + LayerNorm)
3. **Self-play training orchestrator** — the outer loop that runs search, collects training data, trains nets
4. **CFR-D with neural leaf values** — modify existing subgame CFR to use value net at leaves instead of blueprint/equity
5. **Policy network** (optional) — warm-starts CFR iterations for faster convergence
