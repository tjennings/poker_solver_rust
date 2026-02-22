# Preflop Solver: Modeling Postflop Impact

**Status:** Draft / Discussion
**Date:** 2026-02-20
**Problem:** The preflop-only solver with raw equity showdowns finds a limp-trap equilibrium instead of the raise-heavy GTO equilibrium. Suited bluff-raises (T2s, 94s, 93s) have no incentive to raise without postflop streets.

## Option 1: Static EQR Multipliers

Replace `eq * pot` at showdown terminals with `eqr(hand, position, pot_type) * eq * pot`.

Assign each of the 169 canonical hands a position-dependent multiplier that adjusts raw equity toward realized equity. Suited connectors get EQR > 1.0 (implied odds, nut potential), offsuit junk gets EQR < 1.0, IP hands get a bonus over OOP.

**Calibration sources:**
- Extract EQR from existing postflop MCCFR blueprint (compare EV to equity x pot for each hand class)
- Published EQR tables from GTO Wizard or academic references
- Run representative postflop sims to fit hand-class multipliers

**Pros:** Trivial to implement — one change in `terminal_value()`. Fast iteration.
**Cons:** Static — doesn't adapt to pot size, SPR, or the specific preflop action line. Circular calibration problem: you need a good postflop solution to derive the multipliers, but you're trying to avoid building one.

**References:** Matthew Janda, *Applications of No-Limit Hold'em* (2013). Standard industry approach pre-2020.

## Option 2: Abstracted Postflop Tree (Industry Standard)

What Simple Preflop Holdem and HRC v3 do. Instead of showdown at preflop terminals, continue into a bucketed postflop game tree.

**How it works:**
1. At each preflop showdown terminal, attach a simplified 3-street postflop subtree
2. Bucket hands into clusters (e.g., 10K flop / 10K turn / 10K river — Simple Preflop found this sufficient)
3. Bucket boards into texture classes (dry/wet/paired, ~50-200 canonical flops)
4. Use just 2 postflop bet sizes per street (Simple Preflop found this sufficient for preflop accuracy)
5. CFR solves preflop + postflop jointly, but only save the preflop strategy

Key insight from Simple Preflop: "The postflop strategy is not saved due to abstractions — the detailed postflop strategy will be inaccurate, but at the aggregated level it allows you to get an accurate preflop solution."

**Pros:** The principled version of EQR — the solver discovers realization endogenously. Industry gold standard for preflop-focused solvers.
**Cons:** Significantly more computation (10-100x slower). Requires hand bucketing infrastructure (`HandClassV2` is a start but need finer-grained buckets). Memory goes from trivial to moderate.

**References:**
- HRC v3: "imperfect recall abstractions" with 64-16K buckets
- Simple Preflop: 10K/10K/10K buckets with 2 postflop sizes
- Waugh et al. (2009) "A Practical Use of Imperfect Recall"
- Johanson et al. (2013) "Measuring the Size of Large No-Limit Poker Games"

## Option 3: Learned Counterfactual Value Network (DeepStack Approach)

Train a neural network to predict `V(ranges, pot_size, board, position) -> cfv[169]` and use it as the terminal value function.

**How it works:**
1. Generate training data: solve many postflop subgames with MCCFR for various (range, pot_size, board) inputs
2. Train a feedforward network mapping `(range_p1[169], range_p2[169], pot_size, position)` -> `cfv[169]` per player
3. Replace `terminal_value()` with a network forward pass
4. Solve preflop CFR using network outputs as leaf values

This is exactly what DeepStack does: two counterfactual value networks (flop and turn), plus an auxiliary network for end-of-preflop values.

**Pros:** Captures complex interactions (position, SPR, range asymmetry) that static EQR misses. Once trained, inference is fast.
**Cons:** Requires generating training data from postflop solves (thousands of subgames). Neural network training infrastructure. Black-box — hard to debug.

**References:**
- Moravcik et al. (2017) "DeepStack" — Science
- Brown & Sandholm (2018) "Depth-Limited Solving" — NeurIPS
- Lockhart et al. (2022) "Value Functions for Depth-Limited Solving"

## Option 4: Blueprint Rollout Values

Use the existing MCCFR postflop blueprint to estimate terminal values via Monte Carlo rollouts.

**How it works:**
1. At each preflop showdown terminal, instead of `eq * pot`, sample N random boards
2. For each board, play out the hand using the blueprint strategy
3. Average the resulting payoffs to get `E[payoff | hand_i, hand_j, pot_size, position, blueprint_policy]`
4. Cache these values in a 169x169xK table (K = number of distinct pot/position contexts)

**Pros:** Uses infrastructure already built. No bucketing or neural nets needed. Values grounded in actual postflop play.
**Cons:** Noisy (needs many rollouts). Blueprint must be decent for values to be meaningful — chicken-and-egg. Slow to precompute (169x169 x ~100 boards x rollouts).

**References:**
- Lanctot et al. (2009) "Monte Carlo Sampling for Regret Minimization in Extensive Games"
- Related to warm start in Brown & Sandholm (2019) Pluribus supplementary

## Recommendation

**Option 2 (Abstracted Postflop Tree)** is the clear winner for this codebase:

1. Existing hand classification infrastructure (`HandClassV2` with 19 classes + strength + equity bins)
2. Working MCCFR solver for postflop
3. Simple Preflop proved coarse postflop abstractions (2 bet sizes, ~10K buckets) produce accurate preflop strategies
4. Principled — solver discovers EQR endogenously instead of guessing multipliers
5. Avoids neural network training complexity

**Implementation path:**
- Use 19 hand classes (or finer ~50-200 bucket version) as hand abstraction
- Use ~50-100 canonical flop textures as board abstraction
- Attach a 3-street postflop tree with 2 sizes per street at each preflop call/all-in terminal
- Run joint preflop+postflop CFR, extract only the preflop strategy

## Sources

- [GTO Wizard: Equity Realization](https://blog.gtowizard.com/equity-realization/)
- [HRC v3: Enhanced Preflop Solver with Postflop Modeling](https://www.holdemresources.net/blog/2023-hrc-v3-release/)
- [Simple Preflop Holdem](https://simplepoker.com/en/Solutions/Simple_Preflop_Holdem)
- [DeepStack (Moravcik et al., 2017)](https://arxiv.org/abs/1701.01724)
- [Depth-Limited Solving (Brown & Sandholm, 2018)](https://arxiv.org/pdf/1805.08195)
- [Value Functions for Depth-Limited Solving (Lockhart et al., 2022)](https://arxiv.org/abs/1906.06412)
- [Approximating GTO Strategies for Full-scale Poker (U of Alberta)](https://poker.cs.ualberta.ca/publications/IJCAI03.pdf)
- [Pluribus Supplementary Materials](https://noambrown.github.io/papers/19-Science-Superhuman_Supp.pdf)
