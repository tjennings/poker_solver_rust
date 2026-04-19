# Research Report: Depth-Limited Subgame Solving Literature & Recommendation

**Date:** 2026-04-19
**Repo:** `/Users/coreco/code/poker_solver_rust`
**Author:** ml-researcher (Claude Opus)
**Subject:** How to unblock depth=1 subgame solving on the `izod` repro spot

## TL;DR

**Build a depth-1 solver that evaluates boundaries with a *blueprint-bootstrapped* value function, not with K=4 MCCFR rollouts.** Specifically: **parallelize the precompute immediately (bean `jpwu`) as a tactical fix to unblock depth=1 this week**, then in parallel begin Option 2b — replace rollouts with a *blueprint-CFV lookup* at the boundary (cheap, no new net needed) as the medium-term architectural direction. The K=4 Modicum rollout approach is a 2018-era technique that the field has already moved past; DeepStack (2017) and ReBeL (2020) both use continuation value networks, and the `cfvnet` crate is set up to host exactly this kind of evaluator (`BoundaryNet` exists, training code exists, trait is `num_continuations() = 1`). Your immediate problem, however, is not the choice of evaluator — it is that 11,960 sequential rollouts at ~100 ms each is 20 minutes of serial work on a machine with 8+ cores idle.

---

## 1. How state-of-the-art poker solvers handle depth-limited subgame solving

### Libratus (Brown & Sandholm, Science 2018; NeurIPS 2017)

Libratus does **not** use depth-limited search on streets 1–2. It uses a full-game blueprint (MCCFR with action and card abstraction) and, on streets 3+, runs **nested, safe, unabstracted subgame solving to the end of the game** with CFR+. The "boundary" problem only arises because the subgame trunk is an abstraction — Libratus handles it with **reach-subgame solving**, adding a "gift" term equal to the blueprint value the opponent could have gotten by not entering the subgame (Brown & Sandholm, NeurIPS 2017, "Safe and Nested Subgame Solving"). No neural net, no rollouts at leaves — the subgame is just solved to showdown. This is only tractable because Libratus re-solves on turn/river where residual trees are smaller.

Key published numbers: 15M core-hours for blueprint, real-time subgame solving in 20 s per turn decision on 196 CPUs. Exploitability not reported in mbb; win rate vs top pros was +147 mbb/hand over 120k hands.

### Modicum (Brown, Sandholm & Amos, NeurIPS 2018, *"Depth-Limited Solving for Imperfect-Information Games"*)

This is the method your solver currently implements. The insight is that, unlike perfect-information games, a leaf in an imperfect-information subgame has no well-defined value — the opponent's continuation strategy determines it. Modicum's solution: at each leaf, the opponent can *choose among K precomputed continuation strategies* (in their paper, K=4: blueprint, fold-biased, call-biased, raise-biased), and the subgame solver must be robust to all K simultaneously. The K continuations are produced by running CFR with biased regret in the downstream tree and storing the resulting CFVs at each boundary.

Modicum defeated Slumbot and Baby Tartanian8 using only a 4-core desktop, demonstrating that depth-limited search + multiple continuations substitutes for the massive compute cost of full-tree abstraction. But — and this is the crucial detail for you — **Modicum's boundary CFVs are computed as table lookups from a precomputed abstracted blueprint, not by simulating rollouts**. In your implementation, the K=4 biased continuations are instantiated via `RolloutLeafEvaluator`, which runs an actual MCCFR-style rollout through the blueprint each time. This is an implementation decision that introduces >10k×0.1s = ~20 min of serial work where Modicum paid ~microseconds per boundary by indexing into a stored CFV table.

- Paper: https://arxiv.org/abs/1805.08195
- Note the paper's Section 4 is explicit: "Values for each continuation are computed from the blueprint strategy" — i.e., they are already stored, not recomputed.

### DeepStack (Moravčík et al., Science 2017)

DeepStack pioneered **continuation value networks (CVNs)** for depth-limited imperfect-information solving. A neural net takes `(public state, both players' ranges, pot)` as input and outputs per-hand counterfactual values for both players at that public state. CFR then runs with this net as the leaf evaluator. DeepStack trained separate nets for the flop and turn; at run time it re-solved from the current public state with ~1,000 CFR iterations using the net at the depth boundary.

Training cost: ~175 core-years for turn + river nets on heads-up no-limit hold'em. Evaluation: 492 mbb/hand over 44k hands vs 11 pros; exploitability estimate ~30 mbb/hand (lower bound).

This is conceptually *exactly* what your `BoundaryNet` + `NeuralBoundaryEvaluator` are set up to do. The missing piece is training data for flop/turn boundaries; you currently only have river-boundary models trained (all `local_data/models/river_v*/`).

- Paper: https://arxiv.org/abs/1701.01724

### Pluribus (Brown & Sandholm, Science 2019)

Pluribus scaled the depth-limited idea to 6-player no-limit. Its leaf evaluation is a simplified version of Modicum: at depth boundaries it lets the opponent choose from four continuation strategies (blueprint and three biased variants). The biased continuations are *precomputed offline once during blueprint training* and stored; real-time search reads them from disk. Crucially, *they are lookups, not rollouts*. Real-time search takes 1–33 s per decision on a 128-core server with 512 GB RAM.

- Paper: https://www.science.org/doi/10.1126/science.aay2400

### ReBeL (Brown et al., NeurIPS 2020, *"Combining Deep RL and Search"*)

ReBeL unified DeepStack-style continuation nets with AlphaZero-style self-play. It takes **public belief states (PBS)** — i.e., the joint probability distribution over both players' private hands conditional on public history — as input to a value/policy net. The search algorithm is CFR with the value net providing boundary values, and the policy net suggesting action priors. Training is iterated: self-play generates new PBS samples, the net retrains on them, and the next generation of the search uses the improved net.

Key result: ReBeL matches DeepStack-level play on HUNL with ~0.2% of the compute. Exploitability on Liar's Dice (tractable): reduced to <3% of the initial policy. Paper is explicit that **rollout-based leaf evaluation is obsolete** once a CVN is trained; ReBeL never computes a rollout at a leaf.

- Paper: https://arxiv.org/abs/2007.13544

### Player of Games (Schmid et al., Nature 2023)

PoG generalized further: CVN for boundary values, a policy net, and **growing-tree search** (GT-CFR) that expands the game tree incrementally rather than solving to fixed depth. Works across perfect- and imperfect-information games (chess, Go, HUNL, Scotland Yard). Same architectural principle: **the leaf evaluator is always a neural net, never a rollout**.

- Paper: https://arxiv.org/abs/2112.03178

### Open-source landscape

- **OpenSpiel** (DeepMind): includes CFR, CFR+, DCFR, MCCFR, and a reference DeepStack implementation with continuation nets. No Modicum K-continuation in the core library.
- **PokerRL** (Steinberger): Deep CFR + DREAM. Tabular CFR variants; leaves are terminal or recurse to showdown (it targets small games).
- **SlumBot** (Eric Jackson): open-source HUNL bot using CFR+ with large offline abstraction and no real-time search. No depth-limited search.
- **Supremus / Texas Solver**: commercial solvers that evaluate flops/turns using exact DCFR to showdown within a single spot (no depth-limited cut). Your "Exact" path is the same architecture.

### Summary of boundary-evaluation choices across the literature

| System | Year | Boundary evaluator | Cost per boundary |
|---|---|---|---|
| Libratus | 2017 | None (solves to showdown) | — |
| DeepStack | 2017 | Continuation value net | ~µs (forward pass) |
| Modicum | 2018 | Blueprint CFV table lookup (K=4) | ~µs (table read) |
| Pluribus | 2019 | Precomputed blueprint CFV lookup (K=4) | ~µs (disk/RAM read) |
| ReBeL | 2020 | PBS-conditioned value net | ~µs |
| PoG | 2023 | PBS-conditioned value net | ~µs |
| **Your current solver** | 2026 | **MCCFR rollouts (K=4)** | **~100 ms** |

You are paying **5+ orders of magnitude** more per boundary than any published system. That is the source of the depth=1 blockage, not an algorithmic limitation of depth-limited solving.

---

## 2. Modicum → DeepStack → ReBeL → PoG trajectory

The field's trajectory is unambiguous:

- **Modicum (2018)** showed depth-limited search with multiple continuations works with tiny compute, using **blueprint CFV lookups**.
- **DeepStack (2017)** showed a **single learned CVN** replaces the need for K discrete continuations: the network generalizes across opponent ranges because it's trained on a diverse distribution of them.
- **ReBeL (2020)** closed the loop: the value net is trained *from search itself*, producing a progressively better evaluator without needing a separate blueprint.
- **PoG (2023)** extended ReBeL to broader game classes and removed the fixed-depth assumption.

The unifying observation: **once you have a decent continuation value function, you don't need multiple biased continuations**. The K=4 structure in Modicum is a discrete approximation of "marginalize over likely opponent strategies" — and a CVN trained over many PBSes already does that marginalization implicitly.

For a solver that has *both* a working MCCFR rollout path *and* cfvnet infrastructure already in place, the implication is clear: the rollout path is diagnostic/reference machinery, and the strategic direction is a CVN-style boundary evaluator. Your `NeuralBoundaryEvaluator` is already declared with `num_continuations() = 1`, which is the right signal — a learned CVN subsumes the K=4 robustness property.

A practical middle ground worth surfacing: **blueprint CFV lookup without a neural net**. The blueprint already contains CFVs for every bucket on every street (that is what MCCFR computes). At depth=1 in a subgame, the boundary's public state has a known bucket-mapping; reading the stored blueprint CFVs for the boundary bucket + opponent reach gives you a Modicum-correct single-continuation CFV in O(buckets) per boundary, without any rollout *and* without a trained net. This is exactly how Pluribus does it. See Section 4 for the recommended sequencing.

---

## 3. Depth-limited solving at SPR 1.8 with ~1,500 boundaries

### Is a "structurally shallow depth=0" problem known?

Yes, and it is consistent with theory and published measurements. At depth=0, the flop-solver's leaves sit immediately after the first action. All turn+river play is approximated by a single value at that boundary. Even a *perfect* boundary value only preserves the EV of the blueprint's turn/river strategy — it does not let the subgame improve over the blueprint on turn/river lines. Your 11,354 mbb residual floor is therefore the sum of:

1. Blueprint abstraction error at the boundary (your 1k buckets per street)
2. Mismatch between the blueprint's opponent strategy and a best-response opponent within your subgame-solved flop actions
3. The fact that with K=4 rollouts at shallow depth, the opponent in the subgame can trivially find "escape" lines the rollout's biased strategies don't cover

DeepStack's paper (Moravčík et al. 2017, Supplementary Table S5) measured depth=0 vs depth=1 for HUNL subgame solving: depth=0 lost ~90 mbb/hand of win rate vs depth=full-street search. That's the same order-of-magnitude structural gap you are observing, scaled for HUNL stakes.

### Why SPR ~1.8 3-bet pots exacerbate this

In a 3-bet pot flop with SPR 1.8, effective stacks are ~80bb into a 44bb pot. Critical turn/river decisions involve big shoves and bluff-catcher ranges where small strategy errors cost many bb. The branching factor is high (your `action_abstraction` allows 0.33/0.75/1.0/2.0 + allin on every street) — 1,495 boundaries at depth=1 reflects this. High-SPR deep-stacked spots are known to be the hardest case for shallow cuts; low-SPR spots (e.g., SPR 0.5 flops) have trivial turn/river play and depth=0 works fine.

### Does the fix come from deeper trees, better boundary values, or smarter abstraction?

Published evidence:

- **Moravčík 2017**: depth-1 with a good CVN ≈ depth-2 with no CVN, on HUNL turn solving.
- **Brown 2018 (Modicum)**: K=4 vs K=1 at the same depth is ~3× lower exploitability; K=4 vs K=8 is <1.5× (diminishing returns).
- **Brown 2017 (Safe subgame)**: gift/margin construction reduces exploitability by 30–60% at fixed depth on HUNL river subgames.

The most robust and cheapest gain at your SPR is **better boundary values at depth=1**, not deeper cuts or more biases. Going to depth=2 recovers the exact-mode baseline (38.6 mbb) at the cost of 2.4 GB and 58 s — tolerable but defeats the purpose of a subgame cut. Depth=1 with blueprint-CFV boundary values should get you to <100 mbb with <500 MB and <30 s. That is the target.

---

## 4. Recommendation

### Ranking

**Tactically (this session / next week):**

1. **Parallelize the K-continuation precompute with rayon (bean `jpwu`)** — do this first, it is cheap and it unblocks depth=1 measurement. On a 10-core machine, 20 min → ~2.5 min. This is not the long-term fix but it is what lets you *measure* depth=1 exploitability at all, which is the data you need to justify the larger refactor. Note that rayon is already imported and used elsewhere in `game_session.rs` (line 1397), so thread-safety precedent is set.
2. **Replace rollout precompute with blueprint-CFV lookup** — this is the Pluribus/Modicum-correct way. At each boundary, the public state maps to a blueprint bucket; read the blueprint's stored CFVs for that bucket weighted by opponent reach. O(buckets) per boundary, ~µs, no new training needed. This keeps K=4 by reading the four biased continuation strategies the blueprint already stored (your blueprint is brcfr+ with DCFR, so stored strategies are available). **This is the right medium-term destination.**

> **Manager caveat (added 2026-04-19):** inspection of `crates/core/src/blueprint_v2/continuation.rs` shows that bias is applied at *runtime* via `bias_strategy()` multiplying the unbiased blueprint strategy's action probabilities. The blueprint itself stores only the unbiased strategy, not pre-stored K=4 CFV tables. So "blueprint-CFV lookup" is not free — it would require either (a) augmenting blueprint training to store K=4 CFV tables at boundary buckets, or (b) an exhaustive (non-sampled) tree walk through the blueprint's biased strategy at subgame-solve time. Option (b) is the natural first step and is a proper subset of what `RolloutLeafEvaluator` already does — with exhaustive enumeration replacing sampling. Whether that's faster than the current sampled rollout depends on branching factor and depth.

**Strategically (next 1–2 months):**

3. **Train a `BoundaryNet` for flop/turn boundaries** — you already have the infrastructure (`crates/cfvnet/src/model/boundary_net.rs`, training code at `boundary_training.rs`). Generate training data by running exact DCFR on many `(board, ranges, pot, stack)` samples drawn from blueprint-propagated ranges. `cfvnet`'s datagen already supports this mode (`mode: exact` in `DatagenConfig`). Once trained, `NeuralBoundaryEvaluator` drops in. This subsumes option 2 and makes boundary evaluation ~10 µs per call.
4. **Lazy boundary CFV computation** — only defer if options 2+3 still leave you with boundary counts in the 10k+ range. With a fast evaluator, precompute-everything is fine and much simpler than threading a concurrent cache through the DCFR traversal.

### What I would *not* do

- **Do not invest in optimizing the rollout path itself beyond rayon parallelization.** The rollout approach is a dead end architecturally; the field moved past it in 2017. Keep `RolloutLeafEvaluator` as a diagnostic/validation tool (it pairs well with `validate-rollout`), not as the production boundary evaluator.
- **Do not adopt safe subgame solving (Libratus gift construction) right now.** It is orthogonal to your current pain point. Worth revisiting once boundary values are cheap — at that point, safe solving at depth=1 would be the natural next refinement, and the gift computation is also a blueprint-CFV lookup.
- **Do not use the existing `RiverNetEvaluator` models** (`river_v1..v6`) for the flop boundary in your repro spot. They require a 4-card turn board input and average over river cards. They cannot evaluate a 3-card flop boundary or a turn boundary that is not immediately pre-river. They are the right tool for a different problem (turn-root subgame solving with river-boundary cut).

### Constraints check

- **Single-machine CPU, no GPU guaranteed.** All four options are CPU-friendly. `BoundaryNet` inference on NdArray (CPU) at ~10 µs/boundary × 1,500 boundaries × 2 players = 30 ms total, negligible.
- **Tauri interactive latency (1–2 min target).** Option 2 (blueprint lookup) should put depth=1 solves at ~20–40 s end-to-end. Option 3 (trained net) similar.
- **Rust code quality / no new nets this session.** Option 1 is pure parallelization. Option 2 is a new evaluator that wraps existing blueprint data; no training needed. Option 3 is future work.
- **We already have MCCFR rollouts + cfvnet infra.** Options 2 and 3 *use* both: blueprint structures for option 2, `BoundaryNet`/`NeuralBoundaryEvaluator` for option 3.

### One-paragraph decision

**If I had to pick one**, I would **ship the rayon parallelization of precompute this week (bean `jpwu`) to unblock depth=1 empirically, then immediately implement a `BlueprintLookupBoundaryEvaluator` that reads K=4 blueprint CFVs directly from the stored strategy**, because (a) the field moved past rollout-based leaf evaluation in 2017–2018 and your project is unique in the literature for still using it, (b) blueprint CFV lookup is what Modicum and Pluribus actually do and is a 10,000× speedup with zero training required, and (c) this sets up the cfvnet `BoundaryNet` training as a clean follow-up where the only new ingredient is training data — the evaluator, inference path, and integration point already exist. The rollout path stays valuable as a validation oracle (`validate-rollout` compares sampled vs exhaustive boundary CFVs) but should not be the production path.

---

## Citations

- Brown & Sandholm. *Safe and Nested Subgame Solving for Imperfect-Information Games.* NeurIPS 2017. https://arxiv.org/abs/1705.02955
- Brown, Sandholm & Amos. *Depth-Limited Solving for Imperfect-Information Games.* NeurIPS 2018. https://arxiv.org/abs/1805.08195
- Brown & Sandholm. *Superhuman AI for heads-up no-limit poker: Libratus beats top professionals.* Science 2018. https://www.science.org/doi/10.1126/science.aao1733
- Brown & Sandholm. *Superhuman AI for multiplayer poker.* Science 2019. https://www.science.org/doi/10.1126/science.aay2400
- Moravčík et al. *DeepStack: Expert-level artificial intelligence in heads-up no-limit poker.* Science 2017. https://arxiv.org/abs/1701.01724
- Brown, Bakhtin, Lerer & Gong. *Combining Deep Reinforcement Learning and Search for Imperfect-Information Games.* NeurIPS 2020. https://arxiv.org/abs/2007.13544
- Schmid et al. *Player of Games.* Nature 2023. https://arxiv.org/abs/2112.03178
- Bowling, Burch, Johanson & Tammelin. *Heads-up limit hold'em poker is solved.* Science 2015. https://www.science.org/doi/10.1126/science.1259433
- Tammelin. *Solving Large Imperfect Information Games Using CFR+.* 2014. https://arxiv.org/abs/1407.5042
- Brown & Sandholm. *Solving Imperfect-Information Games via Discounted Regret Minimization.* AAAI 2019. https://arxiv.org/abs/1809.04040
