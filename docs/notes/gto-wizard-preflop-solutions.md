# How GTO Wizard Generates Preflop Solutions

*Research notes — 2026-03-29*

## Two Distinct Systems

GTO Wizard operates two separate preflop solving pipelines. Conflating them is a common source of confusion in forum discussions.

### 1. Pre-solved Library (MonkerSolver)

The original solution library uses **MonkerSolver**, a CFR-based solver that traverses the full preflop-through-river game tree with postflop card abstraction (bucketing).

- **Bucketing**: 30/30/30 equity-based buckets per street (flop/turn/river). All hands within a bucket are forced to play the same strategy.
- **Postflop role**: The postflop tree is solved but the strategies are too coarse to save. Its purpose is to provide accurate aggregate EV information that shapes preflop decisions — the standard approach used by all traditional preflop solvers.
- **Multi-way**: True simultaneous multi-player equilibrium finding across all positions. Not decomposed into pairwise matchups. This is why MonkerSolver requires enormous RAM.
- **Convergence**: Settings were ">100 I/N" (iterations per node). MonkerSolver docs recommend 10x the node count as minimum. GTO Wizard targets 0.1–0.3% of the starting pot Nash distance.
- **Rake**: Modeled using PokerStars Zoom structures (e.g., NL500: 5% capped at 0.6 BB; NL50: 5% capped at 4 BB).
- **Bet tree**: Fixed, hand-curated per pot type:
  - SRP flop: 2 donk sizes (27%, 72%), 3 cbet sizes (~25–30%, ~70–80%, ~140–170% overbet)
  - 3-bet pots flop: 3 sizes (20%, 56%, 122%)
  - 4-bet pots flop: 4 sizes (13%, 38%, 67%, all-in)
  - Complexity tiers: Simple (3–6 sizes), Advanced (4–8), Complex (12–19)

### 2. GTO Wizard AI (Proprietary, Neural Network)

The newer system uses **depth-limited CFR with neural network leaf values** — architecturally identical to Brown & Sandholm's depth-limited solving paradigm (NeurIPS 2018).

- **Street-by-street solving**: At each street, a CFR-like solver runs to equilibrium, using a neural network to estimate remaining-game EV at street boundaries instead of expanding through future streets.
- **No card abstraction**: Each hand combo gets its own strategy. Feasible because NN leaf values eliminate the need to traverse the full postflop tree, keeping per-street trees small enough.
- **Neural network training**: Trained through "massive self-play over billions of hands with various stacks, blinds, rake, and ante structures." The NN generalizes across configurations rather than requiring separate training per scenario.
- **Two modes**:
  - **Classic**: Solves street-by-street with NN leaf values at street ends. Better for nodelocking and exploitative analysis.
  - **Fast**: Smaller lookaheads with more frequent re-solving (closer to nested subgame solving). Optimized for large/complex trees.
- **Multi-way**: Supports up to 9 players preflop. Restricts postflop continuation to at most 3 players — when more reach the flop, heuristic rules select which continue (prioritizing players closing the action or with most invested). Adding a third player increases complexity by ~1000x.
- **Bunching effect**: Handles card removal from folded hands in multi-way scenarios.
- **Convergence benchmarks**:
  - River: exactly 0.1% Nash distance (solved exactly, no NN)
  - Turn: average 0.24% EV loss, 90th percentile 0.33% (benchmarked against PioSOLVER on 175 spots)
  - Flop: average 0.12% exploitability (improved from 0.165% after QRE introduction)
  - Multi-way preflop: true Nash distance is intractable; validated via comparison to extended-runtime internal references
- **Dynamic sizing**: An ML model evaluates the marginal value of each candidate bet size by measuring frequency, EV, and "removal regret." Iteratively eliminates the size that adds the least value. Users specify how many sizes they want; the algorithm selects which ones.

## Comparison with Other Solvers

| Solver | Postflop Method | Buckets per Street | Card Abstraction |
|---|---|---|---|
| MonkerSolver | Full tree, bucketed | 30 | Yes |
| Simple Preflop Holdem | Full tree, bucketed | Up to 10k | Yes |
| PioSOLVER Edge | Weighted flop subset (25–362 representative flops), exact postflop | N/A | No |
| HRC v3 | Abstracted postflop, equity-distribution-aware | 64–16k | Yes |
| GTO Wizard AI | NN leaf values at street boundaries | N/A | No |

### PioSOLVER Edge Approach (Notable)

Solves preflop by solving the full postflop game across a weighted **flop subset** (25–362 representative flops selected via evolutionary algorithm). Each flop is solved independently with full postflop CFR. Preflop EVs are the weighted average across all flops. Most computationally expensive of all approaches, but no postflop abstraction error.

### Simple Preflop Holdem

SPH v2.0 uses up to 10k/10k/10k buckets per street — much finer granularity than MonkerSolver's 30/30/30. Their documentation states "a larger number of buckets practically does not affect the strategy" beyond 10k.

### HRC v3

Builds an abstracted postflop game tree with 64 to 16k buckets using equity-distribution-aware bucketing (considers not just hand equity but how equity is realized on future streets). References Johanson et al. 2013.

## Quantal Response Equilibrium (QRE)

GTO Wizard introduced QRE as an alternative to Nash equilibrium for their solutions. QRE models bounded rationality — players make better decisions more often but occasionally err, weighted by EV cost. This produces more "human-like" strategies that are less exploitable against imperfect opponents than pure Nash. Flop exploitability improved from 0.165% to 0.12% after QRE introduction.

## Implications for Our Solver

Our blueprint_v2 MCCFR solver is architecturally similar to MonkerSolver: full-tree CFR with hand abstraction (HandClassV2 or EHS2 buckets). Key observations:

1. **Bucket granularity gap**: MonkerSolver uses 30 buckets/street; SPH found 10k optimal. Our HandClassV2 (19 classes) is much coarser than either. EHS2 and PotentialAwareEmd modes are better abstractions, but bucket count may still be limiting.

2. **Natural evolution path**: What GTO Wizard AI did — train NN leaf evaluators and use them for depth-limited preflop solving with no card abstraction. Our `cfvnet` pipeline already trains a river value network. The next step would be training flop and turn value networks, then running preflop CFR with NN leaf values instead of traversing the full postflop tree.

3. **Dynamic sizing**: The "Dynamic Bet Sizing Model" project aligns with GTO Wizard's approach. Their method: solve with a large candidate set, use an ML model to evaluate marginal value of each size, prune iteratively.

## References

### GTO Wizard Blog Posts

- [All you need to know about our solutions](https://blog.gtowizard.com/all-you-need-to-know-about-our-solutions/) — Solution methodology overview, MonkerSolver settings, bet tree structures
- [Introducing GTO Wizard AI Heads Up Preflop Solver](https://blog.gtowizard.com/introducing-gto-wizard-ai-heads-up-preflop-solver/) — AI system for HU preflop
- [Introducing Multiway Preflop Solving](https://blog.gtowizard.com/introducing-multiway-preflop-solving/) — Multi-way extension, bunching effect, 3-player postflop limit
- [How Solvers Work](https://blog.gtowizard.com/how-solvers-work/) — General CFR explainer
- [GTO Wizard AI Explained](https://blog.gtowizard.com/gto-wizard-ai-explained/) — Depth-limited solving with NN leaf values, street-by-street architecture
- [GTO Wizard AI Custom Multiway Solving](https://blog.gtowizard.com/gto-wizard-ai-custom-multiway-solving/) — Custom multi-way solve configuration
- [GTO Wizard AI Benchmarks](https://blog.gtowizard.com/gto-wizard-ai-benchmarks/) — Nash distance and EV loss benchmarks per street
- [GTO Wizard AI 3-way Benchmarks](https://blog.gtowizard.com/gto_wizard_ai_3_way_benchmarks/) — Three-way accuracy measurements
- [Introducing Quantal Response Equilibrium](https://blog.gtowizard.com/introducing-quantal-response-equilibrium-the-next-evolution-of-gto/) — QRE methodology and exploitability improvements
- [Dynamic Sizing: A GTO Breakthrough](https://blog.gtowizard.com/dynamic-sizing-a-gto-breakthrough/) — ML-based bet size selection and pruning
- [Poker Subsets and Abstractions](https://blog.gtowizard.com/poker-subsets-and-abstractions/) — Card abstraction and bucketing discussion
- [Status and info about our solutions](https://blog.gtowizard.com/status-and-info-about-our-solutions/) — Rake structures, convergence targets

### Academic Papers

- Brown & Sandholm, "Depth-Limited Solving for Imperfect-Information Games" (NeurIPS 2018) — Theoretical foundation for NN leaf value approach
- Johanson et al., "Evaluating State-Space Abstractions in Extensive-Form Games" (AAMAS 2013) — Abstraction methodology referenced by HRC v3
- [GTO Wizard Benchmark (arXiv 2603.23660)](https://arxiv.org/abs/2603.23660) — Formal benchmark paper

### Other Solver Documentation

- [PioSOLVER Blog: Flop Subsets](https://piosolver.com/blog/2015-11-05-flop-subsets/) — Evolutionary algorithm for representative flop selection
- [Simple Poker: Flop Subsets and Bucketing Explainer](https://simplepoker.com/en/News/What_is_flops_subset__How_it_is_used_in_GTO_Solvers__What_is_bucketing__72) — Comparison of subset vs bucketing approaches
- [Simple Preflop Holdem](https://simplepoker.com/en/Solutions/Simple_Preflop_Holdem) — SPH v2.0 bucket count findings
- [HRC v3: Enhanced Preflop Solver with Postflop Modeling](https://www.holdemresources.net/blog/2023-hrc-v3-release/) — Equity-distribution-aware bucketing
- [MonkerWare Guide](https://monkerware.com/guide.html) — MonkerSolver convergence recommendations
- [GTO Wizard AI MIT Poker 2024 (GitHub)](https://github.com/gtowizard-ai/mitpoker-2024) — Open-source bot using GTO Wizard AI infrastructure
