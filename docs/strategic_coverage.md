# HUNL Strategic Elements vs. Solver Coverage

## Solver Pipeline

| Pipeline | Abstraction | Use |
|-|-|-|
| **Preflop LCFR + Postflop** | 169 canonical hands, direct indexing per flop, no clustering | Primary solver — preflop equilibrium with postflop EV backing |

---

## 1. Game Structure

| Element | Status | Notes |
|-|-|-|
| Heads-up (2 players) | Covered | Core game model |
| Position (IP vs OOP) | Covered | P1=SB/OOP acts first all postflop streets |
| Blind structure (SB/BB) | Covered | Configurable, SB=1, BB=2 internal units |
| Stack depth | Covered | Configurable `stack_depth` in BB |
| Antes/straddles | **Gap** | No ante support in preflop config. Matters for HU cash with antes or HU SNGs |
| Tournament/ICM | **Gap** | Cash game only — no ICM pressure modeling |

## 2. Preflop Strategy

| Element | Status | Notes |
|-|-|-|
| Open raise sizing | Covered | Configurable per depth (`raise_sizes`) |
| 3-bet/4-bet/5-bet | Covered | Up to `raise_cap` (default 4) |
| Limp strategy (SB) | Covered | SB can call preflop |
| Preflop all-in | Covered | Always available |
| 169 canonical hands | Covered | Suit isomorphism, direct indexing |
| Preflop equity calculation | Covered | Monte Carlo or uniform equity tables |
| Card removal weights | Covered | `compute_card_removal_weights()` builds 169x169 overlap matrix |
| **Multiple open-raise sizes** | **Gap** | Each raise depth has exactly one size. GTO solutions offer multiple open sizes (e.g., 2.5x and 3x) that the solver mixes between |
| Limp-raise (SB limp then re-raise) | Covered | Legal in tree if raise_cap allows it |

## 3. Postflop Bet Sizing & Action Tree

| Element | Status | Notes |
|-|-|-|
| Pot-fraction bets | Covered | Configurable list (e.g., [0.33, 0.5, 1.0]) |
| Overbets | Covered | Any size >1.0 in `bet_sizes` (e.g., 1.5, 2.0, 3.0) |
| All-in | Covered | Always available regardless of raise cap |
| Check-raise | Covered | Emerges naturally from action tree |
| Donk bet | Covered | OOP acts first, can bet into previous-street aggressor |
| Continuation bet | Covered | IP betting flop after preflop raise emerges from tree |
| Probe bet | Covered | OOP betting when IP checked previous street |
| Float play | Covered | IP calling to bet later streets |
| **Street-specific bet sizes** | **Gap** | Same `bet_sizes` on all streets. GTO solutions use different sizes per street (smaller flop, larger river) |
| **Separate raise sizing** | **Gap** | Bets and raises use same `bet_sizes` list. Raises should often differ (e.g., 3x the bet vs 0.75 pot) |
| **Geometric bet sizing** | **Gap** | No automatic geometric sizing to build toward all-in over remaining streets |
| Max 5 distinct sizes | Partial | Info key encoding supports 5 bet + 5 raise indices. Sufficient for most use but constrains very fine trees |

## 4. Hand Representation

| Element | Status | Notes |
|-|-|-|
| **169 canonical hands** | Covered | Each canonical hand is its own info set per flop — zero information loss from clustering |
| Kicker distinction | Covered | AKs vs AQs are separate info sets (different canonical hand index) |
| Suited vs offsuit | Covered | Distinct canonical hands (AKs ≠ AKo) |
| Pocket pair granularity | Covered | All 13 pairs are separate |

## 5. Board Texture

| Element | Status | Notes |
|-|-|-|
| Per-flop solving | Covered | Each of ~1,755 canonical flops solved independently — texture captured perfectly per flop |
| Suit isomorphism | Covered | Canonical flop enumeration collapses suits |
| Wet vs dry boards | Covered | Different flops produce different equilibria |
| Paired boards | Covered | Solved per flop |
| Monotone/two-tone/rainbow | Covered | Solved per flop |
| Dynamic boards (turn/river changes) | Covered | MCCFR backend samples concrete turns/rivers with real showdown eval |
| Flop metadata | Available | `CanonicalFlop` stores SuitTexture, RankTexture, Connectedness — used for weighting/analysis, not in info key |

## 6. Multi-Street Dynamics

| Element | Status | Notes |
|-|-|-|
| 3 postflop streets (F/T/R) | Covered | Full flop → turn → river progression |
| Progressive board reveal | Covered | `full_board` pre-dealt, revealed incrementally |
| SPR evolution across streets | Covered | `spr_bucket` recomputed each street |
| **Cross-street action history** | **Gap** | Info key encodes only *current street's* actions (6 slots). Previous streets' lines are lost. "Bet-called flop" looks identical to "checked through flop" on the turn. This is the single biggest abstraction loss |
| Street-to-street range narrowing | **Partial** | Emerges from CFR traversal paths but the info key can't condition on prior-street line |
| Delayed c-bet / turn c-bet | **Partial** | The action exists in the tree but the solver can't learn distinct turn strategy based on whether flop was checked through vs bet |
| River polarization | Covered | Emerges from equilibrium naturally |
| Multi-street barreling | **Partial** | Tree supports it but no cross-street line correlation in the key |

## 7. Card Removal / Blockers

| Element | Status | Notes |
|-|-|-|
| Deal-level card removal | Covered | Concrete deals never have card overlap |
| Preflop card removal weights | Covered | Explicit 169x169 weight matrix |
| Blocker-adjusted ranges (subgame CFR) | Covered | `precompute_opp_reach` skips overlapping combos |
| Blocker-adjusted ranges (MCCFR) | Covered (implicit) | Sampling concrete deals handles card removal implicitly |
| **Nut blocker bluffing** | Partial | Emerges somewhat from equilibrium. Each hand is a separate info set so blocker effects are captured per flop |
| Board blockers | Covered | Combo maps exclude board-blocked hands |

## 8. SPR & Stack Management

| Element | Status | Notes |
|-|-|-|
| SPR bucketing | Covered | 5-bit field, `min(eff_stack*2/pot, 31)`, half-SPR resolution |
| Multi-SPR models (preflop pipeline) | Covered | `postflop_sprs: [2, 6, 20]` builds independent models, closest selected at runtime |
| Dynamic SPR tracking | Covered | Recomputed at each new street |
| Short-stack play (low SPR) | Covered | At low SPR many sizes collapse to all-in |
| Deep-stack play (high SPR) | Covered | SPR capped at 31 (~15.5 SPR), adequate for standard depths |
| **SPR interpolation** | **Partial** | No interpolation between models — uses `ratio = actual_spr / model_spr` scaling within a model. Gap between discrete SPR models is a source of approximation error |

## 9. Solver Quality & Convergence

| Element | Status | Notes |
|-|-|-|
| Mixed strategies | Covered | Full probability distributions over actions |
| DCFR discounting | Covered | Configurable α/β/γ |
| Regret-based pruning | Covered | Configurable threshold and warmup |
| Preflop exploitability | Covered | Best-response computation |
| **Postflop exploitability** | **Gap** | No exploitability metric for postflop models — convergence measured by strategy delta only, which is a poor proxy |
| Convergence early stopping | Covered | Strategy delta and exploitability thresholds |
| Iteration warmup/discounting | Covered | First 30 iterations discounted by sqrt(T)/(sqrt(T)+1) |

## 10. Real-Time Play & Adaptation

| Element | Status | Notes |
|-|-|-|
| Blueprint strategy lookup | Covered | `BlueprintStrategy` hash map |
| **Real-time subgame solving** | **Gap** | `SubgameCfrSolver` and `SubgameTree` exist and are tested but not wired into the play loop. Falls back to blueprint |
| **Depth-limited search** | **Gap** | `SubgameConfig.depth_limit` exists but not enforced |
| **Opponent range construction at subgame root** | **Gap** | Missing piece for subgame solving — need to derive opponent's range from blueprint reach probabilities |
| **Safe subgame resolution** | **Gap** | No maxmargin or gift-giving at subgame boundaries (Libratus/Pluribus approach) |
| **Opponent modeling / exploitation** | **Gap** | Pure GTO only. No mechanism to adjust based on observed tendencies |
| **Node locking** | **Gap** | Can't fix one player's strategy and re-solve for the other |

## 11. Advanced Strategic Concepts

| Element | Status | Notes |
|-|-|-|
| Range construction | Covered | 169-hand preflop strategy matrix with mixed frequencies |
| Bluff-to-value ratio | Covered | Emerges from equilibrium |
| Minimum defense frequency | Covered | Solver's calling frequencies satisfy MDF at equilibrium |
| Polarization vs linear ranges | Covered | Emerges naturally, especially on river |
| Equity denial / protection bets | Covered | Emerges on draw-heavy boards |
| Pot control | Covered | Check actions available, solver learns when to control pot |
| Thin value betting | Covered | Each hand is its own info set — full granularity |
| **Multi-street bluff planning** | **Gap** | No cross-street history means the solver can't plan a bluff across streets (e.g., barrel turn only if you bluffed flop) |
| Bet sizing tells (exploitative) | N/A | GTO solver produces balanced sizing by design |
| Range advantage / nut advantage | Partial | Emerges from equilibrium but not an explicit feature |

---

## Ranked Gaps by Strategic Impact

### Tier 1 — Fundamental

**1. Cross-street action history**
The info key discards all prior-street actions. This prevents the solver from learning that your turn strategy should differ based on whether you bet the flop (and got called) versus checked through. In real HUNL, the line taken on earlier streets is arguably the most important information for constructing turn/river strategies.

*Suggestion:* Use the 5 reserved bits (positions 28-24) to encode a compressed prior-street summary — e.g., "was aggressor" / "was caller" / "checked through" per prior street. Even 2-3 bits would capture the most strategically relevant line distinctions.

**2. Real-time subgame solving**
Without this, the solver plays a static blueprint everywhere. Pluribus demonstrated that real-time search is what makes the difference between a good solver and a superhuman one. The infrastructure exists but needs wiring.

*Suggestion:* Wire `SubgameCfrSolver` into `SubgameSolver.solve()`. The missing integration pieces are: (a) construct opponent range at subgame root from blueprint reach, (b) provide continuation values at depth boundaries from blueprint, (c) enforce time budget.

### Tier 2 — Significant

**3. Street-specific bet sizing**
Using identical bet sizes across flop/turn/river is a meaningful simplification. Equilibrium solutions use smaller bets on the flop (information gathering, equity denial) and larger/more polarized bets on the river.

*Suggestion:* Extend `PostflopConfig.bet_sizes` to a per-street map: `{ flop: [0.33, 0.67], turn: [0.5, 1.0], river: [0.67, 1.0, 1.5] }`.

**4. Separate raise sizing**
Raises and bets have different strategic purposes. A raise is typically sized relative to the bet (2.5-3x), while a bet is sized relative to the pot. Sharing `bet_sizes` for both conflates these.

*Suggestion:* Add a `raise_sizes` field alongside `bet_sizes` in `PostflopConfig`.

**5. Postflop exploitability measurement**
Without it, you can't know how close the postflop solution is to equilibrium. Strategy delta ≠ exploitability.

*Suggestion:* Implement best-response computation for the postflop model analogous to the preflop exploitability calculator.

**6. Opponent modeling / exploitation**
Pure GTO leaves significant EV on the table against weak opponents. Even simple frequency-based adjustments (opponent folds too much → bluff more) would be valuable.

*Suggestion:* Start with node locking — fix one player's strategy from observed frequencies, re-solve for the other. This gives maximally exploitative play.

### Tier 3 — Nice-to-have

**7. Multiple preflop open sizes** — Offering 2+ open raise sizes to mix between.

**8. Geometric bet sizing** — Auto-generating sizes that build geometrically toward all-in.

**9. Antes/straddles** — Needed for ante games or certain HU formats.

**10. Safe subgame resolution** — Maxmargin approach ensures subgame strategies don't leak value at boundaries. Important for deep stack play where many subgames connect.
