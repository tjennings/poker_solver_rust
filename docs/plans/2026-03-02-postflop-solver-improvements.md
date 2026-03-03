# Postflop Solver Improvements — Reference Comparison

Date: 2026-03-02
Status: Research / Future Reference

## Context

Compared the external `postflop-solver` DCFR implementation (`/Users/ltj/Documents/code/postflop-solver/src/solver.rs`) with our exhaustive DCFR postflop solver. The external solver is a highly optimized, production-grade HUNL postflop solver. This document captures findings and actionable improvement suggestions.

## Background: Postflop Solver Goal

Our postflop solver exists to provide **accurate EV per hand matchup** back to the preflop solver — it is a terminal value function, not a real-time engine. This informs which optimizations are most relevant.

### Current Architecture Gap: Pot-Type Models

The highest-impact architectural change (independent of this comparison) is switching from a single SPR-keyed postflop model to **one model per (PotType, SPR) pair**:

| Pot Type | Typical SPRs | Position Structure |
|-|-|-|
| Limped | 3.5 | BB is IP |
| Raised | 2-3 | Raiser is IP |
| 3-Bet | 1-1.5 | 3-bettor is IP |
| 4-Bet+ | <1 | Often all-in |

This fixes the limped-pot fallback hack and AA/KK calling preference. ~8-12 models total, embarrassingly parallel.

## Architectural Comparison

| Dimension | External (postflop-solver) | Ours (postflop_exhaustive) |
|-|-|-|
| Data layout | All-hands-at-once: `f32[num_actions × num_hands]` per node | Per-hand-pair: DFS traverses one (hero, opp) pair at a time |
| Traversal | Recursive DFS, all hands simultaneously per node | Recursive DFS per hand pair, parallelized over pairs |
| Parallelism | At chance nodes (deal-outs) | Over 169×169 hand pairs via rayon pool |
| Precision | `f32` regrets/strategy (optional `i16` compression) | `f64` regrets/strategy |
| Tree | Pointer-based with `MutexLike<Node>`, offset arithmetic | Flat `Vec<PostflopNode>` enum with child index arrays |

**Key insight**: The external solver is optimized for **per-board solving with all 1326 hands simultaneously**. Ours is optimized for **abstract equity-table solving with 169 canonical hand pairs**. Fundamentally different use cases.

## Key Optimizations in External Solver We Lack

| Technique | What It Does | Impact |
|-|-|-|
| **Vectorized showdown eval** | Two-pass sorted-strength scan with `cfreach_minus[52]` card-exclusion. O(N) for all hands at once | Eliminates per-pair equity lookups — requires all-hands architecture |
| **Isomorphic chance folding** | `isomorphic_chances()` + swap lists — suit-isomorphic boards share computation | ~4x reduction at turn, ~12x at river |
| **i16 compression** | Regrets/strategies as `i16` with per-node scale factors | 4x memory reduction |
| **FMA slice operations** | `fma_slices_uninit` vectorized across all hands | Auto-SIMD — requires all-hands layout |
| **Inner product (8-wide unroll)** | Manual 8-element unrolling with `get_unchecked`, f64 accumulation | Hand-tuned showdown perf |
| **Custom stack allocator** | Per-recursion temp vectors on stack allocator | Reduces allocator pressure |
| **Node locking** | Freeze specific nodes to fixed strategies | Enables subgame re-solving |
| **Power-of-4 γ reset** | Periodically resets γ discount cycle at nearest lower power of 4 | Prevents over-commitment to early strategies |

## Optimizations We Have That They Don't

| Technique | What It Does |
|-|-|
| **Regret-based pruning (RBP)** | Skip negative-regret actions with stochastic exploration |
| **Pre-computed equity tables** | O(1) showdown lookup per hand pair |
| **Buffer pool reuse** | `parallel_traverse_pooled` — zero per-iteration allocation |
| **Atomic progress counters** | `SolverCounters` for TUI |
| **Multiple CFR variants** | Vanilla, DCFR, CFR+, Linear via `DcfrParams` |
| **Built-in exploitability** | Best-response computation |

## DCFR Discounting Comparison

| Aspect | External | Ours |
|-|-|-|
| α (positive regret) | `t^1.5 / (t^1.5 + 1)` | `t^α / (t^α + 1)`, configurable |
| β (negative regret) | Fixed 0.5 | `t^β / (t^β + 1)`, configurable |
| γ (strategy) | `(t/(t+1))^3` with power-of-4 reset | `(t/(t+1))^γ`, configurable |
| Power-of-4 γ reset | Yes | No |

## Actionable Suggestions (Priority Order)

### 1. Isomorphic Chance Folding — Medium effort, High impact
- [ ] Precompute suit-isomorphism swap lists for turn and river boards
- [ ] Skip redundant boards and apply swap to recover their contributions
- [ ] Orthogonal to current architecture — no structural change needed

### 2. f32 Precision for Regrets — Low effort, Medium impact
- [ ] Switch regret/strategy buffers from f64 to f32
- [ ] Keep f64 only for accumulation (sum operations)
- [ ] Halves memory, improves cache utilization

### 3. Power-of-4 γ Reset — Low effort, Low-Medium impact
- [ ] Add variant to `DcfrParams::strategy_discount()` that resets γ cycle at nearest lower power of 4
- [ ] May improve convergence speed by periodically reopening strategy learning window
- [ ] Simple to A/B test

### 4. Node Locking — Medium effort, Medium impact
- [ ] Add ability to freeze specific nodes to fixed strategies during solving
- [ ] Prerequisite for subgame re-solving architecture
- [ ] Enables: freeze preflop strategy, re-solve postflop subtrees

### 5. i16 Compression — Medium effort, Medium impact
- [ ] For production deployment / stored strategies, not training
- [ ] Per-node scale factors with i16 storage → 4x memory reduction
- [ ] Useful when serving many solved models simultaneously

### 6. All-Hands-At-Once Architecture — High effort, High impact
- [ ] Fundamental restructure to process all hands per node visit
- [ ] Enables vectorized showdown eval and SIMD slice ops
- [ ] Only worth it if moving to per-board solving (1326 hands) rather than abstract equity tables (169 hands)
- [ ] Would be needed if adopting a Libratus-style unified tree approach

## Pluribus/Libratus Research Context

### Unified Tree Approach
Libratus/Pluribus solve one unified tree from preflop through river. Our two-phase architecture (preflop LCFR + postflop EV oracle) is an approximation. Key properties:
- Preflop is the root of the unified tree, no separate solver
- 169 canonical hands preflop (lossless), bucketed postflop
- External-sampling MCCFR samples one board per traversal at each chance node
- Libratus: 15M core-hours (fine abstraction). Pluribus: 12.4K core-hours (coarse + re-solving)
- Preflop/flop use blueprint directly; real-time re-solving only at turn/river

### Convergence
- Neither system published iteration counts — only compute budgets
- Computing full-game exploitability is intractable at ~10^12 info sets
- CFR convergence: O(1/√T). For EV oracle, ~0.01 pot-fraction accuracy is sufficient

### Implication for Us
Our postflop model quality is the accuracy bottleneck. The pot-type-keyed model architecture (item at top) is the most impactful single change, independent of solver-level optimizations.
