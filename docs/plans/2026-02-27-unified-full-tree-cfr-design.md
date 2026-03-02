# Unified Full-Tree CFR Solver

## Problem

The current solver separates preflop and postflop into independent phases:
1. Postflop models are pre-solved per flop, producing aggregate per-hand-pair EVs
2. The preflop LCFR solver consumes those EVs as terminal values

This separation means postflop equilibria are computed assuming uniform hand distributions — they can't condition on which hands actually reach each pot type. "AKo after 3-betting" gets the same postflop EV as "AKo after limping," even though these produce radically different postflop dynamics. This is the #1 strategic gap identified in the coverage analysis.

Iterative refinement (solve preflop → extract ranges → re-solve postflop → repeat) costs roughly the same as solving the full game, without the elegance.

## Design

### Core Idea

Replace the separated preflop/postflop architecture with a single exhaustive CFR solver that traverses the full game tree — preflop through river — for every deal. Regret-based pruning with DCFR exploration makes this tractable by eliminating irrelevant subtrees after a warmup period.

### Architecture

```
UnifiedSolver
├── Game tree: preflop decisions → postflop subtrees (per flop)
├── Regret buffers: preflop (persistent) + postflop (per-batch)
├── Strategy buffers: accumulated incrementally across batches
├── Pruning state: per-node regret threshold for subtree skipping
└── DCFR: discounting + periodic exploration of pruned branches
```

### Training Loop

```
for epoch in 0..num_epochs:
    for flop_batch in canonical_flops.chunks(batch_size):
        # Weight this batch by strategic frequency
        batch_weight = compute_batch_weight(preflop_strategy, flop_batch)

        # Build equity tables for this batch
        equity_tables = compute_equity_tables(flop_batch)

        # Allocate postflop regret/strategy buffers for this batch
        postflop_buffers = allocate(flop_batch, postflop_tree)

        for iteration in 0..iterations_per_batch:
            for (hero_hand, opp_hand) in all_hand_pairs:
                for flop in flop_batch:
                    # Skip if preflop regret for this hand reaching
                    # this flop's pot type is below prune threshold
                    if pruned(hero_hand, pot_type, prune_threshold):
                        if not exploration_iteration(iteration):
                            continue

                    traverse_full_tree(
                        preflop_root,
                        hero_hand, opp_hand,
                        flop, equity_tables[flop],
                        preflop_regrets,     # persistent
                        postflop_buffers,    # batch-local
                    )

        # Fold postflop strategy into aggregate
        merge_strategy(aggregate_strategy, postflop_buffers, batch_weight)

        # Free batch-local postflop buffers
        drop(postflop_buffers)
```

### Full-Tree Traversal

A single traversal starts at the preflop root and continues through postflop:

```
cfr_traverse(node, hero_hand, opp_hand, flop, equity_table, ...):
    match node:
        PreflopDecision:
            # Standard CFR: regret matching → traverse children
            # Regret/strategy buffers indexed by (node, canonical_hand)
            strategy = regret_match(preflop_regrets[node][hero_hand])
            for action, child in children:
                value[action] = cfr_traverse(child, ...)
            update_regrets(...)

        PreflopTerminal(Showdown):
            # Transition to postflop tree for this flop
            pot_type = pot_type_from_raise_count(node)
            spr = compute_spr(pot, stacks)
            postflop_root = select_postflop_tree(pot_type, spr)
            return cfr_traverse(postflop_root, ..., scale=pot)

        PreflopTerminal(Fold):
            return fold_payoff(pot, hero_investment)

        PreflopTerminal(AllIn):
            return equity * pot - hero_investment

        PostflopDecision:
            # Same CFR logic, buffers in batch-local postflop storage
            strategy = regret_match(postflop_regrets[node][hero_hand])
            ...

        PostflopTerminal(Showdown):
            return equity_table[hero_hand][opp_hand] * pot_fraction

        PostflopTerminal(Fold):
            return fold_payoff(pot_fraction, folder)
```

### Regret-Based Pruning

After a warmup period (e.g., 100 iterations), subtrees with cumulative regret below a negative threshold are skipped:

- **Preflop pruning**: If the regret for "3-bet with 72o" is deeply negative, the entire postflop subtree for 72o-in-3bet-pots is never traversed. This is where most of the savings come from — it eliminates postflop work for hand × pot-type combinations that don't occur at equilibrium.

- **Postflop pruning**: Within a postflop subtree, actions with deeply negative regret (e.g., overbetting with a marginal hand) are skipped. Standard regret-based pruning.

- **Threshold**: Configurable. More negative = more aggressive pruning = faster but risks missing edge cases. Suggested default: prune when cumulative regret < `-C * sqrt(iteration)` where C is tunable.

### DCFR Exploration

DCFR's existing α/β/γ discounting applies to the unified solver unchanged. The exploration mechanism for pruned subtrees:

- Every `explore_interval` iterations (e.g., every 50), traverse ALL subtrees regardless of pruning state
- If a previously-pruned subtree's regret has recovered above the threshold, it re-enters normal traversal
- This prevents permanent over-pruning as the strategy evolves

The explore interval is tunable and could also be expressed as a probability (e.g., 2% of iterations explore everything).

### Flop Batch Weighting

Not all flops are equally important. The batch weight reflects how often each flop is strategically relevant:

```
batch_weight(flop) = sum over (hero_hand, opp_hand) pairs:
    preflop_reach(hero_hand) * preflop_reach(opp_hand) * card_compatibility(hero, opp, flop)
```

Where `preflop_reach(hand)` is the probability that the current preflop strategy reaches postflop with that hand (sum of non-fold action probabilities at each preflop decision point). Card compatibility filters out deals where hands conflict with the board.

Weights are recomputed at the start of each epoch from the current preflop strategy.

### Postflop Trees

The postflop tree structure is reused from the existing `PostflopTree`. Each preflop terminal that reaches showdown maps to a postflop tree based on:

- **Pot type**: derived from raise count at the terminal (Limped/Raised/3bet/4bet+)
- **SPR**: computed from pot size and remaining stacks at the terminal

Different pot types could use different postflop tree configs (bet sizes, raise caps), though the initial implementation can share a single config.

### Memory Model

At any point in training, memory holds:

| Component | Size | Persistence |
|-|-|-|
| Preflop regret/strategy buffers | 169 hands × ~100 nodes × 2 buffers | Permanent |
| Preflop tree + equity table | ~169² floats + tree structure | Permanent |
| Postflop regret buffers (one batch) | 169 hands × ~300 nodes × batch_size flops | Per-batch |
| Postflop strategy accumulator | 169 hands × ~300 nodes × 1,755 flops | Permanent (or disk-backed) |
| Equity tables (one batch) | 169² × batch_size flops | Per-batch |

The postflop strategy accumulator is the largest component. Options:
1. **In-memory**: ~169² × 300 × 1,755 × 8 bytes ≈ ~120 GB. Too large.
2. **Disk-backed (memory-mapped)**: same data, backed by mmap. OS pages in/out as needed. Practical.
3. **Incremental merge to summary**: instead of storing the full postflop strategy, extract and store only the per-hand-pair EV at each postflop terminal after each batch. This is what the current architecture stores (PostflopValues). Much smaller: 169² × 2 × 1,755 × 8 bytes ≈ 800 MB.

Option 3 is recommended for the initial implementation. The aggregate strategy doesn't need to be stored in full — only the terminal EVs matter for the preflop solver's next epoch.

**With pruning**, the actual memory is much smaller. After warmup, most (hand, pot_type) pairs are pruned, so the batch-local postflop buffers only need entries for surviving pairs. A sparse allocation scheme (HashMap or conditional allocation) avoids paying for pruned subtrees.

### Convergence

Convergence is measured at two levels:

- **Preflop**: exploitability via best-response computation (existing `compute_exploitability`)
- **Postflop**: per-flop exploitability computed during the EV extraction phase (existing `compute_exploitability_for_flop`)
- **Overall**: the preflop exploitability implicitly captures postflop quality since terminal values depend on postflop strategy

Early stopping: if preflop exploitability drops below threshold and has been stable for N epochs, stop.

### Config

```yaml
solver:
  type: unified_cfr     # new solver type

  # Preflop
  stack_depth: 20
  raise_sizes: [2.5]
  raise_cap: 4

  # Postflop
  postflop_bet_sizes: [0.5, 1.0]
  postflop_max_raises_per_street: 1

  # Training
  epochs: 10
  flop_batch_size: 50           # flops per batch
  iterations_per_batch: 200     # CFR iterations per batch visit
  max_canonical_flops: 0        # 0 = all 1,755

  # DCFR
  dcfr_alpha: 1.5
  dcfr_beta: 0.5
  dcfr_gamma: 2.0
  dcfr_warmup: 30

  # Pruning
  prune_threshold: -1000.0      # cumulative regret below this → skip
  prune_warmup: 100             # iterations before pruning activates
  explore_interval: 50          # re-test pruned branches every N iterations
```

## Relationship to Existing Code

### What's Reused

- `PostflopTree` and tree building (`postflop_tree.rs`) — unchanged
- `PostflopLayout` for buffer indexing (`postflop_abstraction.rs`) — unchanged
- `compute_equity_table` from exhaustive backend (`postflop_exhaustive.rs`) — unchanged
- `PreflopTree` and preflop tree building (`preflop/tree.rs`) — unchanged
- DCFR discounting logic (`preflop/solver.rs`) — adapted
- `PotType::from_raise_count` routing — unchanged
- Card removal weights, canonical hand indexing — unchanged

### What's New

- `UnifiedSolver`: orchestrates the full-tree training loop with batching
- `UnifiedTraversal`: single recursive CFR function spanning preflop → postflop
- Pruning state: tracks per-(hand, node) cumulative regret for skip decisions
- Batch weight computation from preflop reach probabilities
- Sparse postflop buffer allocation (only for non-pruned hand × pot-type pairs)
- Strategy merge: folding batch-local results into the aggregate

### What's Replaced

- The separated `PreflopSolver` + `PostflopAbstraction` pipeline for training
- `postflop_showdown_value` — terminals now traverse into postflop trees directly
- The `PostflopState` intermediary between preflop and postflop

The existing separated architecture remains available as an alternative solver backend (faster, lower quality). The unified solver is a new option selected via config.

## Risks and Mitigations

| Risk | Mitigation |
|-|-|
| Memory pressure from postflop accumulator | Option 3 (incremental EV merge) keeps it to ~800 MB. Sparse allocation for pruned pairs. |
| Slow convergence with batched flops | Preflop regrets accumulate across all batches within an epoch. Postflop convergence depends on iterations_per_batch — set high enough (200+). |
| Over-pruning removes strategically relevant branches | DCFR exploration re-tests pruned branches. Conservative default threshold. |
| Postflop regrets restart each batch visit | Acceptable — the aggregate strategy still converges via incremental merging. Regrets within a batch visit converge enough with 200+ iterations. |
| Much slower than separated architecture | Expected. This is the quality/speed tradeoff. Pruning is the primary mitigation. The separated solver remains available for fast iteration. |

## Not In Scope

- Per-pot-type postflop bet sizes (future enhancement — easy to add once routing works)
- Turn/river card abstraction (full enumeration via equity tables)
- Real-time subgame solving integration
- Multi-threaded batch processing (batches are sequential; parallelism is within-batch via rayon over hand pairs)
