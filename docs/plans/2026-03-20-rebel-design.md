# ReBeL Implementation Design

**Date:** 2026-03-20
**Status:** Approved
**Hardware:** Single RTX 6000 Ada (48GB VRAM) + 16 CPU threads

## Overview

ReBeL (Recursive Belief-based Learning) for HUNL poker, seeded from an existing blueprint V2. Combines deep reinforcement learning with search: a value network learns to predict counterfactual values at public belief states (PBSs), and depth-limited CFR subgame solving uses the value net at leaves.

The key innovation over a from-scratch ReBeL: we seed the value net from a trained blueprint, generating realistic training data from the start. This avoids the cold-start divergence problem reported in other reimplementation attempts.

## Decisions

| Question | Answer |
|----------|--------|
| Value net architecture | Single network for all streets (reuse cfvnet: 2720→7×500→1326) |
| ML framework | Burn (Rust-native, CUDA backend) |
| Input representation | Existing cfvnet format (2720 features: beliefs + board + pot + stack + player) |
| Offline data order | Bottom-up: river → turn → flop → preflop |
| Live self-play policy | Full subgame solving at every decision (pure ReBeL, no policy net shortcut) |
| CFR iterations per subgame | 1024 default, configurable |
| Training data buffer | Disk-backed with mmap random sampling |
| Validation | MSE every epoch, exploitability daily, head-to-head weekly |

## Blueprint's Role: Seeding Realistic Training Data

The blueprint provides **which PBSs to train on** by generating realistic belief distributions. It does NOT provide strategies or values within subgames — the range-solver computes those exactly at full 1326-combo resolution.

**PBS generation flow:**
1. Load trained blueprint (strategy bundle + bucket files)
2. Deal random hole cards for both players
3. Play the hand under blueprint policy: at each decision point, look up the blueprint's strategy for this (node, bucket) pair, use it as the action probability
4. Track reach probabilities for both players through the hand (1326-dim vectors, updated by multiplying by action probabilities at each node)
5. At each betting round boundary, snapshot the PBS: (board, pot, stacks, P0 reach probs, P1 reach probs)
6. These snapshots are the subgame roots to solve

**De-bucketing:** All hands in the same bucket get the same action probability from the blueprint. The reach probability vectors are still per-combo (1326-dim) because different combos have different prior reach even within the same bucket (e.g., from preflop frequencies).

## Stage 1: Offline Data Generation (Bottom-Up)

**River:** Sample millions of river PBSs from blueprint play. Solve each with range-solver (solves to showdown, no leaf evaluator needed). Train river value net.

**Turn:** Sample turn PBSs from blueprint play. Solve with range-solver using depth_limit=0 (turn betting only), river value net as leaf evaluator via `LeafEvaluator` trait. Retrain value net on accumulated turn+river data.

**Flop:** Same pattern, value net at turn boundaries.

**Preflop:** Same pattern, value net at flop boundaries.

At each layer, the single value net accumulates knowledge from all streets trained so far. By preflop, it handles all four streets.

**Volume target:** ~2-5M training examples per street layer, ~10-20M total. Generated in parallel across 16 CPU threads.

## Stage 2: Value Network

**Architecture:** Single Burn MLP for all streets.
- Input: 2720 features (OOP beliefs 1326 + IP beliefs 1326 + board one-hot 52 + rank presence 13 + pot 1 + effective stack 1 + player 1)
- Hidden: 7 layers × 500 units, BatchNorm + PReLU
- Output: 1326 (per-combo counterfactual values)
- Loss: Huber (pointwise)
- Optimizer: Adam, lr=3×10⁻⁴, halve every 800 epochs

**Training data storage:** Disk-backed reservoir buffer. Binary format matches existing `TrainingRecord` layout (~16KB per record). New records appended sequentially. Training samples drawn randomly via mmap. Max buffer size configurable (default 12M records).

**Training cadence during live self-play:** After each batch of N subgame solves (configurable), run one training epoch over a random sample from the buffer.

**Held-out validation set:** Pre-solve ~10K subgames exactly across all streets. Measure MSE every epoch.

## Stage 3: Live Self-Play Loop

Matching Algorithm 1 from the paper:

```
while training_budget_remaining:
    deal random hole cards + board
    set initial PBS (uniform beliefs, root game state)

    while hand not terminal:
        build depth-limited subgame at current PBS
            (current betting round, value net at street boundary leaves)

        run CFR-D for T iterations (default 1024, configurable):
            each iteration:
                traverse subgame tree
                at leaf PBSs: batch query value net on GPU
                update regrets via regret matching

            compute average strategy over all iterations
            compute root infostate values via running average

        record training example: (PBS, root infostate values)

        sample a random CFR iteration t
        use π^t to sample actions for both players
        update beliefs via Bayes rule
        advance to next PBS

    append training examples to reservoir buffer

    every N hands: train value net on random batch from buffer
```

**Parallelism:** 16 CPU threads play independent hands. GPU handles batched value net inference and training.

**Belief updates:** After each action, multiply player i's reach for each infostate by π(a|s_i). Normalize to get new PBS beliefs.

**Exploration:** ε=0.25 during training (one random player takes uniform random action). ε=0 at test time.

## Validation & Metrics

**Tier 1 — Value net MSE (every epoch):** Predict CFVs for held-out test set (~10K pre-solved subgames across all streets). Report MSE per street.

**Tier 2 — Exploitability on fixed benchmark (daily):** Specific TEH spot solved exactly, compare ReBeL strategy exploitability against exact Nash.

**Tier 3 — Head-to-head vs blueprint (weekly):** 100K+ hands, ReBeL agent vs blueprint agent. Measure win rate in mbb/hand.

**Logging:** TUI dashboard showing training loss, validation MSE per street, throughput, GPU utilization, buffer size.

## Crate Structure

```
crates/rebel/
├── src/
│   ├── lib.rs               -- public API
│   ├── config.rs            -- RebelConfig (YAML)
│   ├── pbs.rs               -- PBS representation
│   ├── belief_update.rs     -- Bayesian belief updates
│   ├── self_play.rs         -- Hand simulation loop
│   ├── subgame_solve.rs     -- CFR-D with value net leaves
│   ├── data_buffer.rs       -- Disk-backed reservoir buffer
│   ├── training.rs          -- Value net training orchestration
│   ├── blueprint_sampler.rs -- Extract PBSs from blueprint play
│   └── validation.rs        -- MSE, exploitability, head-to-head
```

**Dependencies:** range-solver, poker-solver-core (card utils, hand eval, blueprint, cfvnet model, LeafEvaluator), burn.

**CLI:** New subcommands in poker-solver-trainer: `rebel-seed`, `rebel-train`, `rebel-eval`.

## Implementation Phases

**Phase 1 — PBS + Blueprint Sampling:** PBS struct, belief updates, blueprint sampler, disk buffer. Target: generate and store millions of realistic PBSs.

**Phase 2 — Offline River Seeding:** Wire range-solver to solve river subgames at sampled PBSs. Train value net on river data. Validate MSE on held-out river solves.

**Phase 3 — Depth-Limited Multi-Street:** Integrate value net as LeafEvaluator. Bottom-up offline seeding: turn → flop → preflop. Retrain value net on accumulated multi-street data.

**Phase 4 — Live Self-Play Loop:** Full ReBeL Algorithm 1. GPU batching. Alternating data gen and training. Exploration.

**Phase 5 — Validation & Tuning:** Exploitability benchmark, head-to-head vs blueprint, TUI dashboard, hyperparameter tuning.

## Existing Infrastructure Leveraged

- `LeafEvaluator` trait with batch interface (`cfv_subgame_solver.rs`)
- `CfvSubgameSolver` orchestrating depth-bounded solving + neural evaluation
- Range-solver with `PLAYER_DEPTH_BOUNDARY_FLAG` and `depth_limit` config
- CfvNet model architecture (Burn-based, 2720→1326)
- Datagen pipeline (situation sampling, solving, binary storage format)
- Blueprint loading and strategy lookup
- Hand evaluation, card isomorphism, equity computation
