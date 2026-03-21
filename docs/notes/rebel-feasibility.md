# ReBeL Feasibility: Single Machine Training

## TEH (Turn Endgame Hold'em) — Feasible

The paper used 1 GPU + 60 CPU threads for TEH (not the 90 DGX-1 cluster — that was full HUNL only). A modern RTX 4090 is ~5.5x faster than their V100 for inference.

**Estimated training time:** 3-7 days on RTX 4090 + 16 cores (speculative — paper doesn't disclose wall-clock time for TEH).

**Apple Silicon caveat:** CUDA needed for efficient batched neural net inference during CFR. wgpu on M-series will be 5-10x slower. RTX 4090 machine is the right hardware.

## Full HUNL — Not Practical

90 DGX-1s (720 V100 GPUs) for 1,750 epochs. Even with 5x faster GPU, that's ~130 GPU-equivalents of compute.

## Compute Bottleneck Breakdown

Data generation is ~95% of compute, neural net training ~5%.

Per training example:
1. Play a hand via self-play (sampling from policy network)
2. At each public state, solve a subgame via 1024 CFR iterations
3. Each CFR iteration queries the value network at every leaf PBS

Leaf evaluations per subgame: ~20-80 per CFR iteration × 1024 iterations = ~20K-80K value net forward passes per subgame. These can be batched per iteration but iterations are sequential.

## Staged Implementation Path

**Stage 1 (mostly done):** River cfvnet — trains a neural net to predict river counterfactual values.

**Stage 2 (straightforward):** Modify range-solver to use cfvnet at depth boundaries. When tree hits river, call neural net instead of building river subtree. We already have `PLAYER_DEPTH_BOUNDARY_FLAG` in the action tree. This gives DeepStack-style turn solving.

**Stage 3 (the real work):** Build the ReBeL self-play loop for TEH:
- Sample random turn boards
- Play turn betting round using current policy
- At each PBS, solve subgame with range-solver + value net at leaves
- Collect (PBS, values) as training data
- Train value net, repeat

New components needed:
- PBS representation (public state + 1326-dim belief vectors per player)
- Self-play data generation loop with reservoir sampling buffer (12M examples)
- Policy network for warm-starting CFR (critical — see warning below)
- Value network training orchestration (alternating data gen / training)

## Key Warning: Policy Network Is Not Optional

A documented reimplementation attempt (GitHub issue #25 on facebook/rebel) tried HUNL with 2 GPUs, reduced to 128 CFR iterations, omitted the policy network, and diverged. Training error increased, validation was inconsistent.

Without warm-starting CFR from a policy net, reduced-iteration subgame solves don't converge well enough, and the value net trains on noisy targets.

## Existing Infrastructure We Can Reuse

- `range-solver/src/solver.rs` — DCFR solver → becomes ReBeL's subgame solver
- `cfvnet/src/model/network.rs` — Value network architecture (close match to ReBeL's needs)
- `cfvnet/src/datagen/solver.rs` — Wrapper for solving and extracting CFVs
- `range-solver/src/action_tree.rs` — Already has `PLAYER_DEPTH_BOUNDARY_FLAG`
- Hand evaluation, card isomorphism, equity computation — all exist

## Related Work

- **Student of Games** (Schmid et al. 2023) — more general but more expensive (3500 TPUv4 actors)
- **DREAM/Single Deep CFR** — simpler, no search at test time, weaker results
- **AlphaHoldem** (AAAI 2022) — end-to-end RL, no CFR, beat Slumbot on single PC in 3 days but not tested against superhuman baselines
- **RL-CFR** (Li et al. ICML 2024) — extends ReBeL with learned action abstractions, beat ReBeL by 64 mbb/h
- No published work demonstrates working single-machine ReBeL for any poker variant
