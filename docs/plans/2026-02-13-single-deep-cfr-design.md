# Single Deep CFR (SD-CFR) Solver Design

**Date:** 2026-02-13
**Paper:** Steinberger, "Single Deep Counterfactual Regret Minimization" (NeurIPS 2019 Deep RL Workshop)
**Reference impl:** https://github.com/EricSteinberger/Deep-CFR
**Status:** Approved

## Overview

A paper-faithful SD-CFR solver as a new crate (`deep-cfr`), complementing the existing tabular MCCFR solver. Uses candle (pure Rust) for neural networks with Metal/CUDA acceleration. SD-CFR is a simplified variant of Deep CFR that eliminates the policy network entirely — instead of training a second network to approximate the average strategy, it stores each iteration's trained value network and computes the average strategy on-the-fly.

**Key advantages over Deep CFR:**
- Removes sampling and approximation error from the policy network (Theorem 2)
- Reduces training time (no policy network training phase)
- Empirically lower exploitability and wins head-to-head (paper Section 6)

## Decisions

- **ML framework:** candle (pure Rust, no Python dependency)
- **Goal:** Second solver alongside tabular MCCFR, not a replacement
- **Algorithm:** SD-CFR (no policy network, store all value nets in model buffer)
- **NN architecture:** Paper-faithful two-branch MLP (~100K params)
- **NN input:** Raw card embeddings with flop isomorphism (1,755 canonical flops)
- **Bet encoding:** Paper-faithful — (occurred, pot_fraction) per betting position
- **Model buffer:** All value nets stored in memory (~360MB for 450 iterations)
- **Average strategy:** Both trajectory sampling (gameplay) and explicit computation (exploitability)
- **Traversal:** Pure external sampling (all actions at traverser, one at opponent)
- **Training scale:** Small initially (100K buffer, 64-dim, ~100K params), scalable later
- **Code organization:** New `crates/deep-cfr/` crate

## Reuse from Existing Codebase

- `Game` trait, `Action`, `Player`, `Actions` — entire game interface
- `KuhnPoker` — convergence test target
- `HunlPostflop` — the real game
- `rs_poker::core::Card`, `Value`, `Suit` — card representation
- `PostflopState` — provides all raw data for feature encoding (board, holdings, pot, stacks, actions)

**Not reused (fundamentally different):**
- `MccfrSolver` — tabular storage doesn't apply
- `InfoKey` / `info_set_key()` — SD-CFR uses raw feature vectors, not packed u64 keys
- `BlueprintStrategy` — tabular strategy storage

## Crate Structure

```
crates/deep-cfr/
├── Cargo.toml            # poker-solver-core, candle-core, candle-nn, rand
├── src/
│   ├── lib.rs            # public API, SdCfrError
│   ├── solver.rs         # SdCfrSolver<G: Game> — outer training loop
│   ├── traverse.rs       # external-sampling traversal using NN strategies
│   ├── network.rs        # AdvantageNet — two-branch MLP (card + bet → trunk → output)
│   ├── memory.rs         # ReservoirBuffer<T> with weighted reservoir sampling
│   ├── model_buffer.rs   # ModelBuffer (B^M) — stores trained value net weights per iteration
│   ├── card_features.rs  # suit isomorphism + card/bet feature encoding
│   ├── eval.rs           # TrajectoryPolicy + ExplicitPolicy (two avg strategy methods)
│   └── config.rs         # SdCfrConfig
```

## Neural Network Architecture

Two-branch MLP, ~100K parameters at dim=64. Only one network type (advantage/value network).

### Card Branch (3 layers)

- Per card: `rank_emb(13, 64) + suit_emb(4, 64) + card_emb(52, 64)` — summed
- Cards with index -1 (not yet dealt) are zeroed out
- Sum within each group (hole, flop, turn, river), concat → 256-dim
- `Linear(256, 64) → ReLU → Linear(64, 64) → ReLU → Linear(64, 64) → ReLU`

### Bet Branch (2 layers)

- 4 rounds × 6 positions × 2 features (occurred, pot_fraction) = 48 floats
- `Linear(48, 64) → ReLU → Linear(64, 64) + skip → ReLU`

### Combined Trunk (3 layers)

- Concat card + bet branches → 128-dim
- `Linear(128, 64) → ReLU → Linear(64, 64) + skip → ReLU → Linear(64, 64) + skip → ReLU`
- Normalize last hidden layer (zero mean, unit variance)
- `Linear(64, |A|)` — output head

### Output Interpretation

- Raw advantages → clamp negatives to 0 → normalize to get strategy
- All advantages ≤ 0 → play highest-advantage action deterministically (not uniform)

Skip connections apply only on layers with matching input/output dim (64→64).

## Sample Type

Only one sample type (no StrategySample — that's the SD-CFR simplification):

```rust
struct AdvantageSample {
    info_set: InfoSetFeatures,    // pre-encoded card + bet features
    iteration: u32,               // CFR iteration t (Linear CFR weight)
    advantages: Vec<f32>,         // per-action instantaneous regrets
    num_actions: u8,
}

struct InfoSetFeatures {
    cards: [i8; 7],               // canonicalized: 2 hole + 3 flop + 1 turn + 1 river
    bets: [f32; 48],              // 4 rounds × 6 positions × (occurred, pot_fraction)
}
```

## Reservoir Buffer

- Reservoir sampling (Vitter 1985): when full, new sample replaces random existing sample with probability `capacity / total_seen`
- Samples never removed between CFR iterations — accumulates across entire run
- One buffer per player for advantage samples

## Training Loop (SD-CFR Algorithm)

```
for t in 1..=T:
    for p in [P1, P2]:                              # alternating updates
        for k in 1..=K:
            deal = game.random_initial_state()
            traverse(deal, p, t, value_nets, adv_mem_p)

        value_net_p = train_from_scratch(adv_mem_p, t)   # fresh weights each iter
        model_buffer_p.push(value_net_p, t)               # SD-CFR: store the net
```

### Key Details

- **Alternating updates:** One player per half-iteration
- **Advantage memory:** One `ReservoirBuffer<AdvantageSample>` per player, accumulates across all iterations
- **Train from scratch:** Fresh random weights each iteration. Loss = weighted MSE, weight = iteration `t`, rescaled by `t / T_max`
- **Model buffer push:** After training, clone the value net weights and store with weight `t`
- **No strategy samples:** At opponent nodes we only sample an action

### Traversal (External Sampling)

- **Traverser nodes:** Explore ALL legal actions, compute per-action counterfactual values, store instantaneous advantages in reservoir buffer
- **Opponent nodes:** Get strategy from current value net (ReLU + normalize), sample ONE action, recurse
- **Chance nodes:** Already resolved (full board pre-dealt, revealed progressively)

### Value Network Training (from scratch each iteration)

- Fresh random weights each CFR iteration (not fine-tuned)
- Loss: weighted MSE over valid actions
- Weights: Linear CFR — sample weight = iteration `t`, rescaled by `t / T_max`
- Optimizer: Adam, lr=0.001, gradient norm clipping
- Illegal action outputs masked to 0 in loss

## Model Buffer (B^M)

```rust
struct ModelBuffer {
    entries: Vec<ModelEntry>,    // one per training iteration
}

struct ModelEntry {
    weights: Vec<u8>,           // serialized candle VarMap (safetensors format)
    iteration: u32,             // CFR iteration t
    weight: f64,                // = t (linear CFR weighting)
}
```

- Grows by one entry per player per iteration (2 buffers, one per player)
- 450 iterations → 450 entries per player → ~180MB per player
- No eviction — paper Section 5.4 recommends keeping all nets (reservoir sampling on B^M degrades performance)

## Average Strategy Computation

### Method 1 — Trajectory Sampling (for gameplay/rollouts)

- At episode start: sample one `ModelEntry` from buffer, weighted by `t`
- Use that single value net for the entire trajectory
- Correctly implements the linear average strategy (Theorem 2, Equation 5)
- Cost: one NN forward pass per decision

```rust
pub struct TrajectoryPolicy {
    buffers: [ModelBuffer; 2],       // one per player
    sampled_nets: [AdvantageNet; 2], // loaded at episode start
}
```

### Method 2 — Explicit Computation (for exploitability)

- At any info set I, compute the full average strategy
- For each value net in buffer, compute reach probability to I
- Weighted sum: σ̄(I,a) = Σ_t [t · π^σt(I) · σt(I,a)] / Σ_t [t · π^σt(I)]
- Reach probabilities computed incrementally along the trajectory
- Cost: |B^M| forward passes per info set (expensive, only for measurement)

```rust
pub struct ExplicitPolicy {
    buffers: [ModelBuffer; 2],
    nets: [Vec<AdvantageNet>; 2],    // all nets loaded
}
```

## Suit Isomorphism

Canonicalize by assigning suits in order of first appearance, scanning: flop[0], flop[1], flop[2], hole[0], hole[1], turn, river. This guarantees flops map to one of 1,755 canonical forms. All cards (hole, turn, river) are permuted with the same mapping.

## Feature Encoding Pipeline

```
PostflopState → canonicalize(hole, board) → InfoSetFeatures { cards: [i8;7], bets: [f32;48] }
             → to_tensors() → (card_indices: Tensor[B,7], bets: Tensor[B,48])
             → net.forward() → Tensor[B, |A|]
```

## Public API

```rust
pub struct SdCfrSolver<G: Game> { ... }

impl<G: Game> SdCfrSolver<G> {
    pub fn new(game: G, config: SdCfrConfig) -> Self;
    pub fn train(&mut self, progress: Option<&ProgressBar>) -> Result<TrainedSdCfr>;
    pub fn step(&mut self, iteration: u32) -> Result<()>;
    pub fn iteration(&self) -> u32;
}

pub struct TrainedSdCfr { ... }

impl TrainedSdCfr {
    pub fn trajectory_policy(&self) -> TrajectoryPolicy;   // for gameplay
    pub fn explicit_policy(&self) -> ExplicitPolicy;        // for exploitability
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn load(path: &Path, device: &Device) -> Result<Self>;
}
```

## Config

| Parameter | Default | Paper 5-FHP |
|---|---|---|
| cfr_iterations (T) | 100 | 300 |
| traversals_per_iter (K) | 1,000 | 300,000 |
| advantage_memory_cap | 100,000 | 2,000,000 |
| hidden_dim | 64 | 64 |
| num_actions (max \|A\|) | 14 | varies |
| sgd_steps_value | 4,000 | 4,000 |
| batch_size | 10,000 | 10,240 |
| learning_rate | 0.001 | 0.001 |
| grad_clip_norm | 1.0 | 10.0 |
| init_value_net | "random" | "random" |

## Error Handling

```rust
#[derive(thiserror::Error, Debug)]
pub enum SdCfrError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no samples in advantage buffer")]
    EmptyBuffer,
    #[error("empty model buffer")]
    EmptyModelBuffer,
}
```

## Testing Strategy

### Unit Tests (per module)

1. **memory.rs** — reservoir sampling uniformity, batch sampling, capacity behavior
2. **card_features.rs** — suit isomorphism produces identical features for isomorphic boards, absent card encoding, bet encoding round-trip
3. **network.rs** — output shape, skip connections, normalization, gradient flow
4. **model_buffer.rs** — push/iterate, weighted sampling distribution, serialization round-trip
5. **traverse.rs** — regret matching correctness (ReLU + normalize), traverser explores all actions, opponent samples one
6. **eval.rs** — trajectory policy samples proportional to iteration weight, explicit policy matches known equilibrium

### Integration Tests

- **Kuhn poker convergence:** SD-CFR should converge to exploitability < 0.05 within ~50 iterations. Validates entire pipeline end-to-end. Reuses existing `KuhnPoker` Game impl.
- **Determinism:** Same seed → identical results
- **Save/load round-trip:** `TrainedSdCfr` save → load → trajectory policy produces same strategy

### Exploitability Measurement

- Kuhn poker: exact best-response via full tree traversal
- HUNL: future work (out of scope for initial implementation)
