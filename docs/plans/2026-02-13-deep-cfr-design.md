# Deep CFR Solver Design

**Date:** 2026-02-13
**Paper:** Brown et al., "Deep Counterfactual Regret Minimization" (ICML 2019)
**Status:** Approved

## Overview

A paper-faithful Deep CFR solver as a new crate (`deep-cfr`), complementing the existing tabular MCCFR solver. Uses candle (pure Rust) for neural networks with Metal/CUDA acceleration. Replaces tabular regret/strategy storage with two neural networks: a value network V (predicts per-action advantages) and a policy network Π (approximates the average strategy).

## Decisions

- **ML framework:** candle (pure Rust, no Python dependency)
- **Goal:** Second solver alongside tabular MCCFR, not a replacement
- **NN input:** Raw card embeddings with flop isomorphism (1,755 canonical flops)
- **Bet encoding:** Paper-faithful — (occurred, pot_fraction) per betting position
- **Training scale:** Small initially (100K buffer, 64-dim, ~100K params), scalable later
- **Code organization:** New `crates/deep-cfr/` crate

## Crate Structure

```
crates/deep-cfr/
├── Cargo.toml            # poker-solver-core, candle-core, candle-nn, rand
├── src/
│   ├── lib.rs            # public API, DeepCfrError
│   ├── solver.rs         # DeepCfrSolver<G: Game> — outer training loop
│   ├── traverse.rs       # external-sampling traversal using NN strategies
│   ├── network.rs        # DeepCfrModel — shared architecture for V and Π
│   ├── memory.rs         # ReservoirBuffer<T> with weighted reservoir sampling
│   ├── sample.rs         # AdvantageSample, StrategySample types
│   ├── card_features.rs  # suit isomorphism + card/bet feature encoding
│   └── config.rs         # DeepCfrConfig
```

## Neural Network Architecture

Two-branch MLP, ~100K parameters at dim=64. Same architecture for V and Π networks.

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

- **Value network V:** Raw advantages per action (regression targets)
- **Policy network Π:** Logits → softmax for action probabilities

Skip connections apply only on layers with matching input/output dim (64→64).

## Sample Types

```rust
struct AdvantageSample {
    info_set: InfoSetFeatures,    // pre-encoded card + bet features
    iteration: u32,               // CFR iteration t (Linear CFR weight)
    advantages: Vec<f32>,         // per-action instantaneous regrets
    num_actions: u8,
}

struct StrategySample {
    info_set: InfoSetFeatures,
    iteration: u32,
    strategy: Vec<f32>,           // σ^t(I) probability vector
    num_actions: u8,
}

struct InfoSetFeatures {
    cards: [i8; 7],               // canonicalized: 2 hole + 3 flop + 1 turn + 1 river
    bets: [f32; 48],              // 24 × (occurred, pot_fraction)
}
```

## Reservoir Buffer

- Reservoir sampling (Vitter 1985): when full, new sample replaces random existing sample with probability `capacity / total_seen`
- Samples never removed between CFR iterations — accumulates across entire run
- Sliding window does NOT work (paper Figure 4) — reservoir sampling is critical for convergence

## Training Loop (Algorithm 1)

```
for t in 1..=T:
    for p in [P1, P2]:
        for k in 1..=K:
            deal = game.random_initial_state()
            traverse(deal, p, t, value_nets, adv_mem_p, strat_mem)

        value_net_p = train_from_scratch(adv_mem_p, t)

    policy_net = train_from_scratch(strat_mem, T)   # only at the end
```

### Traversal (Algorithm 2 — External Sampling)

- **Traverser nodes:** Explore ALL actions, compute advantages, store in advantage memory
- **Opponent nodes:** Compute strategy from value net via regret matching, sample ONE action, store strategy in strategy memory
- **Strategy from NN:** Forward pass → clamp negatives to 0 → normalize. All regrets ≤ 0 → play highest-regret action (not uniform)

### Value Network Training (from scratch each iteration)

- Fresh random weights each CFR iteration (not fine-tuned)
- Loss: weighted MSE over valid actions
- Weights: Linear CFR — sample weight = iteration `t`, rescaled by `2/T`
- Optimizer: Adam, lr=0.001, gradient norm clipping at 1.0

## Suit Isomorphism

Canonicalize by assigning suits in order of first appearance, scanning: flop[0], flop[1], flop[2], hole[0], hole[1], turn, river. This guarantees flops map to one of 1,755 canonical forms. All cards (hole, turn, river) are permuted with the same mapping.

## Feature Encoding Pipeline

```
state → canonicalize(hole, board) → InfoSetFeatures { cards: [i8;7], bets: [f32;48] }
      → to_tensors() → (card_indices: Tensor[B,7], bets: Tensor[B,48])
      → model.forward() → Tensor[B, |A|]
```

## Public API

```rust
pub struct DeepCfrSolver<G: Game> { ... }

impl<G: Game> DeepCfrSolver<G> {
    pub fn new(game: G, config: DeepCfrConfig) -> Self;
    pub fn train(&mut self, progress: Option<&ProgressBar>) -> Result<TrainedPolicy>;
    pub fn step(&mut self, iteration: u32) -> Result<()>;
    pub fn iteration(&self) -> u32;
}

pub struct TrainedPolicy { ... }

impl TrainedPolicy {
    pub fn strategy(&self, state: &impl GameState) -> Vec<f32>;
    pub fn strategy_for_state<G: Game>(&self, game: &G, state: &G::State) -> Vec<f32>;
    pub fn save(&self, path: &Path) -> Result<()>;
    pub fn load(path: &Path, device: &Device) -> Result<Self>;
}
```

## Config (small defaults)

| Parameter | Default | Paper FHP | Paper HULH |
|---|---|---|---|
| cfr_iterations (T) | 100 | 450 | 450 |
| traversals_per_iter (K) | 1,000 | 10,000 | 10,000 |
| advantage_memory_cap | 100,000 | 40,000,000 | 40,000,000 |
| strategy_memory_cap | 100,000 | 40,000,000 | 40,000,000 |
| hidden_dim | 64 | 64 | 64 |
| num_actions (max \|A\|) | 14 | varies | varies |
| sgd_steps_value | 4,000 | 4,000 | 32,000 |
| sgd_steps_policy | 4,000 | 4,000 | 32,000 |
| batch_size | 10,000 | 10,000 | 20,000 |
| learning_rate | 0.001 | 0.001 | 0.001 |
| grad_clip_norm | 1.0 | 1.0 | 1.0 |

## Error Handling

```rust
#[derive(thiserror::Error, Debug)]
pub enum DeepCfrError {
    #[error("candle error: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("no samples in memory buffer")]
    EmptyBuffer,
}
```

## Testing Strategy

### Unit Tests (per module)

1. **memory.rs** — reservoir sampling uniformity, batch sampling, capacity behavior
2. **card_features.rs** — suit isomorphism produces identical features for isomorphic boards, absent card encoding, bet encoding round-trip
3. **network.rs** — output shape, skip connections, normalization, gradient flow
4. **traverse.rs** — regret matching correctness, traverser explores all / opponent samples one

### Integration Test

- **Kuhn poker convergence:** Implement Kuhn poker as a `Game` trait impl. Deep CFR should converge to exploitability < 0.05 within ~50 iterations. This validates the entire pipeline.
- **Determinism:** Same seed → identical results
- **Checkpoint round-trip:** save → load → strategy matches

### Exploitability Measurement

- Kuhn poker: exact best-response via full tree traversal
- HUNL: future work (out of scope for initial implementation)
