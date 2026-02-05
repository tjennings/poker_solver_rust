# Blueprint Solver Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a Nash equilibrium approximation for full HUNL poker using card abstraction, enabling real-time subgame solving.

**Architecture:** Offline blueprint training via MCCFR on abstracted game tree (~30k buckets), with real-time subgame solving for refined decisions during play. Persistent caching avoids re-solving common situations.

**Tech Stack:** Rust, Burn (GPU), existing CardAbstraction, existing GpuCfrSolver

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Blueprint System                          │
├─────────────────────────────────────────────────────────────┤
│  HunlPostflop (Game trait)                                  │
│    ├── State: board, holdings, stacks, history, street      │
│    ├── info_set_key() → uses CardAbstraction buckets        │
│    └── Handles: preflop → flop → turn → river               │
├─────────────────────────────────────────────────────────────┤
│  CardAbstraction (existing)                                 │
│    └── get_bucket(board, holding) → u16 bucket ID           │
├─────────────────────────────────────────────────────────────┤
│  GpuCfrSolver (existing)                                    │
│    └── train(game, iterations) → strategy                   │
├─────────────────────────────────────────────────────────────┤
│  BlueprintStrategy (new)                                    │
│    ├── lookup(info_set) → action probabilities              │
│    └── save/load for persistence (~5GB)                     │
├─────────────────────────────────────────────────────────────┤
│  SubgameSolver (new)                                        │
│    ├── solve(situation, time_budget) → refined probs        │
│    └── Uses blueprint at depth-limit leaves                 │
├─────────────────────────────────────────────────────────────┤
│  SubgameCache (new)                                         │
│    ├── memory_cache: LRU for hot lookups                    │
│    └── disk_cache: Persistent across sessions               │
└─────────────────────────────────────────────────────────────┘
```

## 2. HunlPostflop Game Type

### State Representation

```rust
pub struct PostflopState {
    // Cards
    board: Vec<Card>,           // 0 (preflop) to 5 (river) cards
    p1_holding: [Card; 2],      // Player 1's hole cards
    p2_holding: [Card; 2],      // Player 2's hole cards

    // Betting state
    street: Street,             // Preflop, Flop, Turn, River
    pot: u32,                   // Total chips in pot
    stacks: [u32; 2],           // Remaining stacks [P1, P2]
    to_call: u32,               // Amount to call
    to_act: Option<Player>,     // Who acts next (None if terminal)

    // History (for info set key)
    action_history: Vec<(Street, Action)>,

    // Terminal info
    terminal: Option<TerminalType>,
}

pub enum TerminalType {
    Fold(Player),   // Player folded
    Showdown,       // Went to showdown
}
```

### Info Set Key Format

```
"<bucket>|<street>|<action_history>"

Examples:
- "1234|F|xb50c"       → Bucket 1234, Flop, check-bet50-call
- "5678|T|xb50c|xb75"  → Bucket 5678, Turn, flop actions + turn bet
- "0042|P|r3c"         → Bucket 42, Preflop, raise3x-call
```

The bucket ID from CardAbstraction replaces actual cards, reducing millions of holdings to ~30k buckets.

### Action Abstraction

Fixed bet sizes per street to keep action tree tractable:

| Street | Bet Sizes |
|--------|-----------|
| Preflop | 2.5x, 3x, all-in |
| Flop | 33%, 50%, 75%, 100% pot, all-in |
| Turn | 33%, 50%, 75%, 100% pot, all-in |
| River | 33%, 50%, 75%, 100% pot, all-in |

## 3. Monte Carlo Sampling

### Problem

Full HUNL has ~1.3 trillion game states. Even with abstraction, enumerating all deals is infeasible.

### Solution: MCCFR

Sample deals rather than enumerate:

```rust
impl Game for HunlPostflop {
    fn initial_states(&self) -> Vec<PostflopState> {
        let mut rng = StdRng::seed_from_u64(self.current_seed);
        let mut states = Vec::with_capacity(self.samples_per_iteration);

        for _ in 0..self.samples_per_iteration {
            let mut deck = Deck::new();
            deck.shuffle(&mut rng);
            let p1 = [deck.deal(), deck.deal()];
            let p2 = [deck.deal(), deck.deal()];
            states.push(PostflopState::new(p1, p2, self.stack_depth));
        }
        states
    }
}
```

### Board Progression

When transitioning streets, sample next card(s) from remaining deck:
- Preflop → Flop: Deal 3 cards
- Flop → Turn: Deal 1 card
- Turn → River: Deal 1 card

### Variance Reduction

Use same RNG seed across both players' traversals within an iteration (importance sampling).

## 4. Blueprint Strategy Storage

### Representation

```rust
pub struct BlueprintStrategy {
    // Map from info set key to action probabilities
    strategies: HashMap<String, Vec<f32>>,

    // Metadata
    iterations_trained: u64,
    abstraction_config: AbstractionConfig,
}
```

### Storage

- In-memory: `HashMap` for fast lookup
- On-disk: bincode + lz4 compression
- Estimated size: ~5GB for full HUNL with 30k buckets

### API

```rust
impl BlueprintStrategy {
    /// Extract strategy from trained solver
    pub fn from_solver<B: Backend>(
        solver: &GpuCfrSolver<B, HunlPostflop>
    ) -> Self;

    /// Lookup action probabilities for an info set
    pub fn lookup(&self, info_set: &str) -> Option<&[f32]>;

    /// Persistence
    pub fn save(&self, path: &Path) -> Result<(), BlueprintError>;
    pub fn load(path: &Path) -> Result<Self, BlueprintError>;
}
```

### Training API

```rust
let abstraction = CardAbstraction::load("boundaries.bin")?;
let game = HunlPostflop::new(config, abstraction);
let mut solver = GpuCfrSolver::new(game, device);

// Train with checkpointing
for i in 0..1_000_000 {
    solver.iterate();
    if i % 10_000 == 0 {
        solver.checkpoint(&format!("checkpoint_{}.bin", i))?;
    }
}

let blueprint = BlueprintStrategy::from_solver(&solver);
blueprint.save("blueprint.bin")?;
```

## 5. Subgame Solving

### Purpose

Blueprint uses coarse abstraction (~30k buckets). During play, solve depth-limited subgame with exact cards for better decisions.

### Solver

```rust
pub struct SubgameSolver {
    blueprint: Arc<BlueprintStrategy>,
    abstraction: Arc<CardAbstraction>,
    cache: SubgameCache,
    depth_limit: usize,  // 4 actions ahead
}

impl SubgameSolver {
    pub fn solve(
        &self,
        board: &[Card],
        holding: [Card; 2],
        history: &[(Street, Action)],
        time_budget_ms: u64,
    ) -> Vec<f32> {
        // 1. Check cache first
        // 2. Build depth-limited subgame tree
        // 3. At leaves, use blueprint as continuation value
        // 4. Run CFR+ for time_budget
        // 5. Cache and return refined action probabilities
    }
}
```

### Multi-Valued States (Modicum Approach)

At subgame leaves, don't assume single opponent strategy. Use 4-10 opponent "types":
- Aggressive, passive, balanced variants
- Weight by reach probability
- Reduces exploitability of subgame solution

### Timing Targets

| Street | Target | Rationale |
|--------|--------|-----------|
| Flop | 2-5s | Larger remaining tree |
| Turn | 100-300ms | Medium tree |
| River | 50-100ms | Small tree |

## 6. Subgame Cache

### Problem

Re-solving identical subgames wastes compute. Many game states recur.

### Solution

Two-tier cache with LRU eviction:

```rust
pub struct SubgameCache {
    // Hot: in-memory LRU
    memory_cache: LruCache<SubgameKey, Vec<f32>>,

    // Cold: disk-backed persistent storage
    disk_cache: DiskCache,  // sled or rocksdb

    config: CacheConfig,
}

pub struct CacheConfig {
    max_memory_entries: usize,  // ~100k
    max_disk_size_gb: f64,      // ~10GB
}

#[derive(Hash, Eq, PartialEq, Clone)]
pub struct SubgameKey {
    board_canonical: u64,   // Suit-isomorphic board hash
    holding_bucket: u16,    // Bucket (not exact cards)
    history_hash: u64,      // Action history hash
}
```

### Cache Flow

```rust
impl SubgameSolver {
    pub fn solve(&self, ...) -> Vec<f32> {
        let key = SubgameKey::new(board, holding, history, &self.abstraction);

        // 1. Memory cache hit
        if let Some(probs) = self.cache.memory_get(&key) {
            return probs.clone();
        }

        // 2. Disk cache hit
        if let Some(probs) = self.cache.disk_get(&key)? {
            self.cache.memory_insert(key.clone(), probs.clone());
            return probs;
        }

        // 3. Cache miss: solve and store
        let probs = self.solve_fresh(...);
        self.cache.insert(key, probs.clone())?;
        probs
    }
}
```

### Cache Key Design

- Uses **bucket** (not exact holding) so similar hands share solutions
- **Canonical board hash** ensures suit-isomorphic boards hit same entry
- History hash for action sequence matching

### Cache Warming

Pre-solve common situations offline:
- Standard flop textures (paired, monotone, rainbow, connected)
- Common bet sequences (cbet, check-raise, 3bet pots)
- High-frequency river spots

## 7. File Structure

```
crates/core/src/
├── blueprint/
│   ├── mod.rs           # Module exports
│   ├── strategy.rs      # BlueprintStrategy storage/lookup
│   ├── subgame.rs       # SubgameSolver real-time solving
│   ├── cache.rs         # SubgameCache (memory + disk)
│   └── error.rs         # BlueprintError types
├── game/
│   ├── mod.rs           # Add HunlPostflop export
│   ├── hunl_postflop.rs # Full HUNL game implementation
│   └── ...
```

## 8. Dependencies

New crates needed:
- `lru` - LRU cache implementation
- `sled` or `rocksdb` - Disk-backed key-value store
- `bincode` - Fast serialization (already have)
- `lz4` - Compression for strategy storage

## 9. Training Estimates

| Phase | Time | Output |
|-------|------|--------|
| Boundary generation | ~4 hours | boundaries.bin (~50MB) |
| Blueprint training | ~700 GPU-hours | blueprint.bin (~5GB) |
| Cache warming | ~10 hours | cache.db (~10GB) |

## 10. Success Criteria

1. **Blueprint converges:** Exploitability decreases over training
2. **Real-time solving:** <500ms decisions with warm cache
3. **Cache hit rate:** >80% for common situations
4. **Memory footprint:** <16GB RAM during play
5. **Integration:** Works with existing Tauri app UI
