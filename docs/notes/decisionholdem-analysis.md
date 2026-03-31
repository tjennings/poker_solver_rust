# DecisionHoldem Analysis

Competitive analysis of [DecisionHoldem](https://github.com/jmh-sda/DecisionHoldem) — a C++11 HUNL poker solver with blueprint training and real-time subgame solving.

## Overview

- **Language**: C++11, header-only (~4.3K LOC), Python/Tornado web GUI
- **Build**: `g++ -std=c++11 -mcmodel=large -lpthread` (targets 512GB+ machines)
- **Architecture**: Two-stage — offline blueprint MCCFR + real-time depth-limited subgame solving

## CFR Algorithm

External-sampling MCCFR with:
- **Linear discounting**: `d = t/(t+1)`, applied for first 4M iterations only, every 10K iters
- **CFR pruning (CFVP)**: Skips actions with cumulative regret < -200M after 100K iterations (5% of iterations run without pruning for exploration)
- **Strategy averaging**: Updated every 5K iterations
- **Training**: 100M iterations, 100 pthreads, 2K iterations per thread task, 3-4 days on 48-core CPU
- **Regret bounds**: Clipped to [-210M, 200M] to prevent overflow

Key difference from WarpGTO: their discounting is simpler (linear, stops early) vs our full DCFR with tunable alpha/beta/gamma. DCFR should converge better in later iterations.

## Hand Abstraction

**Method**: Brute-force precomputed binary lookup tables. Every possible (hand, board) → cluster mapping is stored in sorted arrays and queried via binary search at runtime. ~30 GB resident memory.

### Cluster Counts & Files

| Street  | Clusters | Entries per hand | Memory   | Lookup file               |
|---------|----------|-----------------|----------|---------------------------|
| Preflop | 169      | 1 (direct)      | 2.6 KB   | `preflop_hand_cluster.bin` |
| Flop    | 50,000   | 19,600          | ~247 MB  | `flop_hand_cluster.bin`    |
| Turn    | 5,000    | 230,300         | ~2.9 GB  | `turn_hand_cluster.bin`    |
| River   | 1,000    | 2,118,760       | ~25.4 GB | `river_hand_cluster.bin`   |

Additional files:
- `sevencards_strength.bin` — 133M-entry 7-card hand evaluator (~1.34 GB)
- `preflopallin1326.1225.bin` — Preflop all-in equity for all 1326×1326 matchups (~14 MB)

### Card Encoding

Cards are unsigned char 0-51. Hand index: `handid = c1 * 52 + c2` (up to 2652).

Community cards use positional hashing with sorted cards:
- **Flop**: `c0×2704 + c1×52 + c2` (2704 = 52²)
- **Turn**: `c0×140608 + c1×2704 + c2×52 + c3` (140608 = 52³)
- **River**: `c0×7311616 + c1×140608 + c2×2704 + c3×52 + c4` (7311616 = 52⁴)

### Data Structures (Engine.h)

Each hand has its own sorted array of reachable boards:

```cpp
struct flop_community {
    unsigned* keys;      // sorted community card encodings
    unsigned* values;    // cluster IDs (0-49,999)
};
struct turn_community {
    unsigned* keys;
    unsigned* values;    // cluster IDs (0-4,999)
};
struct river_community {
    unsigned* keys;
    unsigned short* values;  // cluster IDs (0-999)
};
```

### Lookup Mechanism

**Preflop**: Direct array lookup — `preflop_cluster[c1*52 + c2]` → O(1).

**Flop/Turn/River**: Binary search over per-hand sorted arrays:
```cpp
unsigned get_flop_cluster(a1, a2, com[]) {
    sort(com, 3);
    community_id = com[0]*2704 + com[1]*52 + com[2];
    return find_flop(a1*52 + a2, community_id);  // binary search
}
```

Query complexity:
- Flop: O(log 19,600) ≈ 14 comparisons
- Turn: O(log 230,300) ≈ 18 comparisons
- River: O(log 2,118,760) ≈ 21 comparisons

### 7-Card Hand Evaluator

133M-entry sorted lookup using bitmask keys:
```cpp
ll bitmask = (1LL << c0) | (1LL << c1) | ... | (1LL << c6);
unsigned short strength = find_strength(bitmask);  // O(log 133M) ≈ 27 comparisons
```
Returns unsigned short (0-65535) hand strength ranking. Used in `compute_winner()` and for subgame leaf evaluation.

### Preflop All-In Equity Table

Stores exhaustive win counts for all hand-vs-hand preflop matchups:
```cpp
int* preflop_allin[2652];   // preflop_allin[myhand][opphand] = win count
const int divisor = 1712304; // normalize to probability
double equity = preflop_allin[c1*52+c2][i*52+j] / (double)divisor;
```
Used for fast EV computation at preflop all-in nodes during subgame solving.

### How Clusters Feed the Game Tree

During CFR traversal, at street transitions the cluster ID indexes directly into the child array:
```cpp
// At chance node (new street dealt):
cnode[player] = cnode[player]->actions + clusters[betting_stage];
```
Simple, cache-friendly — cluster ID is an array offset, not a hash lookup.

### Clustering Generation

**Opaque.** No clustering pipeline exists in the codebase. Files are distributed as pre-generated binaries via Baidu NetDisk. We cannot determine the exact clustering algorithm (k-means, EMD, or other) from the source code alone.

### Limitations vs WarpGTO

| Gap | Detail |
|-----|--------|
| **Not potential-aware** | Each street clusters by current hand strength only. A flush draw and made pair with identical current strength get the same cluster. WarpGTO clusters by distribution over next-street buckets (Pluribus-style). |
| **No suit isomorphism** | Every suited variant stored separately. WarpGTO's canonical board representation compresses the space. |
| **Fixed bucket counts** | Hardcoded 169/50K/5K/1K. WarpGTO's pipeline is configurable. |
| **Massive memory** | ~30 GB resident at runtime vs WarpGTO's compact bucket files. |
| **Opaque generation** | Can't tune, iterate, or improve abstractions without external tooling. |

## Game Tree & Action Abstraction

Hardcoded bet sizing fractions (not configurable):
- **Preflop/Flop**: 0.5x, 1x, 2x, 4x, 8x, 10x, 20x pot + all-in
- **Turn**: Limited to 2 actions, up to 2x pot
- **River**: Limited to 2 actions, up to 1x pot
- Raise formula: `last_raise = pot * action / 200 * 100`

Tree nodes (`strategy_node`): action_len, action chars, regret[], averegret[], child pointers.

No SPR awareness in info set encoding.

## Subgame Solving (Working — WarpGTO Lacks This)

Depth-limited real-time search:

1. **Subtree construction** (`Bulid_Tree.h`):
   - `bulid_subtree_turn2()`: Build turn/river subgame
   - `build_subtree_flop()`: Build flop subgame
   - Dynamically expands based on blueprint action nodes

2. **Blueprint initialization**:
   - On-tree actions: `subgame->regret[i] = node->averegret[i] / 10`
   - Off-tree actions: regrets initialized to 0 (fresh start)

3. **Leaf evaluation**:
   - `getnode_cfv_river()`: Exact hand comparison at river
   - `getnode_cfv_turn()`: Integrates over 46 possible turn/river cards
   - Parallelized with 100-thread pool

4. **Off-tree handling**: When real game produces actions not in blueprint, maps to closest node and initializes fresh — important for any real-time system.

## Range Solver

**None.** Subgame solving still uses abstraction. No exact 1326-combo solver exists.

## Neural Networks

**None.** Pure game-theoretic approach. No value networks or deep learning.

## Desktop App

Python/Tornado web interface — basic game visualization. Also has precompiled `.so` binaries for bot play on public platforms (Slumbot, OpenStack).

---

## Comparison Summary: DecisionHoldem vs WarpGTO

### WarpGTO Advantages
- **Abstraction quality**: Potential-aware EMD clustering vs opaque static lookup tables
- **Exact solving**: Range solver with full 1326-combo granularity
- **Configurability**: YAML configs, PioSOLVER notation, tunable DCFR params
- **Code quality**: Rust workspace, tested, documented, maintainable
- **CFVnet pipeline**: Deep learning infrastructure for depth-limited solving
- **Explorer UI**: Professional Tauri desktop app for strategy analysis
- **Info set design**: SPR-aware 64-bit packed keys

### DecisionHoldem Advantages
- **Working subgame solving**: Live depth-limited re-solving, operational today
- **Off-tree action handling**: Maps unforeseen actions to nearest blueprint node
- **CFR pruning (CFVP)**: Skips dominated actions after warmup period

### What to Learn From DecisionHoldem
1. **Subgame integration pattern**: build subtree → init from blueprint (÷10) → run focused CFR → serve at runtime
2. **Off-tree action mapping**: essential for real-time play
3. **CFR pruning**: could speed up blueprint training by skipping clearly dominated actions
