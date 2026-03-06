# Range Solver Design

**Date**: 2026-03-06
**Status**: Approved

## Goal

Reimplement b-inary/postflop-solver as a self-contained crate (`range-solver`) in the poker_solver_rust workspace. Output must be **identical** to the original for any given configuration. Performance must be comparable (within 1.5x).

A `range-solve` CLI subcommand exposes the solver. A separate comparison crate validates output identity against the original across 1000 random configurations.

## Architecture

### Crate Layout

```
crates/range-solver/
  src/
    lib.rs              # Public API surface
    card.rs             # Card encoding (4*rank+suit), deck, hand representation
    range.rs            # PioSOLVER-format range string parsing
    bet_size.rs         # Bet sizing: pot%, geometric, all-in, raise multipliers
    action_tree.rs      # Abstract betting tree construction
    game.rs             # PostFlopGame: tree building, memory, node arena
    node.rs             # PostFlopNode: player, amounts, children
    evaluation.rs       # Terminal payoff computation, hand strength sorting
    isomorphism.rs      # Suit symmetry detection, card/hand swap tables
    solver.rs           # DCFR iteration loop, exploitability calculation
    mutex_like.rs       # Lock-free single-thread / Mutex multi-thread wrapper
    interface.rs        # Game/GameNode traits

crates/range-solver-compare/
  src/lib.rs            # Comparison harness: generate configs, run both, diff
  tests/identity.rs     # 1000-config identity test
  Cargo.toml            # Depends on range-solver + postflop-solver (path dep)
```

### Self-Contained Design

Zero dependency on `core` crate. Own card encoding, range parsing, evaluation, etc. This ensures full control over float accumulation order, tree traversal, and isomorphism tables -- all critical for output identity.

Module boundaries mirror the original's file structure to simplify correctness auditing.

## Algorithm

### Discounted CFR

- alpha=1.5, beta=0.5, gamma=3.0 (original uses 3.0, not the paper's 2.0)
- Strategy reset: cumulative strategy zeroed when iteration count is a power of 4
- Regret discounting: negative regrets by t^alpha/(t^alpha+1), positive by t^beta/(t^beta+1)
- Strategy sum weighting: (t/(t+1))^gamma
- Multithreaded via rayon (parallelize over root actions)

### Card Encoding

`card_id = 4 * rank + suit` where rank 0=deuce..12=ace, suit 0-3. Must match original exactly.

### Game Tree Storage

- Arena-allocated `Vec<MutexLike<PostFlopNode>>`
- Three buffers per node: strategy, regrets/cfvalues, IP cfvalues
- Support f32 (uncompressed) and i16+scale (compressed) modes
- Node size target: 36 bytes

### Isomorphism

- Turn/river suit symmetry grouping
- Swap tables for card and hand indices when mapping to reference suit
- Must produce identical groupings to original

### Hand Evaluation

- Port the original's internal hand evaluator exactly
- Per-board hand ranking with sorted strength arrays for O(n) terminal evaluation
- Identical relative ordering required

## CLI Interface

```
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs,AKo" \
  --ip-range "QQ-22,AKs-A2s,KQs" \
  --flop "Qs Jh 2c" \
  --turn "8d" \
  --river "3s" \
  --pot 100 \
  --effective-stack 900 \
  --iterations 1000 \
  --target-exploitability 0.5 \
  --bet-sizes config.yaml \
  --compressed
```

Also supports `--config solve.yaml` for complex bet size trees.

### Output Format

Matches original for direct comparison:
- Per-iteration: `iteration: N / total (exploitability = X.XXXXeY)`
- Final: exploitability, memory usage
- Root strategy dump: per-hand action probabilities
- Per-hand EV and equity at root

### Programmatic API

- `solve()` returns exploitability as f32
- `game.strategy()` -> `&[f32]` flat array (actions x hands)
- `game.expected_values(player)` -> `&[f32]`
- `game.equity(player)` -> `&[f32]`

## Comparison Harness

### Config Generation

`generate_random_configs(n: usize, seed: u64) -> Vec<TestConfig>`:
- Deterministic RNG (seed=42), reproducible
- Random ranges (tight 5%, medium 30%, wide 60%)
- Random flops, optional turn/river
- Random pot (20-500), stacks (100-2000)
- Random bet sizes from common pool (33%, 50%, 67%, 100%, 150%, all-in)
- Street mix: ~40% flop-only, ~30% flop+turn, ~30% all three streets

### Identity Test

For each of 1000 configs:
1. Build equivalent CardConfig+TreeConfig for both solvers
2. Run both to 200 iterations
3. Assert exact f32 equality on: exploitability, strategy arrays, expected_values, equity
4. On mismatch: dump config + both outputs for debugging

### Performance Benchmark

- 10 representative configs at 1000 iterations
- Compare wall-clock time
- Assert within 1.5x of original

### Runtime Target

1000-game identity suite completes in under 5 minutes (release mode).

### Decoupling

Original postflop-solver referenced via path dependency in range-solver-compare only. Easy to remove: delete the crate or swap to a serialized-output comparison.
