# Blueprint MP 100bb Lazy/Sparse Design

## Problem

Blueprint MP currently builds the complete public game tree before training and allocates dense storage for every `(node, bucket, action)` slot. That works for shallow or very compact abstractions, but it does not scale to normal 100bb 6-max play with multiple preflop raise depths.

The reproduced 100bb 6-max 500/100/100 config with two preflop raise-depth rows produced:

```text
MP Tree: 272,170,499 nodes
MP Storage: 28,399,883,894 slots
virtual storage: 340.8 GB
```

This is not a parsing bug. It is an eager tree plus dense storage architecture limit.

## Goal

Support 100bb 6-max Blueprint MP training without up-front allocation proportional to every public node, street bucket, and action in the entire game tree.

The first production target is:

- 6 players, 100bb, no ante or BB ante
- 169 preflop buckets, postflop bucket configs such as 500/100/100
- Multiple preflop raise-depth rows
- Existing external-sampling MCCFR semantics
- TUI/no-TUI progress that does not full-scan unvisited storage
- Snapshot/resume for the new sparse format

## Current Architecture Limits

Current setup:

```text
BlueprintMpConfig
  -> MpGameTree::build(...)
       builds Vec<MpGameNode> for every public state
  -> MpStorage::new(tree, bucket_counts)
       allocates regrets:       node * bucket * action AtomicI32
       allocates strategy_sums: node * bucket * action AtomicU64
  -> traverse_external(tree, storage, node_idx, ...)
       recursive traversal by node index
```

The most important coupling is not just the arena tree. It is that `MpStorage` uses `node_idx` as the identity for a decision. Any 100bb-capable design must replace this with a stable infoset key that can be generated during traversal.

## Chosen Direction

Use lazy public-state traversal plus sparse visited-infoset storage.

Keep eager mode for tests, small configs, parity, and debugging. Add a second backend for large configs:

```yaml
training:
  backend: lazy_sparse   # default can remain eager initially
```

The lazy backend does not build `MpGameTree`. It traverses a compact public state, generates legal actions on demand, applies selected actions to produce the next state, and stores regrets only for infosets actually visited.

## Core Interfaces

### Public State And Rules

Extract the reusable parts of `game_tree.rs` into state/rules APIs:

```rust
pub struct MpPublicState {
    stacks: [Chips; MAX_PLAYERS],
    street_bets: [Chips; MAX_PLAYERS],
    contributions: [Chips; MAX_PLAYERS],
    active: PlayerSet,
    all_in: PlayerSet,
    acted_since_aggression: PlayerSet,
    street: Street,
    pot: Chips,
    num_players: u8,
    raise_count: u8,
    to_act: Seat,
    facing_bet: bool,
    last_raise_to: Chips,
    dealer: u8,
    big_blind_amount: Chips,
    action_history: SmallVec<[ActionCode; 32]>,
}

pub struct MpRules {
    game: MpGameConfig,
    action_abstraction: ParsedMpActionAbstraction,
}

impl MpRules {
    pub fn initial_state(&self) -> MpPublicState;
    pub fn legal_actions(&self, state: &MpPublicState, out: &mut SmallVec<[TreeAction; 16]>);
    pub fn apply_action(&self, state: &MpPublicState, action: TreeAction) -> MpStep;
}

pub enum MpStep {
    Decision(MpPublicState),
    Chance { next_street: Street, state: MpPublicState },
    Terminal(MpTerminalState),
}
```

`MpGameTree::build` should eventually call the same rules/action code, so eager and lazy modes cannot diverge.

### Stable Infoset Key

Current `InfoKey128` is close but too rigid as the only long-term key:

- It includes seat, bucket, street, SPR bucket, and action history.
- It has 22 four-bit action slots.
- It panics on overflow.

For 100bb we should introduce an explicit MP storage key:

```rust
pub struct MpInfosetKey {
    seat: Seat,
    street: Street,
    bucket: u16,
    spr_bucket: u8,
    history_hi: u64,
    history_lo: u64,
    history_hash: u64,
    history_len: u16,
}
```

`InfoKey128` can remain the fast-path packed representation when `history_len <= 22`. The sparse backend should not panic if history grows. For longer histories, it should include a stable hash of the full encoded action history.

Action history encoding should use semantic action codes:

- fold
- check
- call
- all-in
- lead size row/index
- raise depth/index

Do not encode floating-point chip amounts directly into the key. Amounts are deterministic from config plus state; the key should encode the abstraction choice that produced the amount.

### Sparse Storage

Add a storage trait to decouple MCCFR from dense `node_idx` layout:

```rust
pub trait MpCfrStorage {
    type Key: Copy + Eq + Hash;

    fn regret_matched_strategy(
        &self,
        key: Self::Key,
        num_actions: usize,
        out: &mut [f64],
    );

    fn get_regret(&self, key: Self::Key, action: usize) -> i32;
    fn add_regret(&self, key: Self::Key, action: usize, delta: i32);
    fn add_strategy_sum(&self, key: Self::Key, action: usize, delta: i32);
    fn average_strategy(&self, key: Self::Key, num_actions: usize, out: &mut [f64]);
}
```

Implementations:

- `DenseMpStorageAdapter`: wraps current `MpStorage` for eager mode.
- `SparseMpStorage`: sharded map keyed by `MpInfosetKey`.

Sparse entry shape:

```rust
pub struct SparseNode {
    num_actions: u8,
    regrets: Box<[AtomicI32]>,
    strategy_sums: Box<[AtomicU64]>,
}
```

Because `bucket` is part of `MpInfosetKey`, a sparse node only needs `num_actions` counters, not `bucket_count * num_actions`. Unvisited infosets behave as all-zero regrets and all-zero strategy sums, which yields uniform current/average strategy.

Use sharding for concurrency:

```text
SparseMpStorage
  shards[256]: Mutex<HashMap<MpInfosetKey, Arc<SparseNode>>>
```

This is simple and good enough for the first implementation. If lock contention becomes visible, move to a concurrent hash map or per-thread local deltas with merge.

## Lazy MCCFR

Add a lazy traversal beside the current eager traversal:

```rust
pub fn traverse_external_lazy(
    rules: &MpRules,
    storage: &SparseMpStorage,
    deal: &DealWithBuckets,
    traverser: Seat,
    state: &MpPublicState,
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: Chips,
    prune: bool,
    prune_threshold: i32,
) -> (f64, PruneStats)
```

At a decision:

1. Generate legal actions into a stack buffer.
2. Compute current player's bucket for the state's street.
3. Build `MpInfosetKey` from state, seat, street, bucket, SPR bucket, and action history.
4. Read regret-matched strategy from sparse storage.
5. If traverser, evaluate every action recursively and update regrets.
6. If opponent, sample one action and update strategy sum.

Chance transitions should not allocate nodes. They advance the public street; board cards already live in `DealWithBuckets`.

Terminal values can reuse the existing terminal payoff code by converting `MpTerminalState` to the current terminal inputs.

## Discounting And Pruning

Dense DCFR discount currently scans every regret and strategy slot. Sparse mode should scan only visited entries.

That changes the meaning from "discount every possible infoset" to "discount every reached infoset." For zero/unvisited infosets, discounting has no effect, so this is equivalent for sparse storage.

Pruning keeps the same logic:

- At traverser nodes, skip non-terminal child actions when regret is below threshold.
- Unvisited action regret is zero, so it will not be pruned by a negative threshold.

## Snapshots

Sparse snapshots need a new format. Do not force sparse state through `BlueprintV2Strategy`, because that assumes dense ordered decision nodes.

Proposed snapshot files:

```text
snapshot_NNNN/
  strategy_sparse.bin      # iterable sparse average strategy entries
  regrets_sparse.bin       # sparse regrets and strategy sums for resume
  metadata.json            # backend, iterations, elapsed, config hash, bucket counts
```

Each sparse entry records:

```rust
struct SparseSnapshotEntry {
    key: MpInfosetKey,
    num_actions: u8,
    regrets: Vec<i32>,
    strategy_sums: Vec<u64>,
}
```

For explorer/export compatibility, add a separate materialization command later:

```text
export-mp-strategy --snapshot snapshot_NNNN --spots scenario.yaml
```

This can materialize only requested spots or sampled diagnostics instead of the entire impossible 100bb tree.

## TUI And Diagnostics

TUI scenarios currently resolve to eager node ids. Lazy mode should resolve scenario strings into `MpPublicState` instead.

Telemetry rules:

- No full sparse-map scans on the render cadence.
- Sample sparse entries for regret health.
- Track created infosets, visited entries, and storage bytes as first-class metrics.
- Keep backpressure so only one telemetry job can run at a time.

## Preflight

Before implementing the lazy backend, add a sizing preflight for eager mode:

```text
cargo run -p poker-solver-trainer --release -- inspect-mp-config \
  --config sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml
```

Output should include:

- effective stack in BB
- action rows per street
- eager tree node count if exact build is allowed, or an estimate if capped
- dense storage slots and virtual bytes
- recommendation: eager or lazy_sparse

Training should fail fast in eager mode when estimated dense storage exceeds a configurable cap.

## Migration Plan

1. **Preflight and guardrails**
   - Add config inspection/sizing output.
   - Add dense storage cap with actionable error text.
   - Keep current trainer behavior for small configs.

2. **Extract rules/state from eager tree**
   - Make `MpPublicState`, parsed action abstraction, legal action generation, state transition, and terminal construction reusable.
   - Update `MpGameTree::build` to call those shared APIs.
   - Add parity tests between old eager transitions and extracted rules.

3. **Introduce storage trait**
   - Add `MpCfrStorage` or equivalent.
   - Adapt eager traversal to use the trait where practical.
   - Keep dense `MpStorage` behavior unchanged.

4. **Implement sparse storage**
   - Sharded sparse map keyed by `MpInfosetKey`.
   - Uniform strategy for absent keys.
   - Sparse discount over visited entries.
   - Snapshot/resume sparse counters.

5. **Implement lazy traversal**
   - New `run_training_lazy_sparse`.
   - No `MpGameTree::build` in lazy setup.
   - Use dynamic actions and sparse storage.

6. **TUI and snapshot integration**
   - Resolve scenarios to public states in lazy mode.
   - Sample sparse telemetry.
   - Add sparse snapshot save/resume.

7. **100bb regression gate**
   - Add a smoke config with 100bb 6-max and multiple preflop raise depths.
   - Verify setup does not allocate dense storage.
   - Verify a small iteration run advances and writes heartbeats.

## Acceptance Criteria

- A 100bb 6-max config with two preflop raise-depth rows starts in lazy sparse mode without eager tree allocation.
- Initial memory use is proportional to visited infosets, not total public tree size.
- No-TUI heartbeat reports iters/sec, created sparse infosets, sampled regret stats, and prune percentage.
- TUI mode does not launch overlapping full-storage scans.
- Sparse snapshot/resume round trip preserves strategy and regret counters for visited infosets.
- Eager mode tests remain green for current small-tree behavior.

