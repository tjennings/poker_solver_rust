# Blueprint Trainer Lazy Tree Phase 0

Bean: `poker_solver_rust-6y86`

Date: 2026-06-03

## Conclusion

The heads-up `blueprint_v2` trainer is not primarily blocked by a missing arena traversal model. It already uses an eager arena-like `GameTree` with `Vec<GameNode>` and `u32` child indices. The real dense pressure point is CFR storage, snapshot/export identity, and compatibility surfaces that assume dense arena-order decision rows.

Phase 1 should therefore avoid a blind "replace map lookup" rewrite. The correct target is:

- Keep traversal on direct node handles.
- Introduce lazy child realization where it reduces up-front public-tree construction cost.
- Replace dense all-row CFR allocation with lazy row allocation for reached decision/bucket rows.
- Preserve dense export compatibility until Phase 2 proves equivalence.
- Treat persistent node/row identity as separate from in-process arena indices.

## Baseline Commands

Full default gate:

```bash
/usr/bin/time -p cargo test --quiet
```

Result after repairing the pre-existing runtime blocker:

```text
passed
real 40.44
user 132.05
sys 21.21
```

Before the blocker repair, the same gate passed but violated the project rule:

```text
cold cargo test:        real 162.70
warm cargo test --quiet: real 72.33
```

The repair moved two slow Tauri diagnostics behind explicit ignored-test runs and fixed an isolated Tauri temp-file test race. Relevant completed beans:

- `poker_solver_rust-v55b`
- `poker_solver_rust-esgz`

Focused HU toy trainer baseline:

```bash
/usr/bin/time -p cargo test -p poker-solver-core blueprint_v2::trainer::tests::train_runs_iterations --quiet
```

Result:

```text
1 passed
real 0.22
user 0.18
sys 0.07
```

MP lazy-sparse precedent baseline, recorded by the Phase 0 research pass:

```bash
/usr/bin/time -p cargo test -p poker-solver-core blueprint_mp -- --nocapture
```

Result:

```text
279 passed
real 7.48
```

MP lazy-sparse config inspection:

```bash
/usr/bin/time -p cargo run -p poker-solver-trainer -- inspect-mp-config \
  -c sample_configurations/blueprint_mp_6max_100bb_lazy_sparse_smoke.yaml
```

Result:

```text
real 2.45
6 players, 100.0bb, buckets 169/500/100/100, backend lazy_sparse
eager backend unsafe
```

MP lazy-sparse smoke training:

```bash
/usr/bin/time -p cargo run -p poker-solver-trainer -- train-blueprint-mp \
  -c sample_configurations/blueprint_mp_6max_100bb_lazy_sparse_smoke.yaml --no-tui
```

Result:

```text
real 3.63
1 meta-iteration
entries=225
regret_slots=599
strategy_slots=599
approx=311.1KB
traversals=6
batch_wall=9.0ms
buckets=6.3ms
traverse=1.8ms
prune=0.0%
```

For Phase 1 comparison runs, use deterministic variants where possible, especially `RAYON_NUM_THREADS=1`, because sparse-entry counts can vary when traversal order is parallel.

## Current HU Blueprint Shape

Tree:

- `crates/core/src/blueprint_v2/game_tree.rs`
- `GameTree` stores `nodes: Vec<GameNode>` and `root: u32`.
- Decision nodes store action vectors and child node indices.
- Action generation and deduplication live in the tree builder and must remain the action-order oracle.

Storage:

- `crates/core/src/blueprint_v2/storage.rs`
- `BlueprintStorage` allocates dense buffers for every `(decision node, bucket, action)` coordinate.
- Layout is based on per-node offsets and arena-order decision mapping.
- Regrets are scaled `i32`; strategy sums are integer strategy accumulators.

Export and resume:

- `crates/core/src/blueprint_v2/bundle.rs`
- `BlueprintV2Strategy` is dense and arena-order oriented.
- Existing `strategy.bin` consumers, including Explorer/Tauri, depend on dense decision order.
- Existing `regrets.bin` resume is dense and insufficiently versioned for a lazy sparse backend.

Trainer:

- `crates/core/src/blueprint_v2/trainer.rs`
- Trainer owns config, eager tree, dense storage, iteration counters, discount cadence, snapshot/export, and TUI callbacks.
- TUI scenarios and audits resolve against current tree/storage identities, so Phase 1 must preserve compatibility or add explicit adapters.

MP precedent:

- `crates/core/src/blueprint_mp/lazy_mccfr.rs`
- `crates/core/src/blueprint_mp/sparse_storage.rs`
- MP already has lazy public states, sparse visited-infoset storage, missing-row uniform semantics, and sparse snapshot entries.
- Do not copy the MP sharded `HashMap` as the HU hot-path model unless Phase 1 proves the locality tradeoff is acceptable. Treat it as a semantics precedent, not an implementation mandate.

## Required Identities

Phase 1 must separate three identities.

`NodeId`:

- In-process arena handle.
- Cheap direct traversal coordinate.
- May be allocation-order dependent.
- Must not be the persisted compatibility identity.

`PublicNodeKey`:

- Stable public-state identity.
- Should encode street, actor, pot, stacks, street contributions/bets, facing-bet state, raise count, last raise-to, terminal/chance boundary kind, board/chance identity when relevant, and packed action-history/hash.
- Used for diagnostics, differential harness output, snapshot metadata, and possible future subtree paging.

`RowKey`:

- CFR storage identity.
- Should not be raw `NodeId`.
- Candidate shape: public-node digest or deterministic decision ordinal, acting player, street, bucket, action-count schema/fingerprint.
- `action_idx` remains the coordinate inside a row and is only valid with the action schema fingerprint.

## Phase 1 Invariants

Legal actions:

- Action order must match the current eager HU generator byte-for-byte.
- Preserve fold, call/check, configured size order, all-in insertion, and deduplication behavior.
- `TreeAction` labels are display, not identity.
- Differential failures must print public key, action list, action index, child kind, bucket, and row key.

Traversal:

- Traverser decisions still evaluate all eligible actions.
- Opponent sampling must use the same RNG stream and current strategy vector under fixed seeds.
- Lazy child realization must be idempotent.
- Concurrency must not create duplicate children or attach rows to different action schemas.

Storage:

- Missing lazy rows mean zero cumulative regret and zero strategy sum.
- Current strategy for a missing row is uniform over legal actions.
- Average strategy for a missing row is uniform unless an explicit export policy says otherwise.
- Preserve HU regret and strategy-sum numeric types unless a separate compatibility bean changes them.
- DCFR discounting must not double-discount or skip already touched rows. Rows first created after a discount boundary need no historical discount because their implicit history was zero.

Buckets:

- Bucket lookup must match current `AllBuckets` behavior for preflop, flop, turn, and river.
- Per-flop bucket lookup and fallback behavior must stay identical.
- The differential harness must include bucket key and bucket source in mismatch output.

Terminal values:

- Fold, showdown, side-pot, rake, and all-in terminal values must match current code exactly.
- Lazy realization must preserve terminal kind and contribution state before utility computation.

Snapshots and export:

- Do not silently load dense `regrets.bin` into lazy storage unless tree fingerprint, action schema fingerprint, bucket counts, and dense decision order all match.
- Add a versioned lazy resume file if Phase 1 introduces lazy storage, for example `lazy_regrets_v1.bin`.
- Lazy snapshot rows should be sorted deterministically.
- Dense `strategy.bin` export must remain available for Explorer/Tauri, either by materializing deterministic dense decision order on export or by routing old consumers through a compatibility adapter.
- Old dense snapshots should either load through the old backend or fail with a specific incompatible-backend error. Silent partial conversion is unacceptable.

Telemetry:

Phase 1 should report:

- realized nodes
- realized decision nodes
- realized children / realization misses
- touched rows
- allocated regret slots
- allocated strategy slots
- approximate arena bytes
- approximate storage bytes
- dense-equivalent slot count
- snapshot row count and bytes

## Differential Harness Requirements

The Phase 1 prep harness (`poker_solver_rust-zgkr`) should compare old and new paths under a fixed tiny HU fixture:

- legal actions at each visited public state
- terminal kind and terminal payoff
- chance transitions
- bucket lookup
- sampled opponent action under fixed RNG
- regret deltas
- strategy-sum deltas
- exported average strategy
- snapshot save/load round trip

The harness should fail loudly with public-state/action-row diagnostics, not just a scalar "strategy differs".

## Compatibility Policy

Phase 1 is correctness-preserving only:

- no strategy pruning
- no disk eviction
- no action abstraction change
- no irreversible row deletion
- no lossy snapshot conversion

SAPCFR+/BRCFR+/baseline buffers are not automatically compatible with lazy rows. If those paths depend on dense vector length equality or dense prediction/baseline layouts, Phase 1 must either keep them on the dense backend or add explicit lazy side-buffer support with tests.

## Open Phase 1 Decisions

- Whether HU lazy child realization should keep a full deterministic decision ordinal map for dense export, or build it only during export.
- Whether lazy rows should be keyed by public-node digest or by a deterministic decision ordinal generated from public keys.
- Whether the initial implementation keeps eager tree construction and only lazifies row allocation first. This may be the lowest-risk first cut because HU already has arena traversal.
- Whether dense snapshot loading remains old-backend-only in Phase 1.

## Go/No-Go For Phase 1

Go:

- Full default suite passes under 60 seconds.
- Phase 0 baseline and invariants are recorded here.
- Phase 1 prep harness bean is unblocked.

No-go:

- Any implementation that combines lazy storage with pruning or disk eviction.
- Any persisted format that uses allocation-order `NodeId` as its sole durable identity.
- Any export change that breaks Explorer/Tauri dense `strategy.bin` consumers without an adapter.
