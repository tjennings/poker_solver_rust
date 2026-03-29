# Per-Street Pruning Control — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Add a `prune_streets` config option so pruning can be disabled per-street (e.g. skip preflop).

**Architecture:** New YAML field `prune_streets` parsed into a `[bool; 4]` bitmask. The mask flows from `trainer.rs` → `traverse_external` → per-node prune decision. `traverse_traverser` and `traverse_opponent` signatures unchanged.

**Tech Stack:** Rust, serde_yaml

---

### Task 1: Add `prune_streets` field to `TrainingConfig`

**Files:**
- Modify: `crates/core/src/blueprint_v2/config.rs:136-222` (TrainingConfig struct)
- Modify: `crates/core/src/blueprint_v2/config.rs:243-307` (default functions)

**Step 1: Write the failing test**

Add to `config.rs` `mod tests`:

```rust
#[test]
fn test_prune_streets_default() {
    // When prune_streets is omitted, all 4 streets should be enabled.
    let yaml = r#"
game:
  name: "Prune Default"
  players: 2
  stack_depth: 200.0
  small_blind: 1
  big_blind: 2

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

action_abstraction:
  preflop:
    - ["5bb"]
  flop:
    - [1.0]
  turn:
    - [1.0]
  river:
    - [1.0]

training:
  iterations: 100

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;
    let cfg: BlueprintV2Config =
        serde_yaml::from_str(yaml).expect("failed to parse config");
    assert_eq!(cfg.training.prune_street_mask(), [true; 4]);
}

#[test]
fn test_prune_streets_subset() {
    let yaml = r#"
game:
  name: "Prune Subset"
  players: 2
  stack_depth: 200.0
  small_blind: 1
  big_blind: 2

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

action_abstraction:
  preflop:
    - ["5bb"]
  flop:
    - [1.0]
  turn:
    - [1.0]
  river:
    - [1.0]

training:
  iterations: 100
  prune_streets: [flop, turn, river]

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;
    let cfg: BlueprintV2Config =
        serde_yaml::from_str(yaml).expect("failed to parse config");
    let mask = cfg.training.prune_street_mask();
    assert_eq!(mask, [false, true, true, true]);
}

#[test]
fn test_prune_streets_empty() {
    let yaml = r#"
game:
  name: "Prune Empty"
  players: 2
  stack_depth: 200.0
  small_blind: 1
  big_blind: 2

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

action_abstraction:
  preflop:
    - ["5bb"]
  flop:
    - [1.0]
  turn:
    - [1.0]
  river:
    - [1.0]

training:
  iterations: 100
  prune_streets: []

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/tmp/snapshots"
"#;
    let cfg: BlueprintV2Config =
        serde_yaml::from_str(yaml).expect("failed to parse config");
    assert_eq!(cfg.training.prune_street_mask(), [false; 4]);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p poker-solver-core test_prune_streets`
Expected: FAIL — `prune_streets` field and `prune_street_mask()` don't exist yet.

**Step 3: Write minimal implementation**

In `TrainingConfig` struct (after `baseline_alpha` field, ~line 221), add:

```rust
    /// Streets on which regret-based pruning is active.
    /// When omitted, all streets are pruned (backwards compatible).
    /// Example: `[flop, turn, river]` disables pruning on preflop.
    #[serde(default)]
    pub prune_streets: Option<Vec<String>>,
```

Add the `prune_street_mask` method as an `impl` block on `TrainingConfig`:

```rust
impl TrainingConfig {
    /// Convert the `prune_streets` config into a `[bool; 4]` mask indexed by
    /// `Street as usize`. When `prune_streets` is `None`, all streets enabled.
    pub fn prune_street_mask(&self) -> [bool; 4] {
        match &self.prune_streets {
            None => [true; 4],
            Some(streets) => {
                let mut mask = [false; 4];
                for s in streets {
                    match s.to_lowercase().as_str() {
                        "preflop" => mask[0] = true,
                        "flop" => mask[1] = true,
                        "turn" => mask[2] = true,
                        "river" => mask[3] = true,
                        other => panic!("unknown street in prune_streets: {other:?}"),
                    }
                }
                mask
            }
        }
    }
}
```

Also add `prune_streets: None,` to the `TrainingConfig` literal in the `test_serialize_round_trip` test (~line 448) so it compiles.

**Step 4: Run test to verify it passes**

Run: `cargo test -p poker-solver-core test_prune_streets`
Expected: 3 tests PASS.

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/config.rs
git commit -m "feat: add prune_streets config field with bitmask helper"
```

---

### Task 2: Thread `prune_streets` mask through `traverse_external`

**Files:**
- Modify: `crates/core/src/blueprint_v2/mccfr.rs:626-707` (traverse_external)
- Modify: `crates/core/src/blueprint_v2/trainer.rs:515-565` (batch loop)

**Step 1: Add `prune_streets` parameter to `traverse_external`**

In `mccfr.rs`, change the `traverse_external` signature (line 626) to add `prune_streets: [bool; 4]` after `prune_threshold`:

```rust
pub fn traverse_external(
    tree: &GameTree,
    storage: &BlueprintStorage,
    deal: &DealWithBuckets,
    traverser: u8,
    node_idx: u32,
    prune: bool,
    prune_threshold: i64,
    prune_streets: [bool; 4],   // NEW
    rng: &mut impl Rng,
    rake_rate: f64,
    rake_cap: f64,
    ev_tracker: Option<&ScenarioEvTracker>,
    full_ev_tracker: Option<&FullTreeEvTracker>,
    baseline_alpha: f64,
) -> (f64, PruneStats) {
```

**Step 2: Pass `prune_streets` through all recursive calls in `traverse_external`**

In the `Chance` arm (~line 648), add `prune_streets` to the recursive call:

```rust
        GameNode::Chance { child, .. } => {
            traverse_external(
                tree, storage, deal, traverser, *child, prune, prune_threshold,
                prune_streets, rng,
                rake_rate, rake_cap, ev_tracker, full_ev_tracker, baseline_alpha,
            )
        }
```

In the `Decision` arm (~line 666), compute `node_prune` and pass it to `traverse_traverser`/`traverse_opponent`, but pass the original `prune` is NOT needed since `traverse_traverser` and `traverse_opponent` call back into `traverse_external` which will recompute for each node:

```rust
            let node_prune = prune && prune_streets[street as usize];

            if player == traverser {
                traverse_traverser(
                    tree,
                    storage,
                    deal,
                    traverser,
                    node_idx,
                    bucket,
                    children,
                    num_actions,
                    node_prune,       // was: prune
                    prune_threshold,
                    prune_streets,    // NEW — needed for recursive traverse_external calls
                    rng,
                    rake_rate,
                    rake_cap,
                    ev_tracker,
                    full_ev_tracker,
                    baseline_alpha,
                )
            } else {
                traverse_opponent(
                    tree,
                    storage,
                    deal,
                    traverser,
                    node_idx,
                    bucket,
                    children,
                    num_actions,
                    node_prune,       // was: prune
                    prune_threshold,
                    prune_streets,    // NEW
                    rng,
                    rake_rate,
                    rake_cap,
                    ev_tracker,
                    full_ev_tracker,
                    baseline_alpha,
                )
            }
```

**Step 3: Update `traverse_traverser` and `traverse_opponent` to accept and forward `prune_streets`**

Both functions need `prune_streets: [bool; 4]` added to their signatures (after `prune_threshold`), and must forward it to their `traverse_external` recursive calls.

`traverse_traverser` (~line 779): add `prune_streets: [bool; 4]` after `prune_threshold` param. Update the recursive `traverse_external` call (~line 819) to pass `prune_streets`.

**Important**: In these child calls, pass `prune` (the original batch-level flag), NOT `node_prune`. The per-street filtering will be re-applied inside `traverse_external` for the child's street. But wait — `traverse_traverser` and `traverse_opponent` receive `node_prune` as their `prune` param. We need the original batch-level `prune` for recursion.

**Correction**: We should NOT modify the `prune` param passed to `traverse_traverser`/`traverse_opponent`. Instead, keep passing the original `prune` to them, and have them pass it through to `traverse_external` as before. The per-street gating happens ONLY in `traverse_external`'s Decision arm.

So the actual change in `traverse_external`'s Decision arm is simpler:

```rust
            let node_prune = prune && prune_streets[street as usize];

            if player == traverser {
                traverse_traverser(
                    // ... same args as before, but prune → node_prune ...
                    node_prune,
                    prune_threshold,
                    prune,            // original prune for child recursion
                    prune_streets,
                    // ... rest unchanged ...
                )
```

Wait — this adds complexity to `traverse_traverser`'s signature (it would need both `node_prune` and `prune_for_children`). Let's avoid that.

**Simplest approach**: Keep `traverse_traverser` and `traverse_opponent` signatures with `prune: bool` meaning "should I prune at THIS node". They forward `prune_streets` and the **original batch `prune`** to `traverse_external` calls, which recomputes `node_prune` for the child's street. But `traverse_traverser` doesn't have access to the original batch `prune`.

**Final simplest approach**: Don't modify the `prune` semantics at all. Just add `prune_streets` to all three functions. `traverse_traverser` and `traverse_opponent` receive `prune` (already per-node), use it for their local pruning decisions, and forward both `prune=true` (since we already decided to enter prune mode for the batch) and `prune_streets` to recursive `traverse_external` calls. Actually no — the `prune` they receive IS the batch-level flag and they use it directly.

Let me re-read the code flow:
1. `trainer.rs`: `let prune = self.should_prune()` — batch-level decision
2. Passed to `traverse_external(... prune ...)`
3. `traverse_external` forwards `prune` unchanged to `traverse_traverser`
4. `traverse_traverser` checks `if prune { ... }` at line 810

The cleanest fix: change the check in `traverse_traverser` line 810 from `if prune {` to check the mask. But `traverse_traverser` doesn't know the street.

**ACTUAL simplest approach (revised)**:
- `traverse_external` passes `node_prune` (not `prune`) to `traverse_traverser`/`traverse_opponent` as before
- `traverse_traverser`/`traverse_opponent` gain `prune_streets: [bool; 4]` and also a new `prune_orig: bool` param for forwarding to recursive `traverse_external` calls
- Their recursive `traverse_external` calls pass `prune_orig` as `prune`

No wait, that's ugly. Let me think again...

**CLEANEST approach**: `traverse_traverser` and `traverse_opponent` don't need the original batch `prune`. They just need `prune_streets` to forward. The `prune` they receive is the node-level decision. When they call `traverse_external` recursively, they should pass... hmm, they need the batch-level `prune` so `traverse_external` can AND it with the child's street mask.

OK here's the truly clean solution:

- `traverse_external` gains `prune_streets: [bool; 4]`
- In the `Decision` arm, compute `node_prune = prune && prune_streets[street as usize]`
- Pass `node_prune` as `prune` to `traverse_traverser`/`traverse_opponent`
- `traverse_traverser`/`traverse_opponent` gain `prune_streets: [bool; 4]`
- Their recursive `traverse_external` calls pass `prune=true` and `prune_streets` — since we're already in a pruning batch, the per-street mask handles the filtering
- BUT this is wrong: if `self.should_prune()` returned false, `prune=false` propagates correctly because `node_prune = false && ...` = false. So children need the original `prune` to preserve the "batch said no pruning" semantics.

**SIMPLEST CORRECT approach**: Don't touch `traverse_traverser`/`traverse_opponent` `prune` semantics at all. Just pass `prune_streets` through, and move the per-street gate INTO `traverse_traverser` by also passing `street`.

Actually, the design doc already solved this correctly:

> Pass `node_prune` (not `prune`) to `traverse_traverser` / `traverse_opponent`.
> Pass `prune` (original) through recursion so deeper streets still check their own mask.

This means `traverse_traverser` and `traverse_opponent` need BOTH: their local `prune` (= `node_prune`) AND the original `prune` for forwarding. That's two bools. Let me just rename for clarity: `prune` stays as the batch-level flag everywhere, and we add `prune_streets` everywhere. The node-level computation happens only in `traverse_external`'s Decision arm, computing `node_prune` and passing it as the local `prune` to `traverse_traverser`. For the recursive `traverse_external` calls from within `traverse_traverser`, we need the batch-level prune.

Fine — let's just add `prune_streets` to all signatures and keep `prune` as-is (batch level). The gate in `traverse_external` computes `node_prune` and passes it to `traverse_traverser`/`traverse_opponent`. Those functions use `prune` (which is now `node_prune`) for local decisions. For their recursive `traverse_external` calls, they need the batch prune. Since `node_prune` is a conservative approximation (if batch said no prune, node_prune is also false), passing `node_prune` as `prune` to recursive calls would incorrectly suppress pruning on the next street if the current street was skipped.

Example: batch prune=true, preflop excluded. At preflop Decision: `node_prune = true && false = false`. Passed to `traverse_traverser` as `prune=false`. `traverse_traverser` calls `traverse_external` with `prune=false`. Now at flop Decision: `node_prune = false && true = false`. Pruning disabled on flop too — WRONG.

So `traverse_traverser`/`traverse_opponent` MUST forward the **batch-level** `prune`, not `node_prune`. This means we need both.

**FINAL approach**: Add two things to `traverse_traverser`/`traverse_opponent`:
- Change nothing about `prune: bool` (stays as local prune decision)
- Add `prune_batch: bool` — the original batch flag, forwarded to recursive `traverse_external`
- Add `prune_streets: [bool; 4]` — forwarded to recursive `traverse_external`

Call sites in `traverse_external` Decision arm:
```rust
let node_prune = prune && prune_streets[street as usize];
traverse_traverser(..., node_prune, prune_threshold, prune, prune_streets, ...)
//                      ^local       ^threshold       ^batch ^mask
```

This is 100% correct but adds 2 params to already-long signatures.

Actually, `prune` is already the batch-level flag. Let me just keep it AS the batch-level flag in `traverse_traverser`/`traverse_opponent` and add the street-level gate there instead. But they don't know the street.

OK, **absolutely simplest correct approach**: just pass `prune_streets` to all functions. In `traverse_traverser`, the prune check (line 810) already gets `prune` as the batch flag. Just add the street to `traverse_traverser`'s params and gate there:

```rust
if prune && prune_streets[street as usize] {
```

This is what Approach A was. It's actually cleaner than B given the recursion requirements. Let me go with this:

1. `traverse_external`: add `prune_streets: [bool; 4]`, pass through unchanged
2. `traverse_traverser`: add `street: Street, prune_streets: [bool; 4]`, change line 810 to `if prune && prune_streets[street as usize] {`
3. `traverse_opponent`: add `prune_streets: [bool; 4]`, forward to recursive `traverse_external`
4. All `prune` semantics stay as batch-level — no dual-prune-flag complexity

This is simpler to implement and reason about. Updating the plan accordingly.

**Step 2: Verify compilation fails (shows all call sites)**

Run: `cargo build -p poker-solver-core 2>&1 | head -30`
Expected: FAIL — signature mismatches at all call sites.

**Step 3: Update all call sites**

Update `traverse_external` Decision arm to pass `street` and `prune_streets` to `traverse_traverser`:

```rust
            if player == traverser {
                traverse_traverser(
                    tree, storage, deal, traverser, node_idx, bucket, children,
                    num_actions, prune, prune_threshold, street, prune_streets,
                    rng, rake_rate, rake_cap, ev_tracker, full_ev_tracker, baseline_alpha,
                )
            } else {
                traverse_opponent(
                    tree, storage, deal, traverser, node_idx, bucket, children,
                    num_actions, prune, prune_threshold, prune_streets,
                    rng, rake_rate, rake_cap, ev_tracker, full_ev_tracker, baseline_alpha,
                )
            }
```

Update `traverse_traverser` prune check (line 810):
```rust
    if prune && prune_streets[street as usize] {
```

Update recursive `traverse_external` calls inside `traverse_traverser` (~line 819) and `traverse_opponent` (~line 926) to pass `prune_streets`.

Update `trainer.rs` batch loop (~lines 555-565) to compute and pass `prune_streets`:

```rust
let prune_streets = self.config.training.prune_street_mask();
```

Then pass `prune_streets` to both `traverse_external` calls.

**Step 4: Run tests**

Run: `cargo test -p poker-solver-core`
Expected: PASS (all existing tests + new config tests).

**Step 5: Commit**

```bash
git add crates/core/src/blueprint_v2/mccfr.rs crates/core/src/blueprint_v2/trainer.rs
git commit -m "feat: thread prune_streets mask through MCCFR traversal"
```

---

### Task 3: Full workspace build and test

**Files:** None (validation only)

**Step 1: Build full workspace**

Run: `cargo build`
Expected: PASS — no downstream crate breakage.

**Step 2: Run full test suite**

Run: `cargo test`
Expected: PASS in < 1 minute.

**Step 3: Update sample config**

Update `sample_configurations/blueprint_v2_200bkt_sapcfr.yaml` to show recommended setting:

```yaml
training:
  prune_streets: [flop, turn, river]
  prune_threshold: -200
```

**Step 4: Commit**

```bash
git add sample_configurations/blueprint_v2_200bkt_sapcfr.yaml
git commit -m "chore: update sample config with per-street pruning and -100bb threshold"
```
