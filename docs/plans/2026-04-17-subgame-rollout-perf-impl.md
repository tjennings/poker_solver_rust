# Subgame Rollout Perf + Hands/Sec Telemetry — Implementation Plan

> **Status: superseded (2026-04-18).** Pivoted to depth-gated MCCFR sampling after ml-researcher
> findings — the original plan's per-hand allocation fixes would yield ~5-10x while the sampling
> pivot yielded 164x. See `2026-04-18-subgame-rollout-sampling-impl.md`.

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Close the rollout-throughput gap to MCCFR's ~100k hands/sec and add live hands/sec telemetry to the Tauri progress bar.

**Architecture:** Six atomic commits on a single feature branch, all in one PR. Commit 1 lands the measurement surface (global atomic counter + `PostflopProgress.rollout_hands_per_sec` + frontend display) so commits 2–6 can each record a before/after delta. Commits 2–6 target one optimization each, never touching MCCFR or range-solver.

**Tech Stack:** Rust (core, tauri-app), TypeScript (frontend), SmallVec for stack-allocated action scratch, rayon for parallel iteration.

**Reference docs:**
- Design doc: `docs/plans/2026-04-17-subgame-rollout-perf-design.md`
- MCCFR mirror pattern: `crates/core/src/blueprint_v2/mccfr.rs:76-80` (`DealWithBuckets`), `:914-921` (scratch-array sizing)

---

## Setup

### Task S1: Create feature branch and worktree

**REQUIRED SUB-SKILL:** hex:using-git-worktrees

**Step 1: Create worktree**

```bash
cd /Users/coreco/code/poker_solver_rust
git worktree add ../poker_solver_rust.rollout-perf -b perf/subgame-rollout
cd ../poker_solver_rust.rollout-perf
```

**Step 2: Confirm baseline green**

```bash
cargo test --all --quiet
```

Expected: all pass in under 1 minute. If tests exceed 1 minute, stop and report — CLAUDE.md rule.

**Step 3: Capture a baseline hands/sec number by hand (manual)**

Run a representative subgame solve via the dev server or Tauri (whichever is fastest) and eyeball the current throughput. We need a single anchor number before we land commit 1. Note it in the PR description draft.

---

## Commit 1 — `feat(postflop): add rollout hands/sec telemetry`

Adds the measurement surface. No perf fixes in this commit.

### Task 1.1: Add `rollout_hands` atomic counter to `PostflopState`

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:889-912` (struct `PostflopState`)
- Modify: `crates/tauri-app/src/postflop.rs:914-935` (`impl Default for PostflopState`)

**Step 1: Add field**

Append to the struct (after `subgame_node`, near line 904):

```rust
    /// Total rollout terminals reached across the current solve. Reset when
    /// `solve_start` is reset. Used for hands/sec telemetry.
    pub rollout_hands: AtomicU64,
```

**Step 2: Initialize in `Default`**

Add in the `Default` block (near line 930):

```rust
            rollout_hands: AtomicU64::new(0),
```

**Step 3: Verify import**

Ensure `AtomicU64` is imported at the top of `postflop.rs`. If only `AtomicU32/AtomicBool` are present, add to the existing `use std::sync::atomic::...` line.

**Step 4: Compile check**

```bash
cargo check -p tauri-app
```

Expected: PASS.

### Task 1.2: Add `rollout_hands_per_sec` field to `PostflopProgress`

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:121-131` (struct `PostflopProgress`)

**Step 1: Add field**

Insert after `elapsed_secs`:

```rust
    /// Rate of rollout terminals reached per second (averaged over solve lifetime).
    /// 0.0 when no rollout evaluator is active or solve has not started.
    pub rollout_hands_per_sec: f32,
```

**Step 2: Populate in `postflop_get_progress_core`**

Modify `crates/tauri-app/src/postflop.rs:1519-1542`. After the `elapsed_secs` let-binding, add:

```rust
    let rollout_hands = state.rollout_hands.load(Ordering::Relaxed);
    let rollout_hands_per_sec = if elapsed_secs > 0.0 {
        (rollout_hands as f64 / elapsed_secs) as f32
    } else {
        0.0
    };
```

Include it in the returned struct.

**Step 3: Reset counter at solve start**

At `crates/tauri-app/src/postflop.rs:1253-1258` (the "Reset progress atomics" block inside `postflop_solve_street_impl`), add:

```rust
    state.rollout_hands.store(0, Ordering::Relaxed);
```

### Task 1.3: Increment counter in rollout terminal path

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs:143-154` (`rollout_from_boundary` signature)
- Modify: `crates/core/src/blueprint_v2/continuation.rs:165-270` (`rollout_inner`)
- Modify: `crates/tauri-app/src/postflop.rs:468-477` (call site)

The counter lives in `tauri-app` but the increment must happen at the rollout terminal. Pass an optional `&AtomicU64` reference through `RolloutContext` (add a new field `hand_counter: Option<&'a AtomicU64>`). MCCFR callers pass `None`; the Tauri path passes `Some(&state.rollout_hands)`.

**Step 1: Add field to `RolloutContext`**

At `crates/core/src/blueprint_v2/continuation.rs:113-125`, add a new field:

```rust
    /// Optional counter incremented once per rollout terminal reached.
    /// Used for hands/sec telemetry. `None` disables the counter (no overhead).
    pub hand_counter: Option<&'a std::sync::atomic::AtomicU64>,
```

**Step 2: Increment at terminal nodes**

At `crates/core/src/blueprint_v2/continuation.rs:177-203` (the `GameNode::Terminal` arm of `rollout_inner`), at the top of the arm add:

```rust
        if let Some(counter) = ctx.hand_counter {
            counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
```

**Step 3: Update Tauri call site**

At `crates/tauri-app/src/postflop.rs:452-462` where `RolloutContext` is constructed, add:

```rust
    hand_counter: Some(&self.hand_counter_ref),
```

Add a new field `hand_counter_ref: Arc<AtomicU64>` to `RolloutLeafEvaluator` (struct defined near line 360, constructor at 378-403). Initialize it from `state.rollout_hands`: thread an `Arc<AtomicU64>` through `build_subgame_solver` (inspect signature; likely needs a new param). If that's invasive, simplest alternative: add a `with_hand_counter(&mut self, c: Arc<AtomicU64>)` setter and call it right after construction in `solve_depth_limited`.

Use whichever wiring is smaller — read the builder first before committing to either.

**Step 4: Update any other `RolloutContext` construction sites**

```bash
grep -rn "RolloutContext {" crates/
```

Any sites not in the Tauri eval loop (tests, benches, CLI) should pass `hand_counter: None`.

**Step 5: Full compile**

```bash
cargo build --all
```

Expected: PASS.

### Task 1.4: Frontend display

**Files:**
- Modify: `frontend/src/game-types.ts` (`SolveStatus` interface)
- Modify: `frontend/src/GameExplorer.tsx` (wherever the progress bar renders iteration/elapsed)

**Step 1: Extend the TS interface**

At `frontend/src/game-types.ts:11-18`:

```typescript
export interface SolveStatus {
  iteration: number;
  max_iterations: number;
  exploitability: number;
  elapsed_secs: number;
  rollout_hands_per_sec: number;
  solver_name: string;
  is_complete: boolean;
}
```

**Step 2: Display next to iteration counter**

In `GameExplorer.tsx`, find where iteration and elapsed are rendered in the solve progress readout (search for `elapsed_secs` or `iteration` display). Add, when `rollout_hands_per_sec > 0`:

```tsx
<span>{Math.round(status.rollout_hands_per_sec).toLocaleString()} hands/s</span>
```

Match existing styling. Mirror MCCFR's `{it/s:.0} it/s` visual.

**Step 3: Typecheck + dev-run spot-check**

```bash
cd frontend && npm run typecheck
```

Expected: PASS.

Bring up the dev server (per CLAUDE.md):
```bash
cargo run -p poker-solver-devserver &
cd frontend && npm run dev
```

Start a subgame solve in the browser. Confirm the hands/sec number appears and is non-zero while the solve is running.

### Task 1.5: Tests

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:2203-2220` (existing `test_get_progress_before_solve`)
- Add: new test `test_get_progress_resets_rollout_counter`

**Step 1: Extend `test_get_progress_before_solve`**

Add an assertion that `progress.rollout_hands_per_sec == 0.0`.

**Step 2: Add reset test**

```rust
#[test]
fn test_get_progress_resets_rollout_counter() {
    let state = PostflopState::default();
    state.rollout_hands.store(1234, std::sync::atomic::Ordering::Relaxed);
    // Simulate solve start by setting solve_start and zeroing counter
    *state.solve_start.write() = Some(std::time::Instant::now());
    state.rollout_hands.store(0, std::sync::atomic::Ordering::Relaxed);
    // Simulate some hands
    state.rollout_hands.store(500, std::sync::atomic::Ordering::Relaxed);
    // Ensure a measurable elapsed_secs (sleep briefly)
    std::thread::sleep(std::time::Duration::from_millis(50));
    let progress = postflop_get_progress_core(&state);
    assert!(progress.rollout_hands_per_sec > 0.0);
}
```

**Step 3: Run**

```bash
cargo test -p tauri-app --quiet test_get_progress
```

Expected: PASS.

### Task 1.6: Verify and commit

**Step 1: Full test suite**

```bash
cargo test --all --quiet
time cargo test --all --quiet  # confirm under 60s
```

Expected: all PASS, elapsed under 1 minute.

**Step 2: Commit**

```bash
git add -A
git commit -m "feat(postflop): add rollout hands/sec telemetry"
```

**Step 3: Capture baseline hands/sec**

Run a representative subgame solve, record the steady-state `rollout_hands_per_sec` in the PR description draft as the **baseline** row. This is the number every subsequent commit must beat.

---

## Commit 2 — `perf(postflop): hoist opponent weights out of per-combo loop`

Addresses suspect #1. Opponent weights only depend on opponent range and hero-combo overlap. Hero-overlap filtering can happen inside the combo worker; the base weights table is shared.

### Task 2.1: Refactor `rollout_chip_values_with_state`

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:411-484`

**Step 1: Write the failing test first**

Add under existing tests (or a new `#[cfg(test)] mod rollout_weights_tests`):

```rust
#[test]
fn test_hoisted_opponent_weights_match_per_combo() {
    // Construct a minimal RolloutLeafEvaluator with a trivial tree and
    // two combos. Compute rollout chip values with the old code path
    // (inline weight vec) and the new code path (hoisted). Assert
    // per-combo results are bit-identical.
}
```

You'll need to inline the old loop into the test as a reference implementation, or snapshot current outputs as a golden first (run with current code, capture, then refactor).

Simpler path: use **snapshot testing** — run the existing rollout on a small fixture, capture outputs, refactor, assert outputs unchanged. Use `insta` if present; else just write golden bytes to a vec and compare.

**Step 2: Refactor** — move the base weights table outside the `par_iter`:

```rust
// Compute opponent-weight base table once — does not depend on hero combo.
// Hero-hand overlap filter still happens per-combo inside the worker.
let opp_base_weights: Vec<f64> = opp_range.iter().copied().collect();

combos
    .par_iter()
    .enumerate()
    .map(|(i, hero_hand)| {
        if hero_range[i] <= 0.0 {
            return 0.0;
        }

        // Zero out overlapping opponent combos, in a reused scratch Vec.
        // par_iter closure captures — each thread gets its own clone of the
        // base table to mutate. Heap alloc still happens but is per-combo,
        // same allocation count as before... so this is NOT the final fix.
        // See step 3.
        ...
```

Wait: naive par_iter still allocates. The real win comes from **separating invariant opponent range from hero-overlap filtering**:
- Opponent base weights (`opp_range.clone()`): allocated **once** before `par_iter`.
- Hero-overlap zeros: applied **in-place on a per-worker scratch vec** when we introduce the scratch struct in commit 5. For now in commit 2, we can use `sample_weighted_filtered` that takes `&[f64]` + a hero-hand and skips overlaps during sampling without materializing a filtered vec.

**Step 3: Introduce `sample_weighted_filtered`**

Check if `sample_weighted` lives in a helper module. If there's a version that takes a skip-predicate or we can add one:

```rust
/// Like `sample_weighted` but skips indices j where `skip(j)` is true.
fn sample_weighted_filtered<F: Fn(usize) -> bool>(
    rng: &mut impl Rng,
    weights: &[f64],
    count: u32,
    skip: F,
) -> Vec<usize> { ... }
```

Use it in the combo worker:

```rust
let sampled = sample_weighted_filtered(
    &mut rng,
    opp_range,
    self.num_opponent_samples,
    |j| opp_range[j] <= 0.0 || cards_overlap(*hero_hand, combos[j]),
);
```

This eliminates the `Vec<f64>` allocation entirely — overlap is checked lazily during sampling.

**Step 4: Remove the old per-combo `weights: Vec<f64>` block** (lines 432-442).

**Step 5: Run tests**

```bash
cargo test -p tauri-app --quiet
cargo test -p range-solver-compare --release --quiet
```

Expected: all PASS.

### Task 2.2: Measure and commit

**Step 1: Measure**

Run the same subgame solve as the baseline. Record new `rollout_hands_per_sec`. Compare to commit-1 baseline.

**Step 2: Commit**

```bash
git add -A
git commit -m "perf(postflop): hoist opponent weights out of per-combo loop"
```

Record delta in PR description:

```
| commit 2: opponent-weight hoist | X hands/s | +Y% vs baseline |
```

---

## Commit 3 — `perf(continuation): precompute 4-street bucket table per rollout`

Addresses suspect #4. Mirror MCCFR's `DealWithBuckets` pattern.

### Task 3.1: Precompute bucket table at rollout boundary

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs:143-153` (`rollout_from_boundary`)
- Modify: `crates/core/src/blueprint_v2/continuation.rs:165-270` (`rollout_inner`)

The boundary state knows the board so far. The rollout only progresses to later streets via Chance nodes. We precompute buckets for *current and remaining* streets before entering `rollout_inner`.

But the bucket for turn/river depends on the *sampled card* at each Chance node, so we can't fully precompute those. What we **can** precompute:

- Current-street buckets for hero and opponent (eliminates the 2 `get_bucket` calls at the first Decision node).
- **After a Chance node,** we still need a fresh lookup, BUT that happens only once per Chance node, not once per Decision node within the new street. The existing `cached_buckets` mechanism already handles within-street caching.

So the real fix here: **convert bucket lookups from mmap-backed calls into cached indices stored on a `RolloutDeal` struct**, and pass it through so every street transition only hits the bucket system once. This matches `DealWithBuckets` in spirit but allocates lazily on chance transitions.

**Step 1: Introduce `RolloutDeal` struct**

Add to `continuation.rs`:

```rust
/// Mutable per-rollout bucket cache, computed lazily as streets progress.
/// Populated entries at index `street_idx` contain `[hero_bucket, opp_bucket]`.
struct RolloutDeal {
    /// `Some` once the hero+opp buckets for that street have been computed.
    /// Streets beyond the current board.len() street are populated as Chance
    /// nodes deal out cards.
    per_street: [Option<[u16; 2]>; 4],
}

impl RolloutDeal {
    fn new() -> Self { Self { per_street: [None; 4] } }

    /// Look up (and cache) the bucket pair for a street.
    fn get_or_compute(
        &mut self,
        street: Street,
        hero: [Card; 2],
        opp: [Card; 2],
        board: &[Card],
        buckets: &AllBuckets,
    ) -> [u16; 2] {
        let i = street as usize;
        if let Some(b) = self.per_street[i] {
            return b;
        }
        let b = [
            buckets.get_bucket(street, hero, board),
            buckets.get_bucket(street, opp, board),
        ];
        self.per_street[i] = Some(b);
        b
    }

    /// Invalidate streets >= `from_street`. Called at Chance transitions.
    fn invalidate_from(&mut self, from_street: Street) {
        for i in (from_street as usize)..4 {
            self.per_street[i] = None;
        }
    }
}
```

**Step 2: Thread `RolloutDeal` through `rollout_inner`**

Replace the `cached_buckets: Option<[u16; 2]>` parameter with `deal: &mut RolloutDeal`. At decision nodes:

```rust
let buckets = deal.get_or_compute(street, hero_hand, opponent_hand, board, ctx.buckets);
```

At chance nodes, after sampling the card, call `deal.invalidate_from(next_street)` before recursing.

**Step 3: Entry point**

`rollout_from_boundary` constructs a fresh `RolloutDeal::new()` and passes `&mut` into `rollout_inner`.

**Step 4: Correctness test**

Add a unit test that exercises hero/opp bucket lookup across two streets (flop → turn transition) and asserts the cached result equals a fresh `ctx.buckets.get_bucket(...)` computation.

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core --quiet
cargo test -p tauri-app --quiet
cargo test -p range-solver-compare --release --quiet
```

Expected: all PASS.

### Task 3.2: Measure and commit

Same protocol as Task 2.2.

---

## Commit 4 — `perf(continuation): replace remaining_deck Vec with u64 mask`

Addresses suspect #2.

### Task 4.1: Convert deck representation to `u64` mask

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs:79-108` (`card_bit`, `remaining_deck`)
- Modify: `crates/core/src/blueprint_v2/continuation.rs:244-268` (`GameNode::Chance` arm)

**Step 1: Add mask helpers**

```rust
/// Build a `u64` mask of used card bits (hero + opp + board).
fn used_mask(hero: [Card; 2], opponent: [Card; 2], board: &[Card]) -> u64 {
    let mut used = 0u64;
    for &c in &hero { used |= 1u64 << card_bit(c); }
    for &c in &opponent { used |= 1u64 << card_bit(c); }
    for &c in board { used |= 1u64 << card_bit(c); }
    used
}

/// Sample a uniformly-random remaining card from a used-card mask.
fn sample_remaining_card(used: u64, rng: &mut impl Rng) -> Card {
    let remaining = !used & ((1u64 << 52) - 1);
    let count = remaining.count_ones();
    debug_assert!(count > 0, "deck exhausted");
    // Pick the Nth set bit.
    let pick = rng.random_range(0..count);
    let mut bits = remaining;
    for _ in 0..pick {
        bits &= bits - 1; // clear lowest set bit
    }
    bit_to_card(bits.trailing_zeros())
}

fn bit_to_card(bit: u32) -> Card {
    use crate::poker::{ALL_VALUES, ALL_SUITS};
    let value = ALL_VALUES[(bit / 4) as usize];
    let suit = ALL_SUITS[(bit % 4) as usize];
    Card::new(value, suit)
}
```

**Step 2: Replace Chance-arm body**

```rust
GameNode::Chance { next_street: _, child } => {
    let child = *child;
    let used = used_mask(hero_hand, opponent_hand, board);
    let deck_count = (!used & ((1u64 << 52) - 1)).count_ones();
    let n = ctx.num_rollouts.min(deck_count);
    if n == 0 { return 0.0; }

    let mut total = 0.0;
    for _ in 0..n {
        let card = sample_remaining_card(used, rng);
        let mut new_board = board.to_vec();  // kept for now; remove in follow-up if hot
        new_board.push(card);
        let child_ev = rollout_inner(...);
        deal.invalidate_from(next_street);  // if commit 3 landed
        total += child_ev;
    }
    total / f64::from(n)
}
```

**Note:** `new_board.to_vec()` is still an allocation per chance-node iteration. We're not eliminating it here — it's a pre-existing alloc and out of scope for this commit. If measurement shows it dominates, a follow-up commit can replace `&[Card]` boards with a small-fixed-array representation. Don't chase that in this PR.

**Step 3: Unit-test the sampler**

```rust
#[test]
fn sample_remaining_card_excludes_used() {
    let hero = [Card::new(Value::Ace, Suit::Spade), Card::new(Value::King, Suit::Spade)];
    let opp = [Card::new(Value::Queen, Suit::Heart), Card::new(Value::Jack, Suit::Heart)];
    let board = vec![Card::new(Value::Two, Suit::Club), Card::new(Value::Three, Suit::Diamond), Card::new(Value::Four, Suit::Club)];
    let used = used_mask(hero, opp, &board);
    let mut rng = SmallRng::seed_from_u64(42);
    for _ in 0..1000 {
        let c = sample_remaining_card(used, &mut rng);
        assert_eq!(used & (1u64 << card_bit(c)), 0, "sampled card was in used mask");
    }
}

#[test]
fn sample_remaining_card_uniform_distribution() {
    // Build a mask with exactly 4 cards remaining. Sample 10k times and
    // assert each card appears roughly 2500 ± tolerance.
    // ...
}
```

**Step 4: Remove `remaining_deck()` and its callers**

```bash
grep -rn "remaining_deck" crates/
```

Delete every callsite (should only be the Chance arm we just rewrote).

**Step 5: Run tests**

```bash
cargo test -p poker-solver-core --quiet
cargo test -p range-solver-compare --release --quiet
```

Expected: all PASS.

### Task 4.2: Measure and commit

Same protocol as Task 2.2.

---

## Commit 5 — `perf(continuation): pass RolloutScratch through rollout_inner`

Addresses suspect #5.

### Task 5.1: Introduce `RolloutScratch`

**Files:**
- Modify: `crates/core/src/blueprint_v2/continuation.rs:39-77` (`bias_strategy`)
- Modify: `crates/core/src/blueprint_v2/continuation.rs:165-243` (`rollout_inner` Decision arm)

**Step 1: Add `smallvec` to core dependencies**

Check `crates/core/Cargo.toml`. If `smallvec` not present, add `smallvec = "1"`.

**Step 2: Define `RolloutScratch`**

```rust
/// Scratch buffers reused across rollouts in one par-unit. Construct once
/// per (combo, rollout) worker; reuse across recursive `rollout_inner` calls.
pub struct RolloutScratch {
    pub action_classes: smallvec::SmallVec<[ActionClass; 16]>,
    pub bias_probs: smallvec::SmallVec<[f32; 16]>,
}

impl RolloutScratch {
    pub fn new() -> Self {
        Self {
            action_classes: smallvec::SmallVec::new(),
            bias_probs: smallvec::SmallVec::new(),
        }
    }
}
```

Size 16 matches MCCFR's action buffers (`mccfr.rs:914-921`).

**Step 3: Rewrite `bias_strategy` to write into scratch**

```rust
pub fn bias_strategy_into(
    out: &mut smallvec::SmallVec<[f32; 16]>,
    probs: &[f32],
    actions: &[ActionClass],
    bias: BiasType,
    factor: f64,
) {
    out.clear();
    if bias == BiasType::Unbiased {
        out.extend_from_slice(probs);
        return;
    }
    let target = match bias {
        BiasType::Fold => ActionClass::Fold,
        BiasType::Call => ActionClass::Call,
        BiasType::Raise => ActionClass::Raise,
        BiasType::Unbiased => unreachable!(),
    };
    let factor = factor as f32;
    out.extend(probs.iter().zip(actions.iter()).map(|(&p, &a)|
        if a == target { p * factor } else { p }
    ));
    let sum: f32 = out.iter().sum();
    if sum > 0.0 {
        for p in out.iter_mut() { *p /= sum; }
    }
}
```

Keep the old `bias_strategy` returning `Vec<f32>` for backward compat if any test imports it; delete if nothing uses it.

**Step 4: Thread `&mut RolloutScratch` through `rollout_inner`**

Add a `scratch: &mut RolloutScratch` parameter. At the Decision arm, replace:

```rust
let action_classes: Vec<ActionClass> = actions.iter().map(classify_action).collect();
let biased = bias_strategy(probs, &action_classes, ctx.bias, ctx.bias_factor);
```

with:

```rust
scratch.action_classes.clear();
scratch.action_classes.extend(actions.iter().map(classify_action));
bias_strategy_into(&mut scratch.bias_probs, probs, &scratch.action_classes, ctx.bias, ctx.bias_factor);
```

**Warning:** the loop that uses `biased[i]` must now use `scratch.bias_probs[i]` BEFORE the recursive `rollout_inner` call, because that call needs `&mut scratch` and will clobber `bias_probs`. Read the probability into a local `f32` first:

```rust
for (i, &child) in children.iter().enumerate() {
    let p = scratch.bias_probs[i];  // snapshot before recursing
    let (child_pot, child_invested) = apply_action(...);
    let child_ev = rollout_inner(..., scratch, child_pot, child_invested);
    ev += f64::from(p) * child_ev;
}
```

**Step 5: Entry point**

`rollout_from_boundary` constructs a `RolloutScratch::new()` and passes `&mut`. Call site in `postflop.rs:464-479` constructs one per `sampled.iter().map(...)` iteration; push construction *outside* the map, reuse across all `sampled`:

```rust
let mut scratch = RolloutScratch::new();
let total: f64 = sampled.iter().map(|&j| {
    rollout_from_boundary(..., &mut scratch)
}).sum();
```

If `rollout_from_boundary` takes scratch by `&mut`, the lambda captures it mutably — fine since `.map()` is sequential.

**Step 6: Tests**

```bash
cargo test -p poker-solver-core --quiet
cargo test -p tauri-app --quiet
cargo test -p range-solver-compare --release --quiet
```

Expected: all PASS.

### Task 5.2: Measure and commit

Same protocol as Task 2.2.

---

## Commit 6 — `perf(postflop): flatten to combo × rollout parallelism`

Addresses suspect #3 reshaped.

### Task 6.1: Flatten the parallel domain

**Files:**
- Modify: `crates/tauri-app/src/postflop.rs:411-484`

**Step 1: Restructure**

Current shape:
```rust
combos.par_iter().enumerate().map(|(i, hero_hand)| {
    // build scratch, ctx
    let total: f64 = sampled.iter().map(|&j| rollout_from_boundary(...)).sum();
    total / sampled.len() as f64
}).collect()
```

New shape: build a flat vector of `(combo_idx, opp_sample_idx)` work items, `par_iter` over it, accumulate into a per-combo `Vec<AtomicF64>` or into a `DashMap`-style structure, then divide by per-combo sample counts.

Simplest approach using rayon's `fold` + `reduce`:

```rust
use rayon::prelude::*;

// Per-combo sample lists computed serially (cheap):
let per_combo_samples: Vec<Vec<usize>> = combos.iter().enumerate().map(|(i, hero)| {
    if hero_range[i] <= 0.0 { return Vec::new(); }
    let mut rng = SmallRng::seed_from_u64(i as u64);
    sample_weighted_filtered(&mut rng, opp_range, self.num_opponent_samples,
        |j| opp_range[j] <= 0.0 || cards_overlap(*hero, combos[j]))
}).collect();

// Flat (combo_idx, sample_idx) pairs:
let flat_work: Vec<(usize, usize)> = per_combo_samples.iter().enumerate()
    .flat_map(|(i, sam)| sam.iter().enumerate().map(move |(k, _)| (i, k)))
    .collect();

// Parallel rollout:
let partials: Vec<((usize, usize), f64)> = flat_work.par_iter().map(|&(i, k)| {
    let opp_j = per_combo_samples[i][k];
    let hero = combos[i];
    let opp = combos[opp_j];
    let mut rng = SmallRng::seed_from_u64((i as u64) * 131 + k as u64);
    let mut scratch = RolloutScratch::new();
    let ctx = ...; // same as before; opp_base_weights only used for sampling, so not needed here
    let ev = rollout_from_boundary(hero, opp, board, &ctx, self.abstract_start_node,
        &mut rng, boundary_pot, boundary_invested, &mut scratch);
    ((i, k), ev)
}).collect();

// Aggregate per combo:
let mut totals = vec![0.0f64; combos.len()];
let mut counts = vec![0u32; combos.len()];
for ((i, _), ev) in partials {
    totals[i] += ev;
    counts[i] += 1;
}

(0..combos.len()).map(|i| {
    if counts[i] == 0 { 0.0 } else { totals[i] / counts[i] as f64 }
}).collect()
```

**Step 2: Watch seeding**

The per-combo RNG seed now must vary with `(i, k)` so different samples get different streams. Seeded with `(i as u64) * 131 + k as u64` (primes help scramble) — good enough.

**Step 3: Confirm correctness**

Compare output of flat vs per-combo implementation on a small fixture. Use the snapshot test infrastructure from commit 2. Tolerance: results should be **very close but not bit-identical** because RNG sequence changes. If the tests tolerate this, good. If not, rework the seeding to match the old scheme exactly (one RNG per combo, seeded with combo index, walked forward by rollout index).

**Step 4: Run tests**

```bash
cargo test --all --quiet
time cargo test --all --quiet  # confirm under 60s
```

Expected: all PASS, under 1 minute.

### Task 6.2: Measure and commit

Same protocol as Task 2.2.

---

## Finalize

### Task F1: Full verification

**REQUIRED SUB-SKILL:** hex:verification-before-completion

**Step 1: Full suite**

```bash
cargo test --all --quiet
cargo test -p range-solver-compare --release --quiet
cargo clippy --all-targets -- -D warnings
cd frontend && npm run typecheck
```

Expected: all PASS. If clippy complains about any new code, fix.

**Step 2: Measure end-to-end**

Run the same subgame solve one more time. Record final `rollout_hands_per_sec`. Compare to baseline.

### Task F2: Open PR

**REQUIRED SUB-SKILL:** hex:finishing-a-development-branch

**Step 1: Push**

```bash
git push -u origin perf/subgame-rollout
```

**Step 2: Open PR**

Title: `perf(subgame): rollout throughput — eliminate per-hand allocs + add hands/sec telemetry`

Body must contain:

```markdown
## Summary
- Adds `rollout_hands_per_sec` to the Tauri progress bar so rollout throughput is visible during solves.
- Five rollout-path optimizations, each in its own commit, each measured.

## Hands/sec attribution

| Commit | Hands/sec | Δ vs prev |
|-|-|-|
| baseline | X | — |
| opponent-weight hoist | | |
| 4-street bucket precompute | | |
| deck u64 mask | | |
| rollout scratch | | |
| flat combo × rollout parallelism | | |

Target was ~100k hands/sec (matching MCCFR). Actual final: [fill].

## Test plan
- [ ] `cargo test --all` — PASS
- [ ] `cargo test -p range-solver-compare --release` — PASS
- [ ] `cargo clippy --all-targets -- -D warnings` — clean
- [ ] `npm run typecheck` in frontend — clean
- [ ] Manually ran a subgame solve in the explorer and verified `hands/s` counter displays.
```

---

## Execution options

**Plan complete and saved to `docs/plans/2026-04-17-subgame-rollout-perf-impl.md`. Two execution options:**

**1. Subagent-Driven (this session)** — Agent teams with parallel implementer-reviewer streams. Fast throughput, but the commits in this plan are strictly sequential (each depends on the previous for measurement attribution), so parallelism doesn't help much here.

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints. Better fit for this plan because commits must land in order and each needs a hands/sec reading.

**Recommendation: 2 (Parallel Session).** The measurement-per-commit discipline benefits from the explicit checkpoints in `hex:executing-plans`.

**Which approach?**
