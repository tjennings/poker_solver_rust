# CFVNet Clippy Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Fix all 18 clippy/compiler warnings in the cfvnet crate.

**Architecture:** Mechanical fixes only — no behavioral changes.

**Tech Stack:** Rust, clippy

---

### Task 1: Fix dead code warnings

**Files:**
- Modify: `crates/cfvnet/src/model/dataset.rs:100`
- Modify: `crates/cfvnet/src/model/training.rs:205`

**Step 1: Remove unused `records()` method in dataset.rs**

Delete the `records()` method at line 100. Verify it's truly unused by checking no callers exist.

**Step 2: Remove unused `reset()` method in training.rs**

The user refactored `spawn_dataloader_thread` to create a new `StreamingReader` per epoch instead of calling `reset()`. Delete the `reset()` method at line 205. Note: `reset()` is used in tests — if so, keep it and add `#[cfg(test)]` or just add `#[allow(dead_code)]`.

Actually, check `streaming_reader_spans_files` test and `streaming_reader_read_one` test — they call `reader.reset()`. If so, annotate the method with `#[cfg(test)]` won't work since it's not in the test module. Use `#[allow(dead_code)]` instead, or better: keep it and suppress with `#[allow(dead_code)]` on just that method.

**Step 3: Run `cargo build -p cfvnet 2>&1 | grep warning`**

Expected: 0 warnings.

**Step 4: Commit**

```
git commit -m "fix(cfvnet): remove dead code and suppress test-only warnings"
```

---

### Task 2: Fix collapsible if statements in training.rs

**Files:**
- Modify: `crates/cfvnet/src/model/training.rs:366-370`
- Modify: `crates/cfvnet/src/model/training.rs:491-494`

**Step 1: Collapse checkpoint scanning ifs (lines 366-370)**

Replace:
```rust
if let Some(rest) = name.strip_prefix("checkpoint_epoch") {
    if let Some(n_str) = rest.strip_suffix(".mpk.gz") {
        if let Ok(n) = n_str.parse::<usize>() {
            best_epoch = best_epoch.max(n);
        }
    }
}
```

With:
```rust
if let Some(n) = name.strip_prefix("checkpoint_epoch")
    .and_then(|rest| rest.strip_suffix(".mpk.gz"))
    .and_then(|n_str| n_str.parse::<usize>().ok())
{
    best_epoch = best_epoch.max(n);
}
```

**Step 2: Collapse send-if-not-empty (line 491)**

Replace:
```rust
if !batch_buf.is_empty() {
    if record_tx.send(batch_buf).is_err() {
        return;
    }
}
```

With:
```rust
if !batch_buf.is_empty() && record_tx.send(batch_buf).is_err() {
    return;
}
```

**Step 3: Run `cargo clippy -p cfvnet 2>&1 | grep "crates/cfvnet/src/model/training.rs"`**

Expected: No warnings from training.rs.

**Step 4: Commit**

```
git commit -m "fix(cfvnet): collapse nested if statements per clippy"
```

---

### Task 3: Derive Default for EvaluationConfig

**Files:**
- Modify: `crates/cfvnet/src/config.rs:235-241`

**Step 1: Replace manual Default impl with derive**

Replace:
```rust
impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            regression_spots: None,
        }
    }
}
```

With `#[derive(Default)]` on the struct (should already have other derives — add Default to the list):

Find the `EvaluationConfig` struct definition and add `Default` to its derive list. Then delete the manual `impl Default`.

**Step 2: Run `cargo clippy -p cfvnet 2>&1 | grep config.rs`**

Expected: No warnings.

**Step 3: Commit**

```
git commit -m "fix(cfvnet): derive Default for EvaluationConfig"
```

---

### Task 4: Fix collapsible if in main.rs

**Files:**
- Modify: `crates/cfvnet/src/main.rs:153-158`

**Step 1: Collapse nested if**

Replace:
```rust
if let Some(parent) = path.parent() {
    if !parent.as_os_str().is_empty() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| {
```

With:
```rust
if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
    std::fs::create_dir_all(parent).unwrap_or_else(|e| {
```

Remove the corresponding extra closing brace.

**Step 2: Commit**

```
git commit -m "fix(cfvnet): collapse if in ensure_parent_dir"
```

---

### Task 5: Fix manual div_ceil in main.rs

**Files:**
- Modify: `crates/cfvnet/src/main.rs:323`

**Step 1: Replace manual div_ceil**

Replace:
```rust
let num_files = (total + chunk_size - 1) / chunk_size;
```

With:
```rust
let num_files = total.div_ceil(chunk_size);
```

**Step 2: Commit**

```
git commit -m "fix(cfvnet): use div_ceil instead of manual implementation"
```

---

### Task 6: Fix needless_range_loop warnings in main.rs

**Files:**
- Modify: `crates/cfvnet/src/main.rs:709,842,1085`

**Step 1: Fix histogram loop at line 709**

The loop uses `i` to compute `lo`/`hi` from `min_val + i * bucket_width`, so it genuinely needs the index. Use `enumerate` on `bucket_counts`:

Replace:
```rust
for i in 0..NUM_BUCKETS {
    let lo = min_val + i as f64 * bucket_width;
    let hi = lo + bucket_width;
    let count = bucket_counts[i];
```

With:
```rust
for (i, &count) in bucket_counts.iter().enumerate() {
    let lo = min_val + i as f64 * bucket_width;
    let hi = lo + bucket_width;
```

**Step 2: Fix rank_counts loop at line 842**

Replace:
```rust
for r in 0..12 {
    if rank_counts[r] > 0 { wheel_run += 1; } else { break; }
}
```

With:
```rust
for &c in &rank_counts[..12] {
    if c > 0 { wheel_run += 1; } else { break; }
}
```

**Step 3: Fix histogram loop at line 1085**

Same pattern as step 1 — replace with `enumerate`. This is the `print_frequency_histogram` function:

Replace:
```rust
for i in 0..NUM_BUCKETS {
    let lo = min_key + i as f64 * bucket_width;
    let hi = lo + bucket_width;
    let count = bucket_counts[i];
```

With:
```rust
for (i, &count) in bucket_counts.iter().enumerate() {
    let lo = min_key + i as f64 * bucket_width;
    let hi = lo + bucket_width;
```

**Step 4: Commit**

```
git commit -m "fix(cfvnet): use iterators instead of range loops per clippy"
```

---

### Task 7: Fix redundant closures and useless vecs in main.rs

**Files:**
- Modify: `crates/cfvnet/src/main.rs:1005-1006,1062-1063` (redundant closures)
- Modify: `crates/cfvnet/src/main.rs:698,1013,1070` (useless vecs)

**Step 1: Fix redundant closures**

At lines 1005-1006 and 1062-1063, replace `|s| key_fn(s)` with `&key_fn` and `|s| val_fn(s)` with `&val_fn`:

```rust
// Before:
let min_key = spots.iter().map(|s| key_fn(s)).fold(f64::INFINITY, f64::min);
let max_key = spots.iter().map(|s| key_fn(s)).fold(f64::NEG_INFINITY, f64::max);
// After:
let min_key = spots.iter().map(&key_fn).fold(f64::INFINITY, f64::min);
let max_key = spots.iter().map(&key_fn).fold(f64::NEG_INFINITY, f64::max);
```

Apply same pattern at lines 1062-1063.

**Step 2: Fix useless vecs**

Replace `vec![0_u32; NUM_BUCKETS]` with `[0_u32; NUM_BUCKETS]` at lines 698, 1070.
Replace `vec![0.0_f64; NUM_BUCKETS]` with `[0.0_f64; NUM_BUCKETS]` at line 1013.

Note: These variables are used with `.iter()` which works on arrays too, so this is safe. Change `let mut bucket_counts = vec![...]` to `let mut bucket_counts = [...]`.

**Step 3: Run `cargo clippy -p cfvnet 2>&1 | grep "crates/cfvnet"`**

Expected: 0 warnings from cfvnet.

**Step 4: Commit**

```
git commit -m "fix(cfvnet): fix redundant closures and useless vecs per clippy"
```

---

### Task 8: Final verification

**Step 1: Run full test suite**

```
cargo test
```

Expected: All tests pass.

**Step 2: Run clippy**

```
cargo clippy -p cfvnet 2>&1 | grep "crates/cfvnet"
```

Expected: 0 cfvnet warnings.
