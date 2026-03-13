# SPR Input + Pot-Relative Output Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Replace pot+stack inputs with single SPR feature and switch to pot-relative CFV targets, matching Supremus normalization.

**Architecture:** Three changes: (1) input shrinks from 2706 to 2705 — replace pot and stack features with `pot/effective_stack`, (2) CFV targets divided by pot during encoding and multiplied by pot at inference, (3) pot-weighted loss removed since pot-relative targets are inherently balanced.

**Tech Stack:** Rust, burn (tensor framework)

---

### Task 1: Update input encoding — SPR feature

**Files:**
- Modify: `crates/cfvnet/src/model/network.rs`
- Modify: `crates/cfvnet/src/model/dataset.rs`

**Step 1: Update INPUT_SIZE in network.rs**

Change:
```rust
pub const INPUT_SIZE: usize = NUM_COMBOS + NUM_COMBOS + 52 + 1 + 1; // = 2706
```
to:
```rust
/// Fixed input feature size: OOP range (1326) + IP range (1326) + board one-hot (52) + SPR (1).
pub const INPUT_SIZE: usize = NUM_COMBOS + NUM_COMBOS + 52 + 1; // = 2705
```

**Step 2: Update encode_record in dataset.rs**

Replace lines 165-168:
```rust
    // Pot (normalized by max pot)
    input.push(rec.pot / 400.0);
    // Effective stack (normalized by max stack)
    input.push(rec.effective_stack / 400.0);
```
with:
```rust
    // SPR: pot / effective_stack (matches Supremus pot/starting_chips convention)
    let spr = if rec.effective_stack > 0.0 {
        rec.pot / rec.effective_stack
    } else {
        0.0
    };
    input.push(spr);
```

**Step 3: Update encode_situation_for_inference in dataset.rs**

Replace lines 143-146:
```rust
    // Pot (normalized by max pot)
    input.push(sit.pot as f32 / 400.0);
    // Effective stack (normalized by max stack)
    input.push(sit.effective_stack as f32 / 400.0);
```
with:
```rust
    // SPR: pot / effective_stack
    let spr = if sit.effective_stack > 0 {
        sit.pot as f32 / sit.effective_stack as f32
    } else {
        0.0
    };
    input.push(spr);
```

Update the doc comment on `encode_situation_for_inference` from:
```
/// Layout: `[OOP_range(1326), IP_range(1326), board_one_hot(52), pot(1), stack(1)]`
```
to:
```
/// Layout: `[OOP_range(1326), IP_range(1326), board_one_hot(52), spr(1)]`
```

**Step 4: Update river_net_evaluator.rs build_input**

In `crates/cfvnet/src/eval/river_net_evaluator.rs`, function `build_input` (around lines 78-94), replace the pot/stack pushes:
```rust
    input.push(pot as f32 / 400.0);
    input.push(effective_stack as f32 / 400.0);
```
with:
```rust
    let spr = if effective_stack > 0.0 {
        pot as f32 / effective_stack as f32
    } else {
        0.0
    };
    input.push(spr);
```

**Step 5: Update compare_turn.rs predict_with_model**

In `crates/cfvnet/src/eval/compare_turn.rs`, find the input encoding section and make the same pot/stack → SPR change.

**Step 6: Update dataset.rs tests**

In `dataset_input_encoding_correct` test: the SPR feature is at index 2704 (was pot at 2704, stack at 2705). Test record has pot=100.0, effective_stack=50.0, so SPR = 100/50 = 2.0.
```rust
    // SPR = 100/50 = 2.0, at index 2704
    assert!((item.input[2704] - 2.0).abs() < 1e-6);
    assert_eq!(item.input.len(), INPUT_SIZE);
    assert_eq!(item.input.len(), 2705);
```

In `dataset_turn_encoding_same_input_size`: update to check 2705.

In `no_player_indicator_in_input`: update assert to 2705.

Update `dataset_input_size_method` — both should return 2705.

**Step 7: Update network.rs tests**

Update `input_size_correct_for_river`: assert INPUT_SIZE == 2705.

**Step 8: Run tests**

Run: `cargo test -p cfvnet dataset` and `cargo test -p cfvnet network`
Expected: PASS

**Step 9: Commit**

```bash
git add crates/cfvnet/src/model/network.rs crates/cfvnet/src/model/dataset.rs \
    crates/cfvnet/src/eval/river_net_evaluator.rs crates/cfvnet/src/eval/compare_turn.rs
git commit -m "feat: replace pot+stack inputs with single SPR feature"
```

---

### Task 2: Pot-relative CFV targets

**Files:**
- Modify: `crates/cfvnet/src/model/dataset.rs`

**Step 1: Divide targets by pot in encode_record**

Replace lines 173-174:
```rust
    let oop_target = rec.oop_cfvs.to_vec();
    let ip_target = rec.ip_cfvs.to_vec();
```
with:
```rust
    // Pot-relative CFV targets: CFV / pot (matching Supremus output normalization).
    let pot_inv = if rec.pot > 0.0 { 1.0 / rec.pot } else { 0.0 };
    let oop_target: Vec<f32> = rec.oop_cfvs.iter().map(|&v| v * pot_inv).collect();
    let ip_target: Vec<f32> = rec.ip_cfvs.iter().map(|&v| v * pot_inv).collect();
```

**Step 2: Update dataset_dual_target_from_record test**

Test record has pot=100.0. Record `i=1` has oop_cfvs[0]=0.01, ip_cfvs[0]=-0.01. After dividing by pot:
```rust
    let item1 = dataset.get(1).unwrap();
    // pot=100, so pot-relative: 0.01 / 100 = 0.0001
    assert!((item1.oop_target[0] - 0.0001).abs() < 1e-7);
    assert!((item1.ip_target[0] - (-0.0001)).abs() < 1e-7);
```

**Step 3: Run tests**

Run: `cargo test -p cfvnet dataset`
Expected: PASS

**Step 4: Commit**

```bash
git add crates/cfvnet/src/model/dataset.rs
git commit -m "feat: pot-relative CFV targets (divide by pot during encoding)"
```

---

### Task 3: Remove pot-weighted loss

**Files:**
- Modify: `crates/cfvnet/src/model/dataset.rs`
- Modify: `crates/cfvnet/src/model/training.rs`
- Modify: `crates/cfvnet/src/model/loss.rs`
- Modify: `crates/cfvnet/src/config.rs`

**Step 1: Remove pot from CfvItem**

In `dataset.rs`, remove `pot: f32` field from `CfvItem`. Remove `pot: rec.pot` from `encode_record`.

**Step 2: Remove pot from training pipeline**

In `training.rs`:
- Remove `pot: Vec<f32>` from `PreEncoded`
- Remove `pot: Tensor<B, 1>` from `ChunkTensors`
- Remove `pot: Tensor<B, 1>` from `MiniBatch`
- Remove all `pot` handling in `from_records`, `into_tensors`, `to_tensors`, `chunk_tensors`, `index_select`, `slice_batch`
- Remove `pot_weighted_loss: bool` from `TrainConfig`
- In the training loop and `compute_val_loss`, remove the `pot_w` variable and pass `None` for `pot_weight` to `cfvnet_loss`

**Step 3: Simplify cfvnet_loss signature**

In `loss.rs`, remove `pot_weight: Option<Tensor<B, 1>>` parameter from `cfvnet_loss`. Pass `None` directly to both `masked_huber_loss` calls. (Keep `masked_huber_loss` unchanged — the pot_weight parameter might be useful for other callers.)

**Step 4: Remove pot_weighted_loss from config**

In `config.rs`:
- Remove `pot_weighted_loss` field from `TrainingConfig`
- Remove `default_pot_weighted_loss` function
- Remove it from the `Default` impl

**Step 5: Remove pot_weighted_loss from sample config**

In `sample_configurations/river_cfvnet.yaml`, remove `pot_weighted_loss` line if present.

**Step 6: Update tests**

- In `training.rs` tests: remove `pot_weighted_loss: false` from all `TrainConfig` constructions
- In `loss.rs` tests: remove pot_weight parameter from `cfvnet_loss` calls in `dual_player_loss_sums_both`
- Keep `masked_huber_loss` pot-weight tests since the function still supports it

**Step 7: Run tests**

Run: `cargo test -p cfvnet`
Expected: PASS

**Step 8: Commit**

```bash
git add crates/cfvnet/src/model/dataset.rs crates/cfvnet/src/model/training.rs \
    crates/cfvnet/src/model/loss.rs crates/cfvnet/src/config.rs \
    sample_configurations/river_cfvnet.yaml
git commit -m "feat: remove pot-weighted loss (pot-relative targets are inherently balanced)"
```

---

### Task 4: Multiply by pot at inference

**Files:**
- Modify: `crates/cfvnet/src/main.rs`
- Modify: `crates/cfvnet/src/eval/river_net_evaluator.rs`
- Modify: `crates/cfvnet/src/eval/compare_turn.rs`

**Step 1: Update cmd_compare in main.rs**

The predict_fn closure (around line 481-498) returns OOP CFVs for comparison against solver EVs (which are absolute). The model now outputs pot-relative values, so multiply by pot:

```rust
    let summary = run_comparison(&cfg.game, &cfg.datagen, num_spots, cfg.datagen.seed, |sit, _solve_result| {
        let input_data = encode_situation_for_inference(sit);
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [1, in_size]),
            &device,
        );
        let range_oop = Tensor::<B, 2>::from_data(
            TensorData::new(sit.ranges[0].to_vec(), [1, NUM_COMBOS]),
            &device,
        );
        let range_ip = Tensor::<B, 2>::from_data(
            TensorData::new(sit.ranges[1].to_vec(), [1, NUM_COMBOS]),
            &device,
        );
        let pred = model.forward(input, range_oop, range_ip);
        let all_cfvs = pred.into_data().to_vec::<f32>().unwrap();
        // Model outputs pot-relative CFVs; multiply by pot for absolute values.
        let pot = sit.pot as f32;
        all_cfvs[..1326].iter().map(|&v| v * pot).collect()
    })
```

**Step 2: Update cmd_evaluate in main.rs**

The evaluate loop (around line 394-428) compares predictions to targets. Both pred and target are now pot-relative, so the metrics are comparable without conversion. However, for mBB metrics, we need the absolute values. Check `compute_prediction_metrics` — if it uses pot for mBB, we need to pass predictions and targets in absolute terms, or adjust the metric.

Read `compute_prediction_metrics` in `metrics.rs` to determine whether to convert back to absolute for metrics, or adjust the metric function. The simplest approach: multiply both pred and target by pot before computing metrics, since the evaluate command wants absolute error metrics.

```rust
    // Convert pot-relative predictions and targets back to absolute for metrics.
    let pot = item.pot;  // Wait — pot is removed from CfvItem!
```

**Important:** We removed `pot` from `CfvItem` in Task 3. But cmd_evaluate needs pot to compute mBB. We need to keep pot accessible. Options:
- Keep `pot` in `CfvItem` as metadata (not for loss, just for metrics)
- Read pot directly from the TrainingRecord in cmd_evaluate

The cleanest approach: keep `pot` in `CfvItem` (restore it). It's just metadata for evaluation, not used in training. Or alternatively, since cmd_evaluate gets records from the dataset, add a method to access raw pot.

**Revised approach for Task 3:** Keep `pot: f32` in `CfvItem` but remove it from `PreEncoded`/`ChunkTensors`/`MiniBatch` (the training pipeline). The field exists for evaluation metadata only.

**Step 3: Update river_net_evaluator.rs**

The evaluator outputs CFVs that feed into the turn solver. The turn solver expects absolute CFVs. So after the forward pass, multiply by pot:

In `river_net_evaluator.rs` around line 188-195, after extracting cfvs:
```rust
    let cfv_offset = if traverser == 0 { 0 } else { NUM_COMBOS };
    // Model outputs pot-relative; convert to absolute for the turn solver.
    let pot_f32 = pot as f32;
    for (i, &idx) in combo_indices.iter().enumerate() {
        if valid_combo_mask[i] {
            cfv_sum[i] += (out_vec[cfv_offset + idx] * pot_f32) as f64;
            cfv_count[i] += 1;
        }
    }
```

**Step 4: Update compare_turn.rs**

Same pattern — multiply model output by pot before comparing to solver absolute EVs.

**Step 5: Run tests**

Run: `cargo test -p cfvnet`
Expected: PASS

**Step 6: Run integration test**

Run: `cargo test -p cfvnet --test integration_test`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/cfvnet/src/main.rs crates/cfvnet/src/eval/river_net_evaluator.rs \
    crates/cfvnet/src/eval/compare_turn.rs
git commit -m "feat: multiply pot-relative predictions by pot at inference"
```

---

### Task 5: Full verification

**Step 1: Run complete test suite**

Run: `cargo test`
Expected: All tests pass

**Step 2: Run clippy**

Run: `cargo clippy -p cfvnet`
Expected: No new warnings

**Step 3: Verify test suite runs in < 1 minute**

Time the run. If > 1 minute, investigate.
