# Design: Pot-Weighted Loss + Log-Uniform Pot Sampling

**Date:** 2026-03-12
**Status:** Approved

## Problem

The model performs significantly worse on large-pot spots (MAE 0.15-0.19, mBB 15,000-18,000) vs small-pot spots (MAE 0.04-0.07, mBB 96-140). Training data convergence is fine — the issue is that:

1. **Unweighted Huber loss** treats 0.1 MAE identically at pot=4 and pot=190, so the optimizer under-invests in large-pot accuracy
2. **Stratified interval sampling** gives equal sample count per interval, but [80,200] is 7.5x wider than [4,20], so the model sees fewer samples per unit of pot in the high range

## Solution

### 1. Pot-Weighted Huber Loss

Thread `pot` as a separate `Tensor<B, 1>` through the training pipeline. Weight each sample's Huber loss by its pot value:

`loss = sum(pot_i * L_i) / sum(pot_i)`

This directly minimizes chip-space MAE normalized across the batch.

**Changes to data pipeline:**
- `CfvItem` — add `pub pot: f32`
- `encode_record` — set `pot: rec.pot`
- `PreEncoded` — add `pot: Vec<f32>`, populated from `item.pot`
- `ChunkTensors` — add `pot: Tensor<B, 1>`
- `MiniBatch` — add `pot: Tensor<B, 1>`

**Changes to loss:**
- `masked_huber_loss` — add `pot_weight: Tensor<B, 1>` parameter. After computing per-sample element loss, multiply each sample's loss by `pot_weight[i]` (broadcast from `[batch]` to `[batch, 1326]`), then divide by sum of pot weights
- `cfvnet_loss` — pass pot through to `masked_huber_loss`

**Config:**
- Add `pot_weighted_loss: bool` to `TrainingConfig` (default `true`) for A/B testing

### 2. Log-Uniform Pot Sampling

Replace stratified interval sampling with log-uniform sampling over a continuous range.

**Config change (breaking):**
- Replace `pot_intervals: Vec<[i32; 2]>` with `pot_range: [i32; 2]`
- Example: `pot_range: [4, 200]`

**Sampling:**
- `sample_pot` computes `exp(uniform(ln(lo), ln(hi)))`, rounded to integer
- Gives equal density per multiplicative factor of pot

**Callers:**
- All references to `pot_intervals` update to `pot_range`
- Update sample config yaml
- Update tests

## What stays unchanged
- Model architecture (inputs, outputs, hidden layers)
- Evaluation/compare pipeline
- Binary training data format (pot is already stored in TrainingRecord)
- Auxiliary game-value loss (unweighted)
