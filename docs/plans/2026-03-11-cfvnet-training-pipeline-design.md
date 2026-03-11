# CFVnet Training Pipeline Completion

**Goal:** Complete the cfvnet train/evaluate/compare pipeline so a model can be trained, saved, loaded, and evaluated end-to-end.

**Architecture:** Wire burn's built-in `Module::save_file`/`load_file` (NamedMpkGz format) into the existing training loop, add epoch progress bars with indicatif, implement the evaluate and compare CLI stubs using existing metrics/comparison infrastructure.

## What exists

- `generate` command: fully working, writes binary training records
- `train` command: training loop runs but `_output_dir` is ignored — model never saved
- `evaluate` command: stub, exits with "not implemented"
- `compare` command: stub, exits with "not implemented"
- `CfvNet` derives `Module` — burn save/load works out of the box
- `metrics.rs`: MAE, max error, mBB computation ready
- `compare.rs`: `run_comparison` harness with pluggable prediction function ready
- `dataset.rs`: loading and item extraction ready
- `config.rs`: training params including `checkpoint_every_n_batches` (to rename)

## Changes

### Model serialization

- Format: `NamedMpkGzFileRecorder<FullPrecisionSettings>`
- Output layout:
  ```
  output_dir/
    model.mpk.gz
    checkpoint_epoch3.mpk.gz
    checkpoint_epoch6.mpk.gz
  ```
- Resume: if `output_dir/model.mpk.gz` exists at train start, load weights and continue

### Training loop (`training.rs`)

- Use `output_dir` parameter (drop underscore)
- Per-epoch indicatif progress bar: `Epoch 3/10 ████████░░ 80% [00:12] train=0.0234 val=0.0312`
- Validation split: hold out configured % of data, compute val loss after each epoch
- Checkpoint: save every N epochs (rename config field to `checkpoint_every_n_epochs`)
- Final save: write `model.mpk.gz` when training completes
- Print "Resuming from checkpoint" when loading existing model

### Evaluate command (`main.rs`)

- Load model from `--model` path using `NamedMpkGzFileRecorder`
- Load dataset from `--data` path
- Run forward pass on all records
- Print MAE, max error, mBB using existing `metrics.rs`

### Compare command (`main.rs`)

- Load model from `--model` path
- Use existing `run_comparison` harness with model inference as prediction function
- Print `ComparisonSummary` (mean MAE, mean max error, mean mBB, worst MAE, worst mBB)

### Error handling

- Model save fails: print error and exit
- Model load fails (resume/evaluate/compare): print error and exit
- Validation split yields 0 samples: skip val loss, print warning

### Config change

- Rename `checkpoint_every_n_batches` to `checkpoint_every_n_epochs` in config.rs and YAML

### Testing

- Extend `overfit_single_batch` to verify model saves to tempdir and loads back
- Smoke tests for evaluate and compare on tiny data
