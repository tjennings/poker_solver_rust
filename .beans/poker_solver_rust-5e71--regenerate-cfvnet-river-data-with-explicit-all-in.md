---
# poker_solver_rust-5e71
title: Regenerate CFVNet river data with explicit all-in
status: todo
type: task
priority: high
created_at: 2026-05-04T15:45:18Z
updated_at: 2026-05-04T15:50:30Z
---

Run full BoundaryNet/CFVNet river datagen after the explicit all-in parsing fix. Use the fixed branch/commit, generate into a fresh local_data/cfvnet directory, run datagen-eval, and save a QA manifest with config, command, git commit, sample count, solver iterations, tree all-in assertions, and output file list.

## Generation Command

CPU/release validation run:

```bash
cargo run -p cfvnet --release -- generate -c sample_configurations/boundary_net_river_datagen.yaml -o local_data/cfvnet/river_fuzzed_allin --num-samples 5000000 --threads 16 --per-file 10000
```

GPU/release variant, if available:

```bash
cargo run -p cfvnet --release --features gpu-datagen -- generate -c sample_configurations/boundary_net_river_datagen.yaml -o local_data/cfvnet/river_fuzzed_allin --num-samples 5000000 --per-file 10000
```

After generation, run:

```bash
cargo run -p cfvnet --release -- datagen-eval -d local_data/cfvnet/river_fuzzed_allin
```

Record git commit, config path/hash, command, sample count, solver_iterations, file count/size, and datagen-eval summary in a QA manifest next to the dataset.

## Notes

Two accidental partial files were created while drafting this bean and then deleted: `local_data/cfvnet/river_fuzzed_allin_UBlmD` and `local_data/cfvnet/river_fuzzed_allin_RdlLp`.
