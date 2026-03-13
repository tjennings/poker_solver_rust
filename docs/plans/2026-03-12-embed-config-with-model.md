# Embed Config With Model Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use hex:executing-plans to implement this plan task-by-task.

**Goal:** Save CfvnetConfig alongside trained models so compare/evaluate commands auto-load the correct config, eliminating hardcoded values and default-related bugs.

**Architecture:** Training writes `config.yaml` to the model output directory. Compare and evaluate commands load it via a shared `load_model_config()` helper. Hardcoded model architecture params, seeds, and solver params are replaced with config values throughout.

**Tech Stack:** Rust, serde_yaml, clap

---

### Task 1: Add `load_model_config` helper and save config at training time

**Files:**
- Modify: `crates/cfvnet/src/main.rs:198-263` (cmd_train)
- Modify: `crates/cfvnet/src/main.rs:530-554` (load_config_or_default — will be replaced)

**Step 1: Write `load_model_config` function**

Add this function in `main.rs`, replacing `load_config_or_default`:

```rust
fn load_model_config(model_dir: &std::path::Path) -> cfvnet::config::CfvnetConfig {
    let config_path = model_dir.join("config.yaml");
    let yaml = std::fs::read_to_string(&config_path).unwrap_or_else(|e| {
        eprintln!(
            "failed to read model config {}: {e}\n\
             hint: this model was saved before config embedding was added — \
             re-train or manually place a config.yaml in the model directory",
            config_path.display()
        );
        std::process::exit(1);
    });
    serde_yaml::from_str(&yaml).unwrap_or_else(|e| {
        eprintln!("failed to parse model config {}: {e}", config_path.display());
        std::process::exit(1);
    })
}
```

**Step 2: Save config in `cmd_train`**

At the end of `cmd_train`, after the match block that runs training (after line 262), add:

```rust
let config_out = output.join("config.yaml");
let yaml = serde_yaml::to_string(&cfg).unwrap_or_else(|e| {
    eprintln!("failed to serialize config: {e}");
    std::process::exit(1);
});
std::fs::write(&config_out, &yaml).unwrap_or_else(|e| {
    eprintln!("failed to write config to {}: {e}", config_out.display());
    std::process::exit(1);
});
println!("Config saved to {}", config_out.display());
```

**Step 3: Delete `load_config_or_default`**

Remove the function at lines 530-554.

**Step 4: Run tests to verify compilation**

Run: `cargo test -p cfvnet --lib -- --quiet 2>&1 | tail -5`
Expected: All existing tests pass (no behavior change yet)

**Step 5: Commit**

```bash
git add crates/cfvnet/src/main.rs
git commit -m "feat: save config.yaml with model and add load_model_config helper"
```

---

### Task 2: Remove hardcoded values from `compare.rs`

**Files:**
- Modify: `crates/cfvnet/src/eval/compare.rs:11-28` (generate_comparison_spot, default_solve_config)
- Modify: `crates/cfvnet/src/eval/compare.rs:66-76` (run_comparison)
- Modify: `crates/cfvnet/src/eval/compare.rs:113-165` (tests)

**Step 1: Fix `generate_comparison_spot` to derive board_size from street**

Change line 13 from:
```rust
sample_situation(datagen, initial_stack, 5, &mut rng)
```
to:
```rust
let board_size = crate::config::board_cards_for_street(&datagen.street);
sample_situation(datagen, initial_stack, board_size, &mut rng)
```

**Step 2: Fix `default_solve_config` to use datagen params**

Change the function signature and body. Replace lines 17-28:

```rust
pub fn default_solve_config(game: &GameConfig, datagen: &DatagenConfig) -> Result<SolveConfig, String> {
    let bet_str = game.bet_sizes.join(",");
    let bet_sizes = BetSizeOptions::try_from((bet_str.as_str(), ""))
        .map_err(|e| format!("invalid bet sizes: {e}"))?;
    Ok(SolveConfig {
        bet_sizes,
        solver_iterations: datagen.solver_iterations,
        target_exploitability: datagen.target_exploitability,
        add_allin_threshold: game.add_allin_threshold,
        force_allin_threshold: game.force_allin_threshold,
    })
}
```

**Step 3: Update `run_comparison` to pass datagen to `default_solve_config`**

Change line 76 from:
```rust
let solve_config = default_solve_config(game_config)?;
```
to:
```rust
let solve_config = default_solve_config(game_config, datagen)?;
```

**Step 4: Update tests**

The tests use `DatagenConfig::default()` which is fine — they just need to ensure board_size is 5 (river) since `default_street()` returns `"river"`. The existing tests should still pass without changes since `DatagenConfig::default()` has `street: "river"` and `solver_iterations: 1000` / `target_exploitability: 0.005`.

**Step 5: Run tests**

Run: `cargo test -p cfvnet --lib -- compare --quiet 2>&1 | tail -10`
Expected: All compare tests pass

**Step 6: Commit**

```bash
git add crates/cfvnet/src/eval/compare.rs
git commit -m "fix: use datagen config for board_size, solver_iterations, target_exploitability in compare"
```

---

### Task 3: Update `cmd_compare` — remove `--config`, use model config, eliminate hardcoded values

**Files:**
- Modify: `crates/cfvnet/src/main.rs:64-78` (Compare CLI variant)
- Modify: `crates/cfvnet/src/main.rs:178-183` (match arm)
- Modify: `crates/cfvnet/src/main.rs:418-489` (cmd_compare function)

**Step 1: Remove `config` from the Compare CLI variant**

Remove lines 75-77:
```rust
        /// Optional YAML config for game parameters
        #[arg(short, long)]
        config: Option<PathBuf>,
```

**Step 2: Update the match arm**

Change lines 178-183 from:
```rust
Commands::Compare {
    model,
    num_spots,
    threads,
    config,
} => cmd_compare(model, num_spots, threads, config),
```
to:
```rust
Commands::Compare {
    model,
    num_spots,
    threads,
} => cmd_compare(model, num_spots, threads),
```

**Step 3: Rewrite `cmd_compare`**

Replace the function (lines 418-489):

```rust
fn cmd_compare(
    model_dir: PathBuf,
    num_spots: usize,
    threads: Option<usize>,
) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::config::board_cards_for_street;
    use cfvnet::eval::compare::run_comparison;
    use cfvnet::model::dataset::encode_situation_for_inference;
    use cfvnet::model::network::{CfvNet, input_size};

    let cfg = load_model_config(&model_dir);
    let board_cards = board_cards_for_street(&cfg.datagen.street);
    let in_size = input_size(board_cards);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(&model_dir);

    let model = CfvNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size, in_size)
        .load_file(&model_path, &recorder, &device)
        .unwrap_or_else(|e| {
            eprintln!("failed to load model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    if let Some(t) = threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .ok();
    }

    println!("Comparing {num_spots} spots against exact solver...");

    let summary = run_comparison(&cfg.game, &cfg.datagen, num_spots, cfg.datagen.seed, |sit, _solve_result| {
        let input_data = encode_situation_for_inference(sit, 0);
        let input = Tensor::<B, 2>::from_data(
            TensorData::new(input_data, [1, in_size]),
            &device,
        );
        let pred = model.forward(input);
        pred.into_data().to_vec::<f32>().unwrap()
    })
    .unwrap_or_else(|e| {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    });

    print_summary(&summary);
}
```

Key changes vs old:
- No `config_path` parameter
- `load_model_config(&model_dir)` replaces manual config loading
- `cfg.training.hidden_layers, cfg.training.hidden_size` replaces `7, 500`
- `cfg.datagen.seed` replaces `42`
- `board_cards_for_street(&cfg.datagen.street)` replaces `5`

**Step 4: Verify compilation**

Run: `cargo check -p cfvnet 2>&1 | tail -5`
Expected: Compiles cleanly

**Step 5: Commit**

```bash
git add crates/cfvnet/src/main.rs
git commit -m "fix: cmd_compare loads config from model dir, removes all hardcoded values"
```

---

### Task 4: Update `cmd_compare_net` and `cmd_compare_exact` — remove `--config`, use model config

**Files:**
- Modify: `crates/cfvnet/src/main.rs:79-105` (CompareNet, CompareExact CLI variants)
- Modify: `crates/cfvnet/src/main.rs:184-194` (match arms)
- Modify: `crates/cfvnet/src/main.rs:491-528` (cmd_compare_net, cmd_compare_exact)

**Step 1: Remove `config` from CompareNet and CompareExact CLI variants**

From CompareNet, remove:
```rust
        /// Optional YAML config
        #[arg(short, long)]
        config: Option<PathBuf>,
```

From CompareExact, remove:
```rust
        /// Optional YAML config
        #[arg(short, long)]
        config: Option<PathBuf>,
```

**Step 2: Update match arms**

CompareNet:
```rust
Commands::CompareNet {
    model,
    river_model,
    num_spots,
} => cmd_compare_net(model, river_model, num_spots),
```

CompareExact:
```rust
Commands::CompareExact {
    model,
    num_spots,
} => cmd_compare_exact(model, num_spots),
```

**Step 3: Rewrite `cmd_compare_net`**

```rust
fn cmd_compare_net(
    model_dir: PathBuf,
    river_model_dir: PathBuf,
    num_spots: usize,
) {
    use cfvnet::eval::compare_turn::run_turn_comparison_net;

    let cfg = load_model_config(&model_dir);
    let model_path = resolve_model_path(&model_dir);
    let river_path = resolve_model_path(&river_model_dir);

    println!("Comparing {num_spots} turn spots against CfvSubgameSolver + RiverNetEvaluator...");

    let summary = run_turn_comparison_net(&cfg, &model_path, &river_path, num_spots, cfg.datagen.seed)
        .unwrap_or_else(|e| {
            eprintln!("comparison failed: {e}");
            std::process::exit(1);
        });

    print_summary(&summary);
}
```

**Step 4: Rewrite `cmd_compare_exact`**

```rust
fn cmd_compare_exact(model_dir: PathBuf, num_spots: usize) {
    use cfvnet::eval::compare_turn::run_turn_comparison_exact;

    let cfg = load_model_config(&model_dir);
    let model_path = resolve_model_path(&model_dir);

    println!("Comparing {num_spots} turn spots against CfvSubgameSolver + ExactRiverEvaluator...");

    let summary = run_turn_comparison_exact(&cfg, &model_path, num_spots, cfg.datagen.seed).unwrap_or_else(|e| {
        eprintln!("comparison failed: {e}");
        std::process::exit(1);
    });

    print_summary(&summary);
}
```

**Step 5: Verify compilation**

Run: `cargo check -p cfvnet 2>&1 | tail -5`
Expected: Compiles cleanly

**Step 6: Commit**

```bash
git add crates/cfvnet/src/main.rs
git commit -m "fix: cmd_compare_net/exact load config from model dir, use datagen.seed"
```

---

### Task 5: Update `cmd_evaluate` — load config from model dir, eliminate hardcoded architecture

**Files:**
- Modify: `crates/cfvnet/src/main.rs:348-416` (cmd_evaluate)

**Step 1: Rewrite `cmd_evaluate`**

```rust
fn cmd_evaluate(model_dir: PathBuf, data_path: PathBuf) {
    use burn::backend::NdArray;
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, NamedMpkGzFileRecorder};
    use burn::tensor::{Tensor, TensorData};
    use cfvnet::config::board_cards_for_street;
    use cfvnet::eval::metrics::compute_prediction_metrics;
    use cfvnet::model::dataset::CfvDataset;
    use cfvnet::model::network::{CfvNet, input_size};

    let cfg = load_model_config(&model_dir);
    let board_cards = board_cards_for_street(&cfg.datagen.street);
    let in_size = input_size(board_cards);

    type B = NdArray;
    let device = Default::default();
    let recorder = NamedMpkGzFileRecorder::<FullPrecisionSettings>::new();
    let model_path = resolve_model_path(&model_dir);

    let model = CfvNet::<B>::new(&device, cfg.training.hidden_layers, cfg.training.hidden_size, in_size)
        .load_file(&model_path, &recorder, &device)
        .unwrap_or_else(|e| {
            eprintln!("failed to load model from {}: {e}", model_path.display());
            std::process::exit(1);
        });

    let dataset = CfvDataset::from_file(&data_path, board_cards).unwrap_or_else(|e| {
        eprintln!("failed to load dataset: {e}");
        std::process::exit(1);
    });

    println!("Evaluating {} records...", dataset.len());

    let mut total_mae = 0.0_f64;
    let mut total_max_error = 0.0_f64;
    let mut total_mbb = 0.0_f64;
    let mut count = 0_u64;

    for i in 0..dataset.len() {
        let item = dataset.get(i).unwrap();

        let input = Tensor::<B, 2>::from_data(
            TensorData::new(item.input.clone(), [1, in_size]),
            &device,
        );
        let pred = model.forward(input);
        let pred_vec: Vec<f32> = pred.into_data().to_vec::<f32>().unwrap();

        let mask: Vec<bool> = item.mask.iter().map(|&v| v > 0.5).collect();
        let pot = item.input[2 * 1326 + board_cards] * cfg.game.initial_stack as f32;

        let metrics = compute_prediction_metrics(&pred_vec, &item.target, &mask, pot);
        total_mae += metrics.mae;
        total_max_error += metrics.max_error;
        total_mbb += metrics.mbb_error;
        count += 1;
    }

    if count == 0 {
        println!("No records to evaluate.");
        return;
    }

    let n = count as f64;
    println!("Results ({count} records):");
    println!("  MAE:       {:.6}", total_mae / n);
    println!("  Max Error: {:.6}", total_max_error / n);
    println!("  mBB:       {:.2}", total_mbb / n);
}
```

Key changes:
- `cfg.training.hidden_layers, cfg.training.hidden_size` replaces `7, 500`
- `board_cards_for_street(&cfg.datagen.street)` replaces hardcoded `5`
- Pot denormalization: need to verify the encoding formula. Check `encode_situation_for_inference` to confirm the correct index and denormalization factor.

**Step 2: Verify pot denormalization is correct**

Before committing, read `crates/cfvnet/src/model/dataset.rs` to find `encode_situation_for_inference` and confirm:
- The index where `pot` is stored (currently assumed `2657 = 2*1326 + 5`)
- The normalization factor (currently `400.0` — should match `initial_stack` or the max pot range)

Update the pot line accordingly. The formula `item.input[2 * 1326 + board_cards] * cfg.game.initial_stack as f32` assumes pot is normalized by `initial_stack`. Verify and adjust.

**Step 3: Run tests**

Run: `cargo test -p cfvnet --lib -- --quiet 2>&1 | tail -5`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/cfvnet/src/main.rs
git commit -m "fix: cmd_evaluate loads config from model dir, removes hardcoded 7/500 architecture"
```

---

### Task 6: Full test suite and cleanup

**Step 1: Run the full test suite**

Run: `cargo test 2>&1 | tail -20`
Expected: All tests pass

**Step 2: Run clippy**

Run: `cargo clippy -p cfvnet 2>&1 | tail -10`
Expected: No warnings

**Step 3: Verify `load_config_or_default` is fully removed**

Search for any remaining references:
Run: `grep -rn "load_config_or_default" crates/cfvnet/`
Expected: No results

**Step 4: Verify no remaining hardcoded `7, 500` in main.rs**

Run: `grep -n "7, 500" crates/cfvnet/src/main.rs`
Expected: No results

**Step 5: Verify no remaining hardcoded seed `42` in compare calls**

Run: `grep -n ", 42," crates/cfvnet/src/main.rs`
Expected: No results (test files may still have `42` which is fine)

**Step 6: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: remove load_config_or_default, final cleanup"
```
