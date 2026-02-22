# trace-hand Diagnostic Subcommand Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `trace-hand` CLI subcommand that traces one or more canonical hands through the full postflop pipeline (EHS features → bucket assignments → postflop EV) and outputs structured JSON.

**Architecture:** New trainer subcommand + new `hand_trace.rs` module in the trainer crate. Loads board/buckets/equity from abstraction cache, rebuilds trees + postflop values via `PostflopAbstraction::build()`, then computes per-hand per-texture trace data. All output is JSON to stdout.

**Tech Stack:** clap (CLI), serde_json (output), existing `PostflopAbstraction::build()`, `compute_all_flop_features()`, `bucket_ehs_centroids()`.

---

### Task 1: Add CLI subcommand definition

**Files:**
- Modify: `crates/trainer/src/main.rs` (Commands enum + dispatch arm)

**Step 1: Add the `TraceHand` variant to `Commands`**

Add after the `DiagBuckets` variant (~line 247):

```rust
    /// Trace hands through the full postflop pipeline (EHS → buckets → EV)
    TraceHand {
        /// YAML config file (same format as solve-preflop)
        #[arg(short, long)]
        config: PathBuf,
        /// Comma-separated canonical hand names (e.g. "AA,KQs,T9o")
        #[arg(long)]
        hands: String,
        /// Directory for abstraction cache
        #[arg(long, default_value = "cache/postflop")]
        cache_dir: PathBuf,
    },
```

**Step 2: Add the dispatch arm**

Add before the closing `}` of the match block (~line 665):

```rust
        Commands::TraceHand { config, hands, cache_dir } => {
            let yaml = std::fs::read_to_string(&config)?;
            let training: PreflopTrainingConfig = serde_yaml::from_str(&yaml)?;
            let pf_config = training.game.postflop_model
                .ok_or("config file has no postflop_model section")?;
            let hand_names: Vec<&str> = hands.split(',').map(str::trim).collect();
            hand_trace::run(&pf_config, &hand_names, &cache_dir)?;
        }
```

**Step 3: Add the module declaration**

Add near the top of `main.rs` with the other mod declarations:

```rust
mod hand_trace;
```

**Step 4: Create stub module**

Create `crates/trainer/src/hand_trace.rs` with:

```rust
//! Hand trace diagnostic: traces hands through EHS → buckets → postflop EV.

use std::path::Path;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;

/// Run the hand trace diagnostic.
pub fn run(
    _config: &PostflopModelConfig,
    _hand_names: &[&str],
    _cache_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("trace-hand: not yet implemented");
    Ok(())
}
```

**Step 5: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles, `trace-hand --help` shows the new subcommand.

**Step 6: Commit**

```
feat(trainer): add trace-hand CLI subcommand stub
```

---

### Task 2: Implement data loading and EHS feature computation

**Files:**
- Modify: `crates/trainer/src/hand_trace.rs`

**Step 1: Parse hand names and load abstraction**

Replace the stub `run` function:

```rust
use std::path::Path;

use poker_solver_core::hands::{CanonicalHand, all_hands};
use poker_solver_core::preflop::board_abstraction::{BoardAbstraction, BoardAbstractionConfig};
use poker_solver_core::preflop::hand_buckets::{
    self, BucketEquity, HandBucketMapping, compute_all_flop_features,
    bucket_ehs_centroids,
};
use poker_solver_core::preflop::postflop_abstraction::PostflopAbstraction;
use poker_solver_core::preflop::postflop_model::PostflopModelConfig;
use poker_solver_core::preflop::postflop_tree::PotType;
use poker_solver_core::preflop::{abstraction_cache, ehs::EhsFeatures};

/// Run the hand trace diagnostic.
pub fn run(
    config: &PostflopModelConfig,
    hand_names: &[&str],
    cache_dir: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    // Parse requested hands
    let all_hands_vec: Vec<CanonicalHand> = all_hands().collect();
    let targets: Vec<(String, usize)> = hand_names
        .iter()
        .map(|name| {
            let hand = CanonicalHand::parse(name)
                .map_err(|e| format!("invalid hand '{}': {}", name, e))?;
            let idx = all_hands_vec
                .iter()
                .position(|h| *h == hand)
                .ok_or_else(|| format!("hand '{}' not found in canonical list", name))?;
            Ok((name.to_string(), idx))
        })
        .collect::<Result<Vec<_>, String>>()?;

    eprintln!("Tracing {} hand(s): {}", targets.len(),
        targets.iter().map(|(n, _)| n.as_str()).collect::<Vec<_>>().join(", "));

    // Build the full postflop abstraction (loads board/buckets/equity from cache,
    // rebuilds trees + solves postflop)
    eprintln!("Building postflop abstraction...");
    let abstraction = PostflopAbstraction::build(
        config,
        None, // no equity table
        Some(cache_dir),
        |phase| eprintln!("  {:?}", phase),
    )?;

    // Compute EHS features for all hands (needed for per-hand trace)
    eprintln!("Computing EHS features...");
    let flop_samples: Vec<Vec<_>> = abstraction.board.prototype_flops
        .iter()
        .map(|f| vec![*f])
        .collect();
    let features = compute_all_flop_features(&all_hands_vec, &flop_samples, &|_| {});
    let num_buckets = config.num_hand_buckets_flop as usize;
    let centroids = bucket_ehs_centroids(&features, &abstraction.buckets.flop_buckets, num_buckets);

    // Build and print JSON output
    let output = build_trace_output(
        &targets, &all_hands_vec, &features, &abstraction, &centroids, config,
    );
    println!("{}", serde_json::to_string_pretty(&output)?);

    Ok(())
}
```

**Step 2: Verify it compiles (won't run yet — `build_trace_output` not implemented)**

Run: `cargo check -p poker-solver-trainer`
Expected: compile error for missing `build_trace_output` (that's Task 3).

**Step 3: Commit**

```
feat(trainer): implement trace-hand data loading and EHS computation
```

---

### Task 3: Build JSON output structure

**Files:**
- Modify: `crates/trainer/src/hand_trace.rs`

**Step 1: Define output types and implement `build_trace_output`**

Add the serde-serializable output types and the builder function:

```rust
use serde::Serialize;

#[derive(Serialize)]
struct TraceOutput {
    hands: Vec<HandTrace>,
}

#[derive(Serialize)]
struct HandTrace {
    hand: String,
    canonical_index: usize,
    textures: Vec<TextureTrace>,
    summary: HandSummary,
}

#[derive(Serialize)]
struct TextureTrace {
    texture_id: usize,
    prototype_flop: String,
    ehs_features: [f64; 3],
    blocked: bool,
    flop_bucket: u16,
    bucket_centroid_ehs: f64,
    postflop_ev: PostflopEvTrace,
}

#[derive(Serialize)]
struct PostflopEvTrace {
    limped: PositionEv,
    raised: PositionEv,
    three_bet: PositionEv,
    four_bet_plus: PositionEv,
}

#[derive(Serialize)]
struct PositionEv {
    sb: f64,
    bb: f64,
}

#[derive(Serialize)]
struct HandSummary {
    avg_ehs: f64,
    bucket_range: [u16; 2],
    blocked_texture_count: usize,
    avg_postflop_ev_raised: PositionEv,
}

fn build_trace_output(
    targets: &[(String, usize)],
    _all_hands: &[CanonicalHand],
    features: &[Vec<EhsFeatures>],
    abstraction: &PostflopAbstraction,
    centroids: &[f64],
    _config: &PostflopModelConfig,
) -> TraceOutput {
    let num_textures = abstraction.board.prototype_flops.len();
    let num_buckets = centroids.len();

    let hands: Vec<HandTrace> = targets
        .iter()
        .map(|(name, &hand_idx)| {
            let mut textures = Vec::with_capacity(num_textures);
            let mut ehs_sum = 0.0f64;
            let mut ehs_count = 0usize;
            let mut blocked_count = 0usize;
            let mut min_bucket = u16::MAX;
            let mut max_bucket = 0u16;
            let mut ev_raised_sb_sum = 0.0f64;
            let mut ev_raised_bb_sum = 0.0f64;
            let mut ev_count = 0usize;

            for tex_id in 0..num_textures {
                let feat = features[hand_idx][tex_id];
                let blocked = feat[0].is_nan();
                let bucket = abstraction.buckets.flop_buckets[hand_idx][tex_id];
                let centroid_ehs = centroids.get(bucket as usize).copied().unwrap_or(0.0);

                // Compute postflop EV for all pot types, averaging over opponent buckets
                let postflop_ev = compute_postflop_ev(
                    &abstraction, bucket, num_buckets,
                );

                if !blocked {
                    ehs_sum += feat[0];
                    ehs_count += 1;
                } else {
                    blocked_count += 1;
                }
                min_bucket = min_bucket.min(bucket);
                max_bucket = max_bucket.max(bucket);
                ev_raised_sb_sum += postflop_ev.raised.sb;
                ev_raised_bb_sum += postflop_ev.raised.bb;
                ev_count += 1;

                let flop_str = format_flop(&abstraction.board.prototype_flops[tex_id]);

                textures.push(TextureTrace {
                    texture_id: tex_id,
                    prototype_flop: flop_str,
                    ehs_features: if blocked { [f64::NAN, f64::NAN, f64::NAN] } else { feat },
                    blocked,
                    flop_bucket: bucket,
                    bucket_centroid_ehs: centroid_ehs,
                    postflop_ev,
                });
            }

            let avg_ehs = if ehs_count > 0 { ehs_sum / ehs_count as f64 } else { f64::NAN };
            let avg_ev_raised = if ev_count > 0 {
                PositionEv {
                    sb: ev_raised_sb_sum / ev_count as f64,
                    bb: ev_raised_bb_sum / ev_count as f64,
                }
            } else {
                PositionEv { sb: 0.0, bb: 0.0 }
            };

            HandTrace {
                hand: name.clone(),
                canonical_index: hand_idx,
                textures,
                summary: HandSummary {
                    avg_ehs,
                    bucket_range: [min_bucket, max_bucket],
                    blocked_texture_count: blocked_count,
                    avg_postflop_ev_raised: avg_ev_raised,
                },
            }
        })
        .collect();

    TraceOutput { hands }
}
```

**Step 2: Add `compute_postflop_ev` and `format_flop` helpers**

```rust
/// Average postflop EV for a hero bucket across all opponent buckets (uniform weighting).
#[allow(clippy::cast_precision_loss)]
fn compute_postflop_ev(
    abstraction: &PostflopAbstraction,
    hero_bucket: u16,
    num_buckets: usize,
) -> PostflopEvTrace {
    let pot_types = [
        (PotType::Limped, "limped"),
        (PotType::Raised, "raised"),
        (PotType::ThreeBet, "three_bet"),
        (PotType::FourBetPlus, "four_bet_plus"),
    ];

    let mut evs = [[0.0f64; 2]; 4]; // [pot_type_idx][pos]
    for (pt_idx, &(pot_type, _)) in pot_types.iter().enumerate() {
        for pos in 0..2u8 {
            let sum: f64 = (0..num_buckets as u16)
                .map(|opp| abstraction.values.get(pot_type, pos, hero_bucket, opp))
                .sum();
            evs[pt_idx][pos as usize] = sum / num_buckets as f64;
        }
    }

    PostflopEvTrace {
        limped: PositionEv { sb: evs[0][0], bb: evs[0][1] },
        raised: PositionEv { sb: evs[1][0], bb: evs[1][1] },
        three_bet: PositionEv { sb: evs[2][0], bb: evs[2][1] },
        four_bet_plus: PositionEv { sb: evs[3][0], bb: evs[3][1] },
    }
}

fn format_flop(flop: &[poker_solver_core::poker::Card; 3]) -> String {
    flop.iter().map(|c| format!("{c}")).collect::<Vec<_>>().join(" ")
}
```

**Step 3: Add serde_json dependency to trainer Cargo.toml if not already present**

Check `crates/trainer/Cargo.toml` — if `serde_json` is already listed, skip this step. Otherwise add:

```toml
serde_json = "1"
```

Also ensure `serde` with `derive` feature is available.

**Step 4: Verify it compiles**

Run: `cargo build -p poker-solver-trainer`
Expected: compiles successfully.

**Step 5: Commit**

```
feat(trainer): implement trace-hand JSON output with per-texture EHS/bucket/EV trace
```

---

### Task 4: Test with sample config

**Step 1: Run trace-hand on fast_buckets config**

```bash
cargo run -p poker-solver-trainer --release -- trace-hand \
  -c sample_configurations/fast_buckets.yaml \
  --hands "AA,KQs,72o" \
  --cache-dir cache/postflop
```

Expected: JSON output to stdout with 3 hand entries, each containing per-texture trace data. AA should have high avg_ehs (~0.85+), 72o should have low avg_ehs (~0.35-). AA should not show bucket assignments in the bottom quartile.

**Step 2: Verify blocked textures show correctly**

Check the JSON output for AA — some textures should show `"blocked": true` where the prototype flop contains both an ace of spades and ace of hearts (or similar). Verify these get reasonable bucket assignments (not the weakest bucket).

**Step 3: Pipe through jq for quick checks**

```bash
# AA's blocked texture count
cargo run -p poker-solver-trainer --release -- trace-hand \
  -c sample_configurations/fast_buckets.yaml \
  --hands "AA" --cache-dir cache/postflop | jq '.hands[0].summary'

# All textures where AA is blocked
... | jq '.hands[0].textures[] | select(.blocked)'
```

**Step 4: Commit (if any fixes needed)**

```
fix(trainer): trace-hand output corrections
```

---

### Task 5: Handle Card Display formatting

**Files:**
- Modify: `crates/trainer/src/hand_trace.rs`

The `format_flop` function uses `{c}` Display formatting. Verify that `Card` implements `Display`. If it doesn't, use the explicit formatting from `bucket_diagnostics.rs` or `hand_buckets.rs` (the `format_card` function).

Check `poker_solver_core::poker::Card` for `Display` impl. If not available, copy the rank/suit match pattern from `hand_buckets.rs:format_card` (lines 747-770).

**Step 1: Verify card formatting**

Run: `cargo build -p poker-solver-trainer`

If `Card` doesn't implement Display, replace `format_flop`:

```rust
use poker_solver_core::poker::{Card, Suit, Value};

fn format_card(card: &Card) -> String {
    let rank = match card.value {
        Value::Ace => 'A', Value::King => 'K', Value::Queen => 'Q',
        Value::Jack => 'J', Value::Ten => 'T', Value::Nine => '9',
        Value::Eight => '8', Value::Seven => '7', Value::Six => '6',
        Value::Five => '5', Value::Four => '4', Value::Three => '3',
        Value::Two => '2',
    };
    let suit = match card.suit {
        Suit::Spade => 's', Suit::Heart => 'h',
        Suit::Diamond => 'd', Suit::Club => 'c',
    };
    format!("{rank}{suit}")
}

fn format_flop(flop: &[Card; 3]) -> String {
    flop.iter().map(format_card).collect::<Vec<_>>().join(" ")
}
```

**Step 2: Commit if changed**

```
fix(trainer): use explicit card formatting in trace-hand
```

---

## Notes

- **Performance:** The bottleneck is `PostflopAbstraction::build()` which rebuilds trees and solves postflop CFR (~200 iterations). Board/buckets/equity load from cache instantly. With `fast_buckets.yaml` settings this takes ~10-30 seconds in release mode.
- **Future improvement:** Cache the full `PostflopAbstraction` (trees + values) to make `trace-hand` instant on repeat runs. Out of scope for this task.
- **NaN in JSON:** `f64::NAN` serializes as `null` in JSON via serde_json, which is the correct behavior for blocked textures.
