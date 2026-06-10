# Training Reference

All commands are run via the `poker-solver-trainer` crate:

```bash
cargo run -p poker-solver-trainer --release -- <subcommand> [options]
```

Always use `--release` for training and diagnostics.

## Blueprint Bundle Formats

The current production HU bundle is the legacy `blueprint_v2` layout containing
`config.yaml`, `strategy.bin`, metadata, and snapshot directories. The planned
universal dense strategy format is specified in `docs/blueprint_format.md`.
That format is a versioned directory bundle with `blueprint.json`,
row/action/probability binary payloads, explicit player/action/bucket
provenance, checksums, and separate optional resumable CFR state. Until the
implementation phases land, `train-blueprint` and `train-blueprint-mp` continue
to use their existing snapshot/export behavior.

## Commands

### train-blueprint

Train a blueprint strategy using MCCFR. See `sample_configurations/blueprint_v2_with_tui.yaml` for a complete config example.

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint \
  -c sample_configurations/blueprint_v2_with_tui.yaml
```

### train-blueprint-mp

Train a multiplayer (2-8 player) blueprint strategy using external-sampling MCCFR.

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint-mp \
  -c <config.yaml>
```

### export-universal

Export a legacy HU `blueprint_v2` bundle into the universal dense blueprint
format (`docs/blueprint_format.md`). Probabilities are passed through bitwise
from the snapshot's `strategy.bin`; the legacy bundle is not modified.

```bash
cargo run -p poker-solver-trainer --release -- export-universal \
  --bundle <legacy_bundle_dir> \
  --snapshot final \            # or snapshot_NNNN (default: final)
  --out <universal_bundle_dir>
```

The output directory contains `blueprint.json`, `strategy.rows.bin`,
`strategy.actions.bin`, `strategy.probs.f32.bin`, and `checksums.json`. The
export is analysis-only (no `cfr.snapshot.bin`); dense HU exports use the
`reject` missing-row policy.

### inspect-mp-config

Inspect a multiplayer blueprint config before training. This reports effective stack depth, bucket counts, action-row counts, and known eager-backend risk patterns before the trainer builds dense tree/storage structures.

```bash
cargo run -p poker-solver-trainer --release -- inspect-mp-config \
  -c sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml
```

#### Config Format

The N-player config uses a different format from the 2-player `train-blueprint` command:

```yaml
game:
  name: "6-max 100bb BB-ante"
  num_players: 6
  stack_depth: 200        # chips (1 BB = 2 chips)
  allow_preflop_limp: false
  blinds:
    - seat: 0
      type: small_blind
      amount: 1
    - seat: 1
      type: big_blind
      amount: 2
    - seat: 1
      type: bb_ante
      amount: 2

action_abstraction:
  max_flop_players: 3           # optional preflop cap; omit for uncapped
  preflop:
    lead: ["5bb", "6bb"]        # opening raise sizes
    raise:
      - ["3.0x"]                 # 3-bet sizes (first raise depth)
      - ["2.5x"]                 # 4-bet+ sizes (repeats for deeper)
  flop:
    lead: [0.33, 0.67, 1.0]     # pot fractions for opening bets
    raise:
      - [0.5, 1.0, 2.0]         # raise sizes (first raise)
  turn:
    lead: [0.5, 1.0]
    raise:
      - [0.67, 1.0]
  river:
    lead: [0.5, 1.0]
    raise:
      - [1.0]

clustering:
  preflop:
    buckets: 169
  flop:
    buckets: 200
  turn:
    buckets: 200
  river:
    buckets: 200

training:
  backend: lazy_sparse     # eager (default) or lazy_sparse
  iterations: 100000
  batch_size: 200
  dcfr_alpha: 1.5
  dcfr_beta: 0.0
  dcfr_gamma: 2.0
  lcfr_warmup_iterations: 5000000

snapshots:
  warmup_minutes: 60
  snapshot_every_minutes: 30
  output_dir: "/data/blueprint_mp_6p"
```

#### Key Differences from `train-blueprint`

| Feature | `train-blueprint` (v2) | `train-blueprint-mp` |
|---------|----------------------|---------------------|
| Players | 2 only | 2-8 |
| Blind structure | `small_blind` + `big_blind` fields | Per-seat `blinds` list with types |
| Bet sizing | Per-street, indexed by raise depth | Lead/raise split per street |
| Info key | 64-bit, 6 action slots | 128-bit, 22 action slots |
| Average strategy storage | Signed 64-bit sums | Saturating unsigned 64-bit sums |
| Side pots | N/A (2 players) | Full multi-way resolution |

#### Sample Configs

- `sample_configurations/blueprint_mp_3player.yaml` -- 3-player 50bb test
- `sample_configurations/blueprint_mp_6player_ante.yaml` -- 6-player 100bb with BB-ante
- `sample_configurations/blueprint_mp_6max_simplified_actions.yaml` -- 6-max 20bb trainer using shared 500/50/50 postflop buckets, compact action sets, and TUI scenarios
- `sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml` -- 6-max 20bb trainer using shared 500/100/100 postflop buckets for the current bucket-quality experiment
- `sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml` -- 6-max lazy-sparse trainer using shared 250/100/20 postflop buckets for lower memory and faster iteration while pruning work continues
- `sample_configurations/blueprint_mp_6max_100bb_lazy_sparse_smoke.yaml` -- 6-max 100bb lazy-sparse regression smoke with two preflop raise rows and one training iteration

#### 100bb Status

100bb is a target stack depth for Blueprint MP. Use `training.backend: lazy_sparse` for 100bb 6-max configs with multiple preflop raise depths; it generates public states on demand and stores only visited infosets. The default `eager` backend still builds the complete public betting tree and dense regret/strategy storage before training, so `inspect-mp-config` will block known 100bb-scale dense-risk patterns unless `lazy_sparse` is selected.

Set `action_abstraction.max_flop_players` to cap how many active players can continue from preflop to the flop. When set, preflop non-closing calls that would consume the last allowed flop-player slot are removed, while action-closing calls are still allowed up to the cap. Omitting the field preserves uncapped action generation.

Lazy sparse DCFR discounting runs in parallel across sparse storage shards. The `discount` timing field in no-TUI telemetry is the wall-clock measurement to watch when checking whether discount passes are still causing single-core pauses. Lazy sparse MP training uses the shared runtime, so `training.time_limit_minutes` stops both no-TUI and TUI runs between lazy batches.

In `--no-tui` mode, lazy sparse progress is reported once per minute with sparse entries, slot counts, approximate storage, allocation growth rates, shard distribution, storage activity, insert attribution, action-limit audit fields, timing buckets for batch wall time, deal sampling, bucket lookup, traversal, DCFR discounting, and console stats collection, plus long-tail traversal telemetry (`max_job`, `max_trav`, and slow counts). Sparse entries, slots, and shard occupancy are maintained with live counters, so heartbeat stats stay O(shards) instead of scanning every visited infoset. The `activity[...]` block reports sparse read probe rate, read hit rate, write probe rate, write hit rate, and insert rate for the heartbeat interval. The `insert_by[...]` block attributes newly allocated infosets by street, top seat, top history-length bin, and action-count shape. Lazy sparse strategy keys use seat, a street-namespaced abstract bucket, and action history; SPR is not part of storage identity. River SPR-0 states suppress new lead/raise/all-in aggression while preserving check, fold, call, and all-in-call resolution, which keeps low-SPR river histories from expanding into many strategically similar betting branches. The `action_limit[...]` block audits max observed per-street raise counts and any decisions/aggressive actions beyond configured raise rows plus one all-in aggression allowance. When the negative-action subtree purge experiment is enabled, the `neg_action[...]` block reports `blocked_edges`, cumulative and per-second `new_pruned`, `reactivated`, `purge_calls`, `rows_purged`, `regret_slots_purged`, `strategy_slots_purged`, `blocked_skips`, and purge scan time as `purge_scan=<interval>/<total>`. Purge scans run at the lazy DCFR discount boundary, batch all still-blocked child prefixes into one sparse-storage pass for that boundary, and include their wall time in the lazy discount timing bucket. These fields help diagnose whether throughput dips line up with sparse storage growth, shard imbalance, compute phases, reporting overhead, new allocation pressure, lookup pressure, action-history/key explosion, action-limit escape, purge scans, or a single long traversal holding the batch barrier.

When `tui.enabled: true`, lazy sparse MP training launches the multiplayer TUI instead of no-TUI logs. The lazy sparse TUI shows live iterations, throughput, sampled regret telemetry, prune percentage, sampled strategy-delta movement, and configured scenario hand grids without materializing the dense public tree. Scenario grids resolve configured spots against the lazy public state and read average strategy from sparse infoset keys. The metrics panel also shows compact raw strategy probes for each configured scenario using `tui.strategy_probe_hands` (default: selected suited Ax, K9s, 22, 72o); each probe reports dominant average action (`a`), dominant current regret-matched action (`c`), total strategy-sum mass (`s`), and whether the sparse row is present (`P`), missing/uniform (`M`), or present with zero strategy-sum mass (`Z`). Pressing `p` pauses or resumes the lazy runtime between batches. Pressing `s` in lazy sparse TUI writes a sparse checkpoint containing `sparse_entries.bin` and `metadata.json`; it does not synthesize the HU-style dense `strategy.bin` bundle. The hotkey line reports the manual snapshot lifecycle as queued, writing, saved with the `snapshot_NNNN` directory name, or failed with a concise error. Lazy sparse resume remains unsupported: sparse snapshots do not persist blocked-edge purge state or full runtime/cadence metadata.

The universal dense format will eventually let lazy sparse MP write read-only
analysis bundles without materializing an eager public tree. Those exports are
not resumable until the missing blocked-edge purge state and runtime cadence are
part of the snapshot contract.

Run the 500/100/100 6-max experiment with:

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint-mp \
  -c sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml
```

Run the 250/100/20 6-max experiment with:

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint-mp \
  -c sample_configurations/blueprint_mp_6max_250f_100t_20r.yaml
```

---

### range-solve

Solve a postflop spot with exact (no abstraction) Discounted CFR. Uses the `range-solver` crate -- a self-contained reimplementation of b-inary/postflop-solver producing identical output.

Solves a **single spot** with full hand granularity (1326 hole card combos, no bucketing) and suit isomorphism reduction.

```bash
# River spot with specific ranges
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs,AKo" \
  --ip-range "22+,A2s+,KQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 200 \
  --iterations 1000

# Flop spot (turn + river solved internally via chance nodes)
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "AA,KK,QQ,AKs" \
  --ip-range "TT-66,AQs-ATs,KQs,QJs" \
  --flop "Qs Jh 2c" \
  --pot 100 --effective-stack 300 \
  --iterations 500

# Custom bet sizing
cargo run -p poker-solver-trainer --release -- range-solve \
  --oop-range "QQ+,AKs" --ip-range "22+,A2s+" \
  --flop "Ah Kd 7c" --turn "2s" \
  --pot 80 --effective-stack 160 \
  --oop-bet-sizes "33%,67%,a" --oop-raise-sizes "2.5x" \
  --ip-bet-sizes "33%,67%,a" --ip-raise-sizes "2.5x" \
  --iterations 1000 --target-exploitability 0.3
```

Options:
- `--oop-range <RANGE>` -- OOP player's range in PioSOLVER format (required)
- `--ip-range <RANGE>` -- IP player's range (required)
- `--flop <CARDS>` -- Flop cards, e.g. `"Qs Jh 2c"` (required)
- `--turn <CARD>` -- Turn card, e.g. `"8d"` (optional)
- `--river <CARD>` -- River card, e.g. `"3s"` (optional; requires `--turn`)
- `--pot <N>` -- Starting pot size (default: 100)
- `--effective-stack <N>` -- Effective stack size (default: 100)
- `--iterations <N>` -- Maximum DCFR iterations (default: 1000)
- `--target-exploitability <F>` -- Stop early when exploitability drops below this (default: 0.5)
- `--oop-bet-sizes <SIZES>` -- OOP bet sizes, comma-separated (default: `"50%,100%"`)
- `--oop-raise-sizes <SIZES>` -- OOP raise sizes (default: `"60%,100%"`)
- `--ip-bet-sizes <SIZES>` -- IP bet sizes (default: `"50%,100%"`)
- `--ip-raise-sizes <SIZES>` -- IP raise sizes (default: `"60%,100%"`)
- `--compressed` -- Use 16-bit compressed storage (less memory, slightly less precision)

**Bet size syntax:**
| Format | Meaning | Example |
|-|-|-|
| `N%` | Pot-relative | `50%` = half pot |
| `Nx` | Previous-bet-relative (raises only) | `2.5x` = 2.5x previous bet |
| `Ne` | Geometric over N streets | `2e` = geometric over 2 streets |
| `Nc` | Additive (chips) | `100c` = 100 chips |
| `a` | All-in | |

**Output:** Per-iteration exploitability, then a per-hand strategy table at the root node showing action probabilities for each hole card combo.

**Street determination:** Automatically set from which cards are provided:
- Flop only -> solves from flop (turn + river as chance nodes)
- Flop + turn -> solves from turn (river as chance node)
- Flop + turn + river -> solves river only (fastest)

**Algorithm:** Discounted CFR with a=1.5, b=0.5, g=3.0. Strategy resets at power-of-4 iterations. Multithreaded via rayon.

---

### bench-rollout

Benchmark the rollout boundary evaluator in isolation (does not run DCFR). Loads a blueprint bundle and drives the rollout evaluator in a tight loop for a bounded wall time, reporting throughput metrics: ms/call, calls/sec, total hands, and hands/sec.

```bash
cargo run -p poker-solver-trainer --release -- bench-rollout \
  --bundle /path/to/blueprint/bundle \
  --duration-secs 10 \
  --board Ks7h2c \
  --pot 100 --stacks 200 \
  --enumerate-depth 2 \
  --opponent-samples 8
```

Options:
- `--bundle <DIR>` -- Path to blueprint bundle directory (must contain `config.yaml`, `strategy.bin` or `snapshot_*/strategy.bin`, and `buckets/`) (required)
- `--duration-secs <N>` -- Wall-time duration in seconds (default: 10)
- `--board <CARDS>` -- Flop board cards, e.g. `"Ks7h2c"` (default: `Ks7h2c`)
- `--pot <N>` -- Starting pot size in chips (default: 100)
- `--stacks <N>` -- Starting stack per player in chips (default: 200)
- `--enumerate-depth <N>` -- Decision levels to enumerate before sampling (default: 2). Higher = more accurate but slower.
- `--opponent-samples <N>` -- Opponent hands sampled per hero combo (default: 8). Higher = less variance but slower.

---

### validate-rollout

Compare sampled rollout CFVs against exhaustive (exact) rollout CFVs per combo. Runs the exhaustive evaluator once as a baseline, then runs the sampled evaluator multiple times and aggregates the results to separate stochastic noise from systematic bias. Reports max/mean/L2 diffs in both pot-fraction and mbb/hand units.

The current PASS criterion is `max_abs_diff < 2 mbb/hand` (strict). In practice, `mean_abs_diff` under 1 mbb/hand is the more important signal for DCFR convergence, since the outer solver averages over many iterations.

```bash
cargo run -p poker-solver-trainer --release -- validate-rollout \
  --bundle /path/to/blueprint/bundle \
  --board Ks7h2c \
  --pot 100 --stacks 200 \
  --num-runs 5 \
  --enumerate-depth 2 \
  --opponent-samples 8
```

Options:
- `--bundle <DIR>` -- Path to blueprint bundle directory (required)
- `--board <CARDS>` -- Flop board cards (default: `Ks7h2c`)
- `--pot <N>` -- Starting pot size in chips (default: 100)
- `--stacks <N>` -- Starting stack per player in chips (default: 200)
- `--num-runs <N>` -- Number of sampled runs to aggregate (default: 5)
- `--pass-threshold <F>` -- Pass threshold for max_abs_diff in pot-fraction units (default: 0.02, i.e. 2 mbb/hand)
- `--enumerate-depth <N>` -- Decision levels to enumerate before sampling (default: 2)
- `--opponent-samples <N>` -- Opponent hands sampled per hero combo (default: 8)

**Output:** Per-traverser (OOP, IP) reports showing nonzero combos, max/mean/L2 diffs with stddev across runs, and PASS/FAIL verdict.

---

### Rollout Config Parameters

When using the Tauri explorer or dev server for subgame solving, the following rollout parameters are configurable via the settings UI:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_enumerate_depth` | 2 | Decision levels to fully enumerate before sampling. Set to 255 for exhaustive rollouts (pre-sampling behavior). |
| `rollout_opponent_samples` | 8 | Opponent hands sampled per hero combo. More samples = less variance, higher accuracy, slower. |
| `rollout_num_samples` | 3 | Chance-node samples (random runout cards) per rollout evaluation. |

See `docs/architecture.md` (Sampled Rollout Evaluator) for algorithmic details.

---

### gpu-range-solve

GPU-accelerated version of `range-solve` using custom CUDA kernels via the `gpu-range-solver` crate. Same inputs and output format as `range-solve`. Requires an NVIDIA GPU with CUDA 12.1+.

```bash
cargo run -p poker-solver-trainer --release -- gpu-range-solve \
  --oop-range "QQ+,AKs" --ip-range "JJ-99,AQs" \
  --flop "Qs Jh 2c" --turn "8d" --river "3s" \
  --pot 100 --effective-stack 100 --iterations 500
```

Options are identical to `range-solve` except `--compressed` is not supported.

**Architecture:** Hand-parallel CUDA kernel — one thread block per subgame, up to 1024 threads handling 1024 hands in parallel. Tree traversal is sequential within the block; `__syncthreads` is used only for fold terminal evaluation (card-blocking reduction). No cooperative groups required.

**Performance characteristics:**
- CUDA context initialization: ~280ms one-time cost per invocation
- Per-iteration: ~0.6-1.2ms (vs ~0.02-0.08ms CPU) for single river subgames
- GPU advantage is in **throughput**: 142 independent subgames solved simultaneously when batched (one per SM on RTX 6000 Ada)
- Best for: batched datagen (many subgames), not single-spot analysis (use `range-solve` for that)

---

### cluster

Run the potential-aware clustering pipeline to build bucket assignments for all four streets. Uses Pluribus-style bottom-up abstraction: river (equity k-means) → turn (EMD over river buckets) → flop (EMD over turn buckets) → preflop (EMD over flop buckets).

```bash
cargo run -p poker-solver-trainer --release -- cluster \
  -c sample_configurations/blueprint_v2_with_tui.yaml \
  -o output/buckets
```

Options:
- `-c <CONFIG>` -- YAML config file (uses the `clustering` section)
- `-o <DIR>` -- Output directory for `.buckets` files

Produces four files: `river.buckets`, `turn.buckets`, `flop.buckets`, `preflop.buckets`. If bucket files already exist in the output directory, clustering is skipped.

Clustering config parameters (in the `clustering` section of the YAML):

| Parameter | Default | Description |
|-|-|-|
| `algorithm` | `potential_aware_emd` | Clustering algorithm |
| `river.buckets` | -- | Number of river buckets |
| `turn.buckets` | -- | Number of turn buckets |
| `flop.buckets` | -- | Number of flop buckets |
| `preflop.buckets` | -- | Number of preflop buckets |
| `kmeans_iterations` | 100 | K-means iterations per street |
| `seed` | 42 | Random seed for board sampling |
| `<street>.metric.enabled` | `false` | Enable experimental combined ground distance for turn/flop clustering |
| `<street>.metric.potential_weight` | `0.0` | Uniform adjacent-bucket movement weight when the experimental metric is enabled |
| `<street>.metric.equity_weight` | `1.0` | Child centroid equity-gap weight when the experimental metric is enabled |
| `<street>.metric.nut_distance_weight` | `0.0` | Sampled nut-distance-gap weight when the experimental metric is enabled |
| `<street>.metric.nut_distance_transform` | `linear` | Shape applied to the normalized nut-distance channel: `linear`, `sqrt`, or `log1p` |
| `<street>.metric.nut_distance_cap` | unset | Optional cap applied to the normalized nut-distance channel before `nut_distance_transform` |
| `<street>.metric.nut_sample_boards` | `200` | River boards sampled to estimate bucket nut-distance scores for the experimental metric |

The experimental metric is opt-in per clustering street. Defaults preserve the existing potential-aware EMD behavior. When enabled, potential, equity-gap, and nut-distance-gap channels are each normalized by their mean positive adjacent gap before weights are applied, so weights are comparable across bucket builds. The nut-distance channel can also be capped and shaped after normalization; this lets it act as a guardrail for nut hierarchy without letting rare extreme gaps dominate the potential-aware child-bucket distribution. Clustering writes `metric_scales.json` beside the bucket files, and `diag-clusters --scorecard-json` includes it when present. A typical first candidate is to enable it on `turn` and/or `flop` with `equity_weight: 1.0`, a small `nut_distance_weight`, and `potential_weight: 0.0`; use `nut_distance_cap` when the audit shows improved turn hierarchy but worse flop tail spread.

### Heuristic V3 Algorithm

Set `algorithm: heuristic_v3` in the clustering config to use deterministic two-axis bucketing instead of the default `potential_aware_emd` clustering.

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `nut_distance_bits` | 6 | 2-8 | Bits for nut distance quantization (2^N bins) |
| `equity_delta_bits` | 4 | 2-8 | Bits for equity delta quantization (2^N bins) |

Total buckets per street = `2^nut_distance_bits * 2^equity_delta_bits`. Default: 1,024.

Example config: `sample_configurations/heuristic_v3_1024bkt.yaml`

To generate bucket files:
```bash
cargo run -p poker-solver-trainer --release -- cluster -c sample_configurations/heuristic_v3_1024bkt.yaml -o ./local_data/clusters_heuristic_v3
```

### diag-clusters

Diagnostics for pre-computed cluster bucket files.

```bash
# Basic bucket distribution report
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets

# Equity audit (sample boards and check intra-bucket equity consistency)
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --audit --audit-boards 100

# Cross-street transition matrices (verify potential-aware linkage)
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --transitions

# Transition consistency audit (sample boards and compare next-street bucket distributions)
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --transition-audit --transition-audit-boards 50

# Hand-class assignment audit (trace class/strength groups into buckets)
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --hand-class-audit --hand-class-audit-boards 25

# Machine-readable scorecard for regression comparison
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --hand-class-audit --scorecard-json output/bucket_scorecard.json

# Sample hands from a specific bucket
cargo run -p poker-solver-trainer --release -- diag-clusters \
  -d output/buckets --sample-bucket river 5
```

Options:
- `-d <DIR>` -- Directory containing `.buckets` files (required)
- `--audit` -- Run equity audit by sampling boards
- `--audit-boards <N>` -- Number of boards to sample for audit (default: 50)
- `--transitions` -- Print cross-street transition matrices for adjacent street pairs (preflop→flop, flop→turn, turn→river)
- `--transition-audit` -- Sample flop→turn and turn→river boards and report whether combos in each current bucket produce similar next-street bucket distributions. Board and holding lookups are canonicalized together, matching runtime bucket lookup semantics.
- `--transition-audit-boards <N>` -- Number of boards to sample for transition consistency audit (default: 20)
- `--hand-class-audit` -- Sample flop/turn/river boards and trace bucket assignments by private-card contribution (`board`, `1h`, `2h`), made hand class, rank-like intra-class strength, and equity decile. Reports contribution/class/strength groups that scatter across many buckets, buckets that mix incompatible populations, and possible strength-order inversions.
- `--hand-class-audit-boards <N>` -- Number of boards to sample for hand-class audit (default: 10)
- `--hand-class-audit-top <N>` -- Rows to show in each hand-class audit section (default: 10)
- `--scorecard-json <PATH>` -- Write a stable JSON scorecard for regression comparison. Includes bucket-size skew metrics for every loaded street, selected Kxs/Qxs suited-hand profiles, `metric_scales.json` when present, and a sampled river nut-distance audit using `--hand-class-audit-boards`/`--hand-class-audit-top` for sample size/detail. When `--hand-class-audit` is enabled, also includes skipped lookup counts, class/strength spread metrics, mixed-bucket entropy/equity spans, and strength-order inversion summaries.
- `--sample-bucket <STREET> <BUCKET_ID>` -- Show 10 sample hands from the given bucket
- `--centroid-emd <STREET>` -- Placeholder; centroid EMD requires feature vectors not stored in bucket files

### diff-clusters

Compare two sets of bucket files to measure quality improvement and clustering similarity.

```bash
cargo run -p poker-solver-trainer --release -- diff-clusters \
  --dir-a /path/to/old/clusters \
  --dir-b /path/to/new/clusters \
  --sample-boards 200

# Verbose mode with equity histogram
cargo run -p poker-solver-trainer --release -- diff-clusters \
  --dir-a /path/to/old/clusters \
  --dir-b /path/to/new/clusters \
  --sample-boards 200 \
  --verbose
```

- `--dir-a`, `--dir-b` -- directories containing `.buckets` files to compare
- `--sample-boards` -- boards to sample for equity audit (default 200, 0 = skip)
- `--verbose` -- show per-equity-bin bucket histogram

Reports per-street: bucket size stats, intra-bucket equity std (lower = better), and Adjusted Rand Index (1.0 = identical groupings, 0.0 = random agreement).

---

### compare-solve

Compare a blueprint's strategy against an exact subgame re-solve on a specific postflop spot. Loads a blueprint bundle, extracts ranges at the specified node, solves the subgame with DCFR (range-solver), and reports strategy deltas between the blueprint and exact solutions.

```bash
cargo run -p poker-solver-trainer --release -- compare-solve \
  --bundle /path/to/blueprint/bundle \
  --spot 'sb:2bb,bb:10bb,sb:22bb,bb:call|JhTh9h|bb:15bb,sb:call|7d' \
  --iters 200 --river-boundary cfvnet \
  --river-model /path/to/cfvnet_river.onnx
```

Options:
- `--bundle <DIR>` -- Path to blueprint bundle directory (required)
- `--snapshot <NAME>` -- Snapshot name, e.g. `snapshot_0013`; defaults to latest
- `--spot <SPOT>` -- Spot encoding: actions and board cards separated by `|` (required)
- `--iters <N>` -- DCFR iterations for both solves (default: 200)
- `--tolerance <F>` -- Max per-cell strategy delta before non-zero exit (default: 0.0, disabled)
- `--verbose` -- Print per-iteration progress
- `--flop-boundary <MODE>` -- Flop boundary evaluator: `exact` (default), `cfvnet`, `exact_subtree`, or `exact_oracle`
- `--turn-boundary <MODE>` -- Turn boundary evaluator: `exact` (default), `cfvnet`, `exact_subtree`, or `exact_oracle`
- `--river-boundary <MODE>` -- River boundary evaluator: `exact` (default), `cfvnet`, `exact_subtree`, or `exact_oracle`
- `--flop-model <PATH>` -- ONNX model path (required when `--flop-boundary=cfvnet`)
- `--turn-model <PATH>` -- ONNX model path (required when `--turn-boundary=cfvnet`)
- `--river-model <PATH>` -- ONNX model path (required when `--river-boundary=cfvnet`)
- `--flop-model-kind <MODE>` / `--turn-model-kind <MODE>` / `--river-model-kind <MODE>` -- Cfvnet inference contract. `direct` (default) evaluates the supplied boundary board as-is and is required for direct BoundaryNet models trained on that street with normalized EV output (`chip_ev / (pot + effective_stack)`). `river_enumerated_turn` is the legacy adapter: 4-card turn boards enumerate all valid river runouts and average a river model. Use it only when intentionally applying a river model at a turn boundary. `direct_normalized_legacy` evaluates the supplied boundary board as-is but adapts the current Python-exported direct checkpoint output from `bcfv * pot / (pot + effective_stack)` into the evaluator's expected runtime units.
- `--oracle-orientation <MODE>` -- Hidden `exact_oracle` diagnostic. Accepted values are `current`, `swap`, `sign-flip`, and `swap-sign-flip`; use only to audit OOP/IP and sign orientation at the raw boundary-CFV handoff.
- `--oracle-scale <FLOAT>` -- Hidden `exact_oracle` diagnostic. Multiplies raw oracle CFVs before boundary injection; default `1.0`.
- `--exact-iters <N>` -- Hidden diagnostic override for exact solve iterations; defaults to `--iters`.
- `--subgame-iters <N>` -- Hidden diagnostic override for subgame solve iterations; defaults to `--iters`.
- `--dump-boundary-cfvs` -- Hidden boundary diagnostic. Before the subgame solve, forces a seeded boundary evaluation pass, compares root action CFVs and regret-input pressure for the injected candidate boundary against `exact_oracle`, compares each boundary contribution against `exact_oracle` raw CFVs, prints an `exact_subtree` raw-control comparison, and prints aggregate buckets by player/all-in state, pot, SPR, reach density, and oracle magnitude before the cached evaluator CFV stats.
- `--boundary-cfv-max-mean-abs <F>` -- Hidden diagnostic gate for `--dump-boundary-cfvs`. Fails the run if either player's aggregate candidate-vs-oracle boundary CFV mean absolute error exceeds `F`.
- `--boundary-cfv-min-corr <F>` -- Hidden diagnostic gate for `--dump-boundary-cfvs`. Fails the run if either player's aggregate candidate-vs-oracle boundary CFV correlation is below `F`.
- `--oracle-iteration-aligned` -- Hidden `exact_oracle` diagnostic. Runs exact and subgame in lockstep and evaluates each subgame boundary against the exact continuation at the same iteration. Requires matching `--exact-iters` and `--subgame-iters`.
- `--root-update-trace-iters <csv>` -- Hidden diagnostic for `--oracle-iteration-aligned`. Prints exact/subgame root action CFV gaps and root regret-update gaps for the listed zero-based iterations.

Boundary modes:
- `exact` -- Continue solving through the full tree with no depth boundary.
- `cfvnet` -- Use an ONNX CFV network at the selected depth boundary.
- `exact_subtree` -- At each boundary, solve a fresh exact subtree using the live boundary reaches.
- `exact_oracle` -- Diagnostic mode: solve the full exact game first, then evaluate each depth boundary with that solved exact continuation and the subgame's live boundary reaches. This isolates boundary integration from exact-subtree re-solving noise.

#### Gadget Flags (Safe Re-solving)

The following flags enable safe subgame re-solving via a CFR-D gadget. See `docs/architecture.md` (Safe Subgame Solving) for algorithmic details.

- `--gadget` -- **Option A2 (per-boundary CFR-D gadget).** Enables the [Burch 2014 Section 3](https://webdocs.cs.ualberta.ca/~bowling/papers/14aaai-cfrd.pdf) safe re-solving gadget via tree modification: a 4-node gadget subtree (`G_IP -> [Terminate_IP | G_OOP -> [Terminate_OOP | existing cfvnet]]`) is injected at each cfvnet depth-boundary, with per-hand per-boundary opt-out CFVs from blueprint CBVs. Traverser-dependent activation ensures each player's gadget is regret-matched only on their own traverser pass; on the non-owner's pass the gadget is a passthrough. Startup prints `gadget mode: tree`. Mutually exclusive with `--gadget-clamp`.

- `--gadget-clamp` -- **Legacy post-clamp diagnostic.** Preserves access to the previous `GadgetEvaluator` wrapper approach (post-clamp to the opt-out floor at depth boundaries). Retained for A/B comparison while Option A2 rolls out. Mutually exclusive with `--gadget`. Will be retired in a follow-up change.

- `--gadget-provider <PROVIDER>` -- Opt-out value source. `blueprint-cbv` (default) reads from the bundle's `CbvTable`; `constant` uses a fixed value from `--gadget-constant`. Applies to both `--gadget` and `--gadget-clamp` paths.

- `--gadget-constant <F>` -- Constant opt-out value (pot-normalised bcfv) when `--gadget-provider=constant` (default: 0.0). Ignored otherwise.

---

## Training TUI Dashboard

When `tui.enabled: true` in the config, `train-blueprint` launches a full-screen terminal dashboard instead of text output.

**Parallel Training:** Blueprint V2 automatically uses all available CPU cores. Each batch of `batch_size` deals (default: 200) is processed in parallel using Rayon's thread pool. LCFR discount and snapshots run between batches. Set `RAYON_NUM_THREADS=N` to limit core usage.

**Strategy Delta Stopping:** Set `target_strategy_delta` in the training config to auto-stop when the average strategy stabilises. The delta is the mean max absolute probability change across all (node, bucket) information sets between metric checks. Checked every `print_every_minutes`. Example: `target_strategy_delta: 0.001` stops when the strategy is changing by less than 0.1% on average.

**Resume Training:** Set `resume: true` under `snapshots:` to continue from the latest valid checkpoint in `output_dir`. The trainer considers numbered `snapshot_NNNN/` directories and `final/` when they contain `regrets.bin` plus readable `metadata.json` with `iteration` and `elapsed_minutes`; metadata-missing checkpoints are skipped. Candidates are ordered by metadata `iteration`, then metadata `elapsed_minutes`, then `final/` status, then numbered snapshot index. A stale `final/` directory no longer overrides a newer numbered snapshot, but `final/` wins when its metadata is equal to or newer than the best numbered checkpoint.

**Snapshot Retention:** Set `max_snapshots: N` under `snapshots:` to keep only the N most recent snapshots. After each save, older `snapshot_NNNN/` directories are deleted. The `final/` directory is never pruned. Omit or set to `null` for unlimited retention.

**Left panel:** iteration progress, throughput sparkline, exploitability chart
**Right panel:** tabbed 13x13 strategy grids for configured scenarios

**Hotkeys:**
- `p` -- pause/resume training
- `s` -- trigger immediate snapshot; MP TUI reports queued, writing, saved, or failed status on the hotkey line
- `e` -- trigger exploitability calculation
- left/right arrows -- switch scenario tabs
- `q` -- quit gracefully

**Convergence indicators:** Cells where strategy has stabilized (delta < 0.01) show a bright green border. As training progresses, more cells "light up" -- giving visual feedback on convergence.

Use `--no-tui` to disable the dashboard and use text output instead.

See `sample_configurations/blueprint_v2_with_tui.yaml` for a complete example.

### N-Player TUI (`train-blueprint-mp`)

The multiplayer trainer supports a TUI dashboard. Enable it in the YAML config:

```yaml
tui:
  enabled: true
  scenarios:
    - name: "UTG open"
      spot: ""
```

Alternatively, omit `--no-tui` from the command line (the TUI is enabled by default when `tui.enabled: true`). With `training.backend: lazy_sparse`, the dashboard reports live global telemetry, sampled strategy movement, and configured hand-grid scenarios backed by sparse strategy rows.

#### Position Labels

Seat positions are assigned standard poker labels based on the number of players:

| Players | Seat 0 | Seat 1 | Seat 2 | Seat 3 | Seat 4 | Seat 5 | Seat 6 | Seat 7 |
|---------|--------|--------|--------|--------|--------|--------|--------|--------|
| 2       | SB     | BB     |        |        |        |        |        |        |
| 3       | BTN    | SB     | BB     |        |        |        |        |        |
| 4       | CO     | BTN    | SB     | BB     |        |        |        |        |
| 5       | HJ     | CO     | BTN    | SB     | BB     |        |        |        |
| 6       | UTG    | HJ     | CO     | BTN    | SB     | BB     |        |        |
| 7       | UTG    | UTG1   | HJ     | CO     | BTN    | SB     | BB     |        |
| 8       | UTG    | UTG1   | UTG2   | HJ     | CO     | BTN    | SB     | BB     |

Action order is UTG-first (seat after BB), proceeding clockwise.

#### Spot Encoding

Each scenario's `spot` field encodes the action sequence leading to the decision point. The format is a comma-separated list of `position:action` pairs in action order:

- Empty string `""` -- first to act (UTG in 6-max), no preceding actions
- `"utg:fold"` -- UTG folded, now HJ's turn
- `"utg:5bb"` -- UTG raised to 5bb (in chips), now HJ faces a raise
- `"utg:fold,hj:fold,co:fold,btn:fold,sb:call"` -- everyone folds to SB who limps, now BB acts

Actions use the same size labels as `action_abstraction.preflop.lead` (e.g., `5bb`, `6bb`) or the keyword `fold`/`call`.

#### Example Scenarios

**Opening ranges** (who raises first when it folds to them):

```yaml
scenarios:
  - name: "UTG open"
    spot: ""
  - name: "HJ open"
    spot: "utg:fold"
  - name: "CO open"
    spot: "utg:fold,hj:fold"
  - name: "BTN open"
    spot: "utg:fold,hj:fold,co:fold"
  - name: "SB open"
    spot: "utg:fold,hj:fold,co:fold,btn:fold"
  - name: "BB vs limp"
    spot: "utg:fold,hj:fold,co:fold,btn:fold,sb:call"
```

**3-bet ranges** (who re-raises a UTG open):

```yaml
scenarios:
  - name: "HJ vs UTG"
    spot: "utg:5bb"
  - name: "CO vs UTG"
    spot: "utg:5bb,hj:fold"
  - name: "BTN vs CO"
    spot: "utg:fold,hj:fold,co:5bb"
```

#### Pagination

The TUI displays up to 6 strategy grids per page. When more than 6 scenarios are configured, they are split across multiple pages. Use the left/right arrow keys to navigate between pages.

See `sample_configurations/blueprint_mp_6player_ante.yaml` for a full 12-scenario config spanning 2 pages.

## Blueprint Training Configuration

All `game:` section values are in **chips** (1 BB = 2 chips). Example: `stack_depth: 200` = 100 BB, `small_blind: 1`, `big_blind: 2`. `allow_preflop_limp` defaults to `true`; set it to `false` to remove unopened cold limps while keeping folds, configured open sizes, SB completion, and BB checks. Preflop action sizes use chip amounts with a `bb` suffix: `"5bb"` = raise to 5 chips (2.5 BB). Display converts to BB at the UI/CLI boundary only (dividing by 2). See `docs/architecture.md` for full unit convention.

The `training:` section of the blueprint YAML config controls the MCCFR training loop. Key parameters:

### Optimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `optimizer` | `"dcfr"` | CFR variant: `"dcfr"`, `"sapcfr+"`, `"brcfr+"`, `"lcfr"`, `"cfr+"` |
| `storage_backend` | `"dense"` | HU blueprint_v2 CFR storage backend: `"dense"` or opt-in `"sparse"`/`"lazy"` |
| `dcfr_alpha` | `1.5` | Positive regret discount exponent. Higher = retain positive regrets longer |
| `dcfr_beta` | `0.0` | Negative regret discount exponent. Used by DCFR only (SAPCFR+ floors to 0) |
| `dcfr_gamma` | `2.0` | Strategy sum discount exponent. Higher = weight recent strategies more |
| `dcfr_epoch_cap` | `null` | Optional cap on discount epoch counter. Prevents discount from converging to 1.0 |
| `sapcfr_eta` | `0.5` | SAPCFR+ prediction step size. 0 = no prediction (DCFR+RM+), 1 = full PCFR+, 0.5 = dampened |
| `brcfr_eta` | `0.6` | BRCFR+ BR prediction weight. Scales the best-response signal in strategy computation |
| `brcfr_warmup_iterations` | `0` | Iterations of pure DCFR+ before the first BR prediction pass |
| `brcfr_interval` | `100000000` | Iterations between BR prediction passes (after warmup) |

**DCFR** (default): Discounted CFR with polynomial decay. Positive regrets multiplied by `t^α/(t^α+1)`, negative by `t^β/(t^β+1)`, strategy sums by `(t/(t+1))^γ`. Standard choice from Brown & Sandholm 2019.

**SAPCFR+**: Simplified Asymmetric Predictive CFR+. Combines DCFR discount with RM+ (negative regret flooring) and predictive strategy computation. Stores previous iteration's instantaneous regret as a prediction, then computes strategy from `R + eta * prediction` instead of raw cumulative regrets. Based on Xu et al. 2025. Requires extra ~1.1 GB for prediction buffer. Since negative regrets are floored to 0, `dcfr_beta` is ignored and `prune_threshold` should be 0 or negative.

**LCFR**: Linear CFR (α=β=γ=1). Used by Pluribus. Simplest discounting scheme.

**CFR+**: Regret matching+ with negative regret flooring. No discounting.

**BRCFR+**: Best-Response augmented DCFR+. Layers periodic best-response prediction passes on top of the standard DCFR+ optimizer. During the warmup phase (`brcfr_warmup_iterations`), behaves identically to DCFR+. After warmup, a full BR traversal runs every `brcfr_interval` iterations for both players. The BR-derived per-infoset regrets are stored in the prediction buffer and used in strategy computation as `R_tilde = max(0, R + eta * decay * v_br)`. The decay factor starts at 1.0 after each BR pass and decreases linearly to 0.0 over the refresh interval, so stale predictions fade naturally. When decay reaches 0, behavior is pure DCFR+. Exploitability is measured for free during each BR pass (no separate exploitability calculation needed). Requires the same prediction buffer as SAPCFR+ (~1.1 GB extra). Based on ideas from CFR-BR (Johanson 2012) with decay scheduling.

### HU Storage Backend

`train-blueprint` defaults to eager dense storage. Dense storage allocates every `(decision node, bucket, action)` regret and strategy-sum slot before training starts and is still the safest default for existing production configs.

Set `training.storage_backend: "sparse"` to use the HU sparse row backend:

```yaml
training:
  storage_backend: "sparse"
  optimizer: "sapcfr+"
  use_baselines: true
  regret_floor: 0
```

Sparse storage keeps the current eager `blueprint_v2` game tree, but CFR rows are allocated only when traversal writes to a `(decision node, bucket)` pair. Missing rows behave exactly like all-zero dense rows: zero regrets, strategy sums, predictions, and baselines, with uniform current and average strategy. Sparse training uses the same SAPCFR+ prediction, baseline, and regret-floor settings as dense storage.

`brcfr+` is dense-only for HU `blueprint_v2` in this slice. Configs that combine `storage_backend: "sparse"` with `optimizer: "brcfr+"` fail fast with an explicit error instead of silently changing semantics.

Sparse internals are not exposed to Explorer/Tauri bundle consumers. Snapshots still write dense-compatible `strategy.bin`, `regrets.bin`, metadata, CBVs, and hand-EV files. Resume also remains dense-compatible: a sparse run can resume from a dense snapshot by loading `regrets.bin` and realizing only non-zero projected rows. There is no sparse HU on-disk snapshot default.

In no-TUI progress output, sparse training adds a storage line with realized rows/slots, inserts, read/write probe and hit counters, dense-equivalent slots/bytes, and approximate sparse resident bytes.

### Example: BRCFR+ Configuration

```yaml
training:
  cluster_path: "./local_data/buckets/200_v1"
  time_limit_minutes: 7200
  optimizer: "brcfr+"
  brcfr_eta: 0.6
  brcfr_warmup_iterations: 300000000
  brcfr_interval: 100000000
  dcfr_alpha: 1.5
  dcfr_gamma: 2.0
  dcfr_epoch_cap: 40
  batch_size: 4000
```

### Variance Reduction

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_baselines` | `false` | Enable VR-MCCFR variance-reducing baselines (Schmid et al. 2019) |
| `baseline_alpha` | `0.01` | Baseline EMA learning rate. Lower = smoother estimates, slower adaptation |

When enabled, the opponent traversal uses learned baselines to reduce sampling variance by up to 1000×. Each (node, bucket, action) gets an exponential moving average of observed counterfactual values. The baseline-corrected formula is unbiased and degenerates to standard sampling when baselines are zero. Requires extra ~1.1 GB for the baseline buffer (same size as regret buffer).

### External Baseline Strategy-Frequency Validation

`training.baseline_validation` enables periodic comparison of the learned average strategy against a fixed external preflop baseline. This is a convergence diagnostic only. It compares action frequencies with total-variation distance; it is not an EV pass/fail check and does not invoke the range solver.

The current baseline integration is pinned to `local_data/baselines/cash_hu_20bb_cev.json` and requires the target config to match:

- `game.stack_depth: 40` (20bb in repo chip units)
- `game.small_blind: 1`, `game.big_blind: 2`
- `game.allow_preflop_limp: false`
- `clustering.preflop.buckets: 169`
- `action_abstraction.preflop` rows `["2.5bb"]` then `["5bb"]`

The reproducible sample uses the existing `local_data/buckets/500f_500t_500r_v2` postflop bucket set (`500/500/500`) via `training.cluster_path`; the baseline comparison itself remains preflop-only.

Example:

```yaml
training:
  cluster_path: "./local_data/buckets/500f_500t_500r_v2"
  baseline_validation:
    enabled: true
    baseline_path: "local_data/baselines/cash_hu_20bb_cev.json"
    interval_iterations: 1000
    interval_minutes: 0
    top_n_spots: 5
    top_n_combos_per_spot: 5
```

Run the reproducible sample with:

```bash
cargo run -p poker-solver-trainer --release -- train-blueprint \
  --config sample_configurations/blueprint_v2_hu_20bb_baseline_validation.yaml \
  --no-tui
```

No-TUI logs and the TUI diagnostics panel report aggregate TV, root TV, first-response TV, worst-spot TV, coverage, skipped zero-mass rows, invalid rows, unsupported spots/actions, and the top worst spots/combo rows. Validation is cadence-bound by `interval_iterations` and/or `interval_minutes`; it does not run per traversal. Sparse/lazy storage is supported through `active_storage().average_strategy()` without dense projection.

If the stack, blinds, limp policy, preflop buckets, tree actions, or baseline schema do not match the pinned 20bb cEV target, the trainer rejects the validation path before scoring rows.

### Schedule & Pruning

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chance_continuation_mode` | `sampled_full_deal` | Lazy MP chance-continuation mode. `sampled_full_deal` samples a complete board up front; `sampled_turn_exact_river` samples through the turn and averages values over all legal rivers at river chance/showdown boundaries; `sampled_flop_exact_turn_river` samples through the flop and averages over all legal turn/river continuations |
| `lcfr_warmup_iterations` | `0` | Iterations before discounting starts |
| `lcfr_discount_interval` | `1` | Iterations between discount applications |
| `prune_after_iterations` | `0` | Warmup boundary before opt-in MP traversal pruning and negative-action subtree purge can start |
| `traversal_pruning_enabled` | `false` | Opt in to ordinary MP regret-threshold traversal pruning. This skips eligible traverser-side action branches, but does not physically delete sparse rows or strategy sums |
| `prune_threshold` | `-300` | Cumulative regret threshold for ordinary traversal pruning when `traversal_pruning_enabled` is true |
| `prune_explore_pct` | `0.05` | Fraction of post-warmup batches that disable ordinary traversal pruning and explore all actions |
| `negative_action_subtree_purge_enabled` | `false` | Opt in to the parsed/configured negative-action subtree purge experiment |
| `negative_action_prune_below` | `-1` | Negative-action purge candidate threshold for cumulative regret |
| `negative_action_reactivate_at` | `0` | Cumulative regret value at or above which a purged action can reactivate |
| `negative_action_purge_mode` | `scan_history_prefix` | Purge candidate detection mode. Currently only `scan_history_prefix` is supported |
| `batch_size` | `200` | Deals per parallel batch |
| `time_limit_minutes` | `0` | Stop after this many minutes (0 = unlimited) |
| `purify_threshold` | `0.0` | Purify strategies with probability below this threshold (0 = disabled) |

**Important for SAPCFR+**: Since RM+ floors negative regrets to 0, they can't accumulate below the prune threshold. Set `prune_threshold: 0` to effectively disable pruning, or use a small negative value as a safety margin.

**Negative-action subtree purge experiment**: The training parser accepts `negative_action_subtree_purge_enabled`, `negative_action_prune_below`, `negative_action_reactivate_at`, and `negative_action_purge_mode` under `training:`. These keys are configured for the negative-action subtree purge experiment. Purge/block behavior is inactive until `prune_after_iterations`; before that warmup boundary, negative regrets do not drop nodes. The current sample experiment config is `sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml`.

```yaml
training:
  backend: lazy_sparse
  prune_after_iterations: 600000
  prune_explore_pct: 0.0
  negative_action_subtree_purge_enabled: true
  negative_action_prune_below: -1
  negative_action_reactivate_at: 0
  negative_action_purge_mode: scan_history_prefix
```

With `negative_action_subtree_purge_enabled: true`, lazy traversal starts checking aggressive action-edge cumulative regret only once `meta_iter >= prune_after_iterations`. Passive actions (`Fold`, `Check`, `Call`, and all-in calls that do not increase the current max bet) are never persistent subtree-purge candidates, because in multiplayer trees they can contain later players' decision nodes. Ordinary MP traversal pruning is separate: with `traversal_pruning_enabled: true`, a traverser-side nonterminal branch whose current strategy probability is zero can be skipped when its regret is below `prune_threshold`; this skip does not physically delete sparse rows or strategy sums. If an aggressive action regret is below `negative_action_prune_below`, the negative-action purge path marks the edge blocked and retains its child action-history prefix for a later boundary sweep. Traversal skips blocked aggressive edges instead of allocating more rows below them, but ordinary traversal does not physically delete descendant sparse rows. Immediately after each lazy DCFR discount, storage scans the currently blocked edge set and rereads each parent action regret after discounting. A blocked edge whose regret reaches `negative_action_reactivate_at` is unblocked without purging its child subtree. Edges that remain blocked are batched into one sparse-storage scan; the child row and all already visited descendants below any stored prefix are purged while sibling histories are preserved. Future visits below an edge whose subtree was purged allocate fresh sparse rows, so descendants resume with first-visit/default behavior: zero cumulative regrets and uniform strategy until new updates arrive.

The current 6-max experiment is `sample_configurations/blueprint_mp_6max_500f_100t_100r.yaml`; run it with the `train-blueprint-mp` command shown above in the Blueprint MP section.

**Regret overflow**: Regrets are stored as `i32` (×1000 scaling, max ~2.1M). If `lcfr_discount_interval` is too large, regrets overflow and the trainer panics with a clear message. For SAPCFR+ (which only accumulates positive regrets), keep the discount interval reasonable (e.g., 1M-10M).

### Example: SAPCFR+ with Baselines

```yaml
training:
  cluster_path: "./local_data/buckets/1k_v3"
  time_limit_minutes: 7200
  optimizer: "sapcfr+"
  sapcfr_eta: 0.5
  dcfr_alpha: 1.5
  dcfr_gamma: 2.0
  dcfr_epoch_cap: 80
  lcfr_warmup_iterations: 10000000
  lcfr_discount_interval: 10000000
  prune_after_iterations: 10000000
  prune_threshold: 0
  batch_size: 2000
  use_baselines: true
  baseline_alpha: 0.01
```

See `sample_configurations/blueprint_v2_1kbkt_sapcfr.yaml` for the full config.

---

## CFVnet Training Pipeline

The `cfvnet` crate trains Deep Counterfactual Value Networks following the Supremus/DeepStack approach: solve random subgames, extract per-combo counterfactual values, and train a neural network to predict them. Networks are trained bottom-up: river first, then turn-boundary, then flop-boundary.

BoundaryNet uses one canonical model IO contract across training and inference: OOP range (1326) + IP range (1326) + board one-hot (52) + rank presence (13) + `pot/(pot+stack)` + `stack/(pot+stack)` + player. Ranges are in canonical combo order, public-board-blocked combos are zeroed, and remaining finite non-negative mass is normalized to sum 1 after blockers. The model output is `chip_cfv / (pot + effective_stack)`. Some dataset records store CFVs as `chip_cfv / pot`; loaders convert that storage unit into the normalized BoundaryNet target. Range-solver half-pot BCFV units are a legacy runtime adapter only.

### River Network

#### Generate River Training Data

**CPU backend (default):**

```bash
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/river_cfvnet.yaml \
  --output data/river_training.bin \
  --num-samples 1000000 \
  --threads 8
```

**GPU backend (NVIDIA GPU required):**

```bash
cargo run -p cfvnet --release --features gpu-datagen -- generate \
  --config sample_configurations/river_cfvnet.yaml \
  --output data/river_training.bin \
  --num-samples 1000000
```

Set `datagen.backend: "gpu"` in the YAML config to use GPU solving. The GPU backend solves batches of river subgames simultaneously using the hand-parallel CUDA kernel. Each batch launches up to `gpu_batch_size` (default: 142) subgames in a single kernel launch.

```yaml
datagen:
  street: "river"
  backend: "gpu"          # "cpu" (default) or "gpu"
  gpu_batch_size: 142     # subgames per GPU launch (default: 142)
  num_samples: 1000000
  solver_iterations: 500
```

CFVnet datagen accepts the same bet-size token style as the range solver for pot-relative sizes and all-in. In nested form, rows are keyed by the number of bets already made on the street: `game.bet_sizes[0]` is used for the first bet, `game.bet_sizes[1]` for the second bet / first raise, and so on. Raise depths past the configured rows are forced to all-in only. Example:

```yaml
game:
  bet_sizes:
    - ["25%", "50%", "100%", "a"]  # first bet
    - ["25%", "75%", "a"]          # second bet / first raise
    - ["a"]                        # third bet is all-in only
```

The `a` token is preserved as an explicit all-in action in the generated range-solver tree. `datagen.bet_size_fuzz` perturbs only pot-relative sizes; all-in remains exact and unfuzzed.

To generate river data from actual reached blueprint spots, use
`sample_configurations/river_cfvnet_sampled_spots.yaml`. Set
`datagen.sampled_river_spots: true` and point `datagen.blueprint_bundle_path`
at a full blueprint bundle. This samples a concrete river board, walks the
blueprint strategy through preflop/flop/turn, and uses the reached river pot,
effective stack, and line-conditioned OOP/IP ranges as the river subgame input.
Those ranges are board-blocked and normalized before writing records.

```bash
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/river_cfvnet_sampled_spots.yaml \
  --output local_data/cfvnet/river_sampled_spots_v1 \
  --num-samples 1000000 \
  --threads 16 \
  --per-file 10000
```

To verify that more DCFR iterations improve the exact same sampled spots,
run the fixed-spot convergence diagnostic. It samples the river spots once,
then resolves those same spots at each iteration cap. By default it disables
`datagen.target_exploitability`, so the table measures the iteration ceiling
rather than early stopping:

```bash
cargo run -p cfvnet --release -- sampled-river-convergence \
  --config sample_configurations/river_cfvnet_sampled_spots.yaml \
  --num-spots 5 \
  --iterations 50,100,250,500,1000,2500,5000
```

Add `--respect-target` to test production early-stop behavior. If
`avg_iter` stays well below the iteration cap, increasing
`datagen.solver_iterations` will not improve the output for those spots unless
`datagen.target_exploitability` is lowered or disabled.

**GPU requirements:**
- NVIDIA GPU with CUDA 12.1+ and compute capability ≥ 6.0
- Build with `--features gpu-datagen` to enable the GPU dependency
- Ranges must produce ≤ 1024 hands per player (games exceeding this fall back to CPU)
- True batching (142 games per launch) requires matching hand counts across games — use blueprint-derived ranges (`blueprint_path`) for best GPU utilization; random RSP ranges may have varying counts

#### Train the River Network

```bash
cargo run -p cfvnet --release -- train \
  --config sample_configurations/river_cfvnet.yaml \
  --data data/river_training.bin \
  --output models/river_v1
```

#### Training Configuration

The `training` section of the YAML config controls the network architecture and training loop. Key parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_layers` | 7 | Number of hidden layers |
| `hidden_size` | 500 | Width of each hidden layer |
| `batch_size` | 2048 | Mini-batch size |
| `epochs` | 2 | Number of training epochs |
| `learning_rate` | 0.001 | Initial learning rate (cosine annealed) |
| `lr_min` | 0.00001 | Minimum learning rate at end of schedule |
| `huber_delta` | 1.0 | Huber loss delta threshold |
| `aux_loss_weight` | 1.0 | Weight for auxiliary game-value loss |
| `validation_split` | 0.05 | Fraction of data reserved for validation |
| `checkpoint_every_n_epochs` | 1000 | Save checkpoint every N epochs (0 = disabled) |
| `shuffle_buffer_size` | 262144 | Streaming shuffle buffer capacity (records). Larger = better shuffle quality, more RAM |
| `prefetch_depth` | 4 | Number of pre-encoded batches buffered in the channel ahead of the training loop |

The training loop uses a **streaming dataloader** with an eviction-based shuffle buffer. A background reader fills a buffer of `shuffle_buffer_size` records, then continuously reads one record at a time — each new record randomly replaces a buffer slot and the evicted record flows into the next batch. This keeps disk reads continuous and eliminates pipeline stalls. A second encoder thread encodes batches in parallel via rayon and sends them through a bounded channel with `prefetch_depth` slots. Every record is seen exactly once per epoch. Increase `shuffle_buffer_size` for better randomization; increase `prefetch_depth` to keep the GPU fed when encoding is slow.

#### Evaluate on Held-Out Data

```bash
cargo run -p cfvnet --release -- evaluate \
  --model models/river_v1 \
  --data data/river_validation.bin
```

#### Compare Against Exact Solves

```bash
cargo run -p cfvnet --release -- compare \
  --model models/river_v1 \
  --num-spots 100
```

### Turn Network

Turn training requires a trained river network. The turn datagen solves random 4-card board situations using DCFR with the river CFV network as leaf evaluator (instead of solving all 46 river runouts exactly).

#### Generate Turn Training Data

Set `datagen.street: "turn"` and `game.river_model_path` in the config:

```yaml
game:
  initial_stack: 200
  bet_sizes: ["25%", "50%", "100%", "a"]
  river_model_path: "models/river_v1/model"
datagen:
  street: "turn"
  num_samples: 1000000
  solver_iterations: 1000
```

```bash
cargo run -p cfvnet --release -- generate \
  --config sample_configurations/turn_cfvnet.yaml \
  --output data/turn_training.bin \
  --num-samples 1000000
```

#### Train the Turn Network

```bash
cargo run -p cfvnet --release -- train \
  --config sample_configurations/turn_cfvnet.yaml \
  --data data/turn_training.bin \
  --output models/turn_v1
```

#### Compare Turn Model Against River Net Evaluator

Validates the turn model by comparing its predictions against fresh PostFlopGame solves using the river network as leaf evaluator:

```bash
cargo run -p cfvnet --release -- compare-net \
  --model models/turn_v1 \
  --river-model models/river_v1 \
  --num-spots 100
```

#### Compare Turn Model Against Exact River Solves

Validates the turn model against PostFlopGame with exact river evaluation (solves all 46 runouts). Slow but provides ground-truth comparison:

```bash
cargo run -p cfvnet --release -- compare-exact \
  --model models/turn_v1 \
  --num-spots 20
```

### BoundaryNet (Direct Boundary-CFV Model)

BoundaryNet is a sibling model to CfvNet that outputs **solver-native boundary CFVs** directly for new direct models. These are bcfv units: `0.0` is the break-even half-pot baseline, `+1.0` is one half-pot above break-even, and `-1.0` is one half-pot below. It uses BoundaryNet input encoding with pot/stack as fractions of total stake, but the output target is the exact value convention consumed by the range-solver boundary evaluator.

The current local direct checkpoint was exported from the Python BoundaryNet trainer, which stores the target as `bcfv * pot / (pot + effective_stack)`. Use `direct_normalized_legacy` for that checkpoint; it applies `bcfv = output * (pot + effective_stack) / pot` at inference.

Python model exports write `model_artifact.yaml` with `model.output_unit` and `model.recommended_model_kind`. `compare-solve` validates that artifact when it is present and rejects incompatible model-kind choices, for example using `direct` on a Python checkpoint that declares `bcfv_scaled_by_pot_over_total_stake`.

BoundaryNet is designed as a depth-boundary evaluator for the range-solver, enabling turn solving with neural network leaf values at river boundaries.

At runtime, BoundaryNet/CFVNet values are treated as conditional boundary values. The model input ranges are normalized to sum to 1 after blockers, and the range-solver applies the live, blocker-aware opponent reach when it consumes the evaluator output. The raw-CFV evaluator path is reserved for exact/oracle evaluators that already return opponent-reach-integrated chip CFVs.

#### Train a BoundaryNet

Use direct turn-boundary data generated with the turn-boundary pipeline. The data must store bcfv targets, not pot-relative EV/share values:

```bash
cargo run -p cfvnet --release -- train-boundary \
  --config sample_configurations/river_cfvnet.yaml \
  --data data/river_training.bin \
  --output models/boundary_v1
```

#### Evaluate BoundaryNet

Reports bcfv MAE with per-SPR bucket breakdown (<1, 1-3, 3-10, 10+):

```bash
cargo run -p cfvnet --release -- eval-boundary \
  --model models/boundary_v1 \
  --data data/river_validation.bin
```

#### Compare BoundaryNet Against Ground Truth

Compares model predictions against datagen ground truth, reporting per-SPR MAE and worst-case error:

```bash
cargo run -p cfvnet --release -- compare-boundary \
  --model models/boundary_v1 \
  --data data/river_validation.bin \
  --num-positions 100
```

#### Using BoundaryNet in the Explorer

To enable neural boundary evaluation for turn solving in the explorer, configure the model path in the Tauri app's postflop settings. When set, turn subgame solving uses BoundaryNet at river boundaries instead of full-depth or rollout evaluation.

### Direct Boundary Datasets

Direct boundary datasets use manifest-backed binary shards with the same `TrainingRecord` layout. The street determines the board length:

- `turn_boundary`: 4-card boards, targets produced from a river model or exact river solves.
- `flop_boundary`: 3-card boards, targets produced by solving flop games to 4-card turn boundary leaves and evaluating those leaves with the direct turn-boundary model.

Generate a flop-boundary pilot dataset:

```bash
cargo run -p cfvnet --release --features onnx -- generate \
  -c sample_configurations/flop_boundary_oracle_datagen.yaml \
  -o local_data/cfvnet/flop_boundary/turn_net_v1/data.bin \
  --num-samples 1000 \
  --per-file 1000
```

For `datagen.street: "flop_boundary"`, `game.board_size` must be `3` and `datagen.turn_boundary_target_source` must be `"turn_net"`. The existing `game.river_model_path` field currently carries the direct turn-boundary ONNX path for this mode.

### Inspect Training Data Distribution

Print frequency histograms (stack size and pot size, 20 equal-width buckets) for generated training data:

```bash
cargo run -p cfvnet --release -- datagen-eval \
  --data data/river_training.bin

# Also works with a directory of split files
cargo run -p cfvnet --release -- datagen-eval \
  --data data/river_chunks/
```

### Compare Output

All compare commands (`compare`, `compare-net`, `compare-exact`) print:
- Summary statistics (mean/worst MAE and mBB)
- Best and worst spots by mBB
- mBB error histograms by stack size and pot size (20 equal-width buckets)
- Frequency histograms by stack size and pot size

### Configuration

See `sample_configurations/river_cfvnet.yaml` for all options. Key parameters:

| Parameter | Default | Description |
|-|-|-|
| `datagen.street` | `"river"` | Street to generate data for (`"river"`, `"turn"`, `"turn_boundary"`, or `"flop_boundary"`) |
| `datagen.backend` | `"cpu"` | Solver backend: `"cpu"` or `"gpu"` (GPU requires `--features gpu-datagen`) |
| `datagen.gpu_batch_size` | 142 | Subgames per GPU kernel launch (only with `backend: "gpu"`) |
| `datagen.num_samples` | 1,000,000 | Training situations to generate |
| `datagen.sampled_river_spots` | `false` | For `street: "river"`, walk a blueprint bundle to reached river decisions instead of using random/RSP or precomputed preflop-only ranges |
| `datagen.blueprint_bundle_path` | none | Full blueprint bundle directory used when `sampled_river_spots` is enabled |
| `datagen.solver_iterations` | 1000 | DCFR iterations per situation |
| `datagen.target_exploitability` | none | Optional early-stop threshold as a pot fraction: stop when exploitability in chips is `<= target_exploitability * pot`; at 100bb/200-chip stacks, `0.01` means roughly `5 * pot` mbb/h |
| `game.river_model_path` | none | Path to trained river model for turn generation, or direct turn-boundary ONNX for `flop_boundary` |
| `training.hidden_layers` | 7 | MLP depth |
| `training.hidden_size` | 500 | Hidden layer width |
| `training.batch_size` | 2048 | Training batch size |
| `training.epochs` | 2 | Training epochs |

---

## Convergence Testing

Test CFR algorithm convergence against an exact baseline using the `convergence-harness` crate. Defines a small tractable game ("Flop Poker") via YAML config, solves it exactly with range-solver DCFR, then compares MCCFR with bucketing against that baseline.

### Game Config

Define the game in a YAML file (see `sample_configurations/convergence_test.yaml`):

```yaml
game:
  flops:
    - "QhJdTh"   # draw-heavy, connected
    - "Ks7d2c"   # dry, rainbow
    - "8c8d3h"   # paired
  starting_pot: 2
  effective_stack: 20
  bet_sizes: "50%,100%,a"
  raise_sizes: "50%,100%,a"

baseline:
  max_iterations: 1000
  target_exploitability: 0.001

mccfr:
  iterations: 1000000
  buckets:
    preflop: 169
    flop: 169
    turn: 200
    river: 200
  checkpoints: [1000, 10000, 100000, 500000, 1000000]
```

### Generate Exact Baseline

Solves each flop exactly with range-solver DCFR (no abstraction). One-time cost — may take minutes to hours depending on game size.

```bash
cargo run -p convergence-harness --release -- generate-baseline \
  --config sample_configurations/convergence_test.yaml \
  --output-dir baselines/convergence
```

Produces: `summary.json`, `convergence.csv`, `strategy.bin`, `combo_ev.bin` in the output directory. Also prints a colored 13x13 SB strategy matrix on exit.

### Run MCCFR Comparison

Runs MCCFR with potential-aware bucketing on the same game, clusters each flop at startup (~2-10s per flop), then compares against the baseline via head-to-head EV (mbb/hand).

```bash
cargo run -p convergence-harness --release -- run-solver \
  --config sample_configurations/convergence_test.yaml \
  --baseline-dir baselines/convergence \
  --output-dir results/mccfr_run
```

At each checkpoint, prints: `h2h mbb/hand = -X.XX (OOP -X.XX, IP -X.XX)`. Negative = MCCFR loses to the exact solution. Final summary:

```
=== Result ===
solver:     MCCFR (200t/200r buckets)
iterations: 1000000
time:       4.4s
mbb/hand:   -230.90
output:     results/mccfr_run
```

### Compare Saved Results

Re-compare any two saved solver results without re-solving:

```bash
cargo run -p convergence-harness --release -- compare \
  --baseline-dir baselines/convergence \
  --result-dir results/mccfr_run
```

### Key Metrics

| Metric | Meaning |
|-|-|
| **mbb/hand** | Milli-big-blinds per hand lost vs exact strategy (negative = losing). 1000 mbb = 1 bb. |
| **L1 distance** | Average total variation distance between strategies per info set. < 0.05 = excellent. |
| **Combo EV diff** | Per-hand EV difference at each decision node. |

### Bucket Sweep

Test different bucket counts to find the optimal abstraction granularity:

```bash
for bkt in 10 50 100 200 500; do
  cargo run -p convergence-harness --release -- run-solver \
    --config sample_configurations/convergence_test.yaml \
    --baseline-dir baselines/convergence \
    --output-dir /tmp/sweep_${bkt} \
    2>&1 | grep "mbb/hand"
done
```

---

## Cloud Training (AWS)

See [`docs/cloud.md`](cloud.md) for running training jobs on AWS EC2 instances via the `solver-cloud` CLI.
