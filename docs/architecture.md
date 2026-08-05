# WarpGTO — Solver Architecture

## Overview

The solver finds a Nash equilibrium for heads-up no-limit Texas Hold'em using **blueprint_v2**, an MCCFR-based solver that operates over the full game tree with hand abstraction (HandClassV2 or EHS2 buckets). The solver samples random deals, traverses preflop through river, and accumulates regrets and strategy sums at each information set. A shared DCFR module (`cfr/dcfr.rs`) handles iteration weighting and regret discounting.

For single-spot analysis, a separate **range solver** provides exact (no-abstraction) DCFR solving with full 1326-combo granularity.

For depth-limited solving, the **cfvnet** pipeline trains neural networks to approximate river counterfactual values, enabling efficient subgame re-solving without full river traversal.

```
GameConfig → HunlPostflop (Game trait) → MccfrSolver → BlueprintV2Strategy
                    ↓
              random deals (full_board)
              HandClassV2 / EHS2 abstraction
```

## Project Structure

```
poker_solver_rust/
├── crates/
│   ├── core/                  # Core solver library
│   │   └── src/
│   │       ├── game/          # Action/Player types, HunlPostflop game
│   │       ├── cfr/           # CFR utilities (regret matching, shared DCFR/LCFR logic)
│   │       ├── blueprint_v2/  # MCCFR blueprint trainer, strategy storage, config
│   │       ├── blueprint_mp/  # N-player (2-10) MCCFR blueprint trainer
│   │       ├── abstraction/   # Card abstraction (isomorphism, EHS2, HandClassV2)
│   │       ├── hand_class.rs  # 19-variant hand classification system
│   │       ├── info_key.rs    # Info set key encoding (64-bit packed)
│   │       ├── agent.rs       # Agent system (TOML-configured play styles)
│   │       ├── simulation.rs  # Arena-based agent simulation
│   ├── range-solver/          # Exact postflop range solver (DCFR, PioSOLVER-compatible)
│   ├── range-solver-compare/  # Comparison harness vs b-inary/postflop-solver
│   ├── trainer/               # CLI for training, diagnostics
│   ├── tauri-app/             # Desktop strategy explorer (Tauri)
│   ├── devserver/             # HTTP mirror of Tauri API for browser debugging
│   ├── cfvnet/                # Deep CFV network: river datagen, training, evaluation
│   └── test-macros/           # #[timed_test] proc macro
├── frontend/                  # React/TypeScript explorer UI
│   └── src/
│       ├── Explorer.tsx       # Strategy browsing interface
│       ├── Simulator.tsx      # Agent simulation interface
│       └── invoke.ts          # Tauri/fetch abstraction layer
├── agents/                    # Agent personality configs (TOML)
├── sample_configurations/     # Training config presets (YAML)
├── docs/                      # Architecture, training, explorer, cloud docs
```

### Component Summary

**`core`** -- The heart of the solver. Contains the blueprint_v2 MCCFR solver (full game tree traversal with hand abstraction), shared CFR utilities (regret matching, DCFR/LCFR iteration weighting via `cfr/dcfr.rs`), card abstractions (HandClassV2, EHS2), hand classification, info set key encoding, and the agent simulation framework.

**`trainer`** -- CLI entry point that orchestrates blueprint training (`train-blueprint`), range solving, and diagnostics. Parses YAML configs and drives the core library.

**`tauri-app`** -- Desktop application for exploring solved strategies. Loads blueprint_v2 bundles, navigates the game tree, queries strategy/EV at any node, and runs agent simulations.

**`devserver`** -- Lightweight HTTP server that mirrors all Tauri exploration commands as POST endpoints, enabling browser-based UI development without the full Tauri build cycle.

**`frontend`** -- React/TypeScript UI shared between the Tauri app and dev server. Provides the strategy explorer (tree browsing, action frequencies, EV display) and agent simulator. Auto-detects Tauri vs browser environment.

**`range-solver`** -- Self-contained postflop solver that takes hero/villain ranges, bet sizes, and board cards, then solves to Nash equilibrium using Discounted CFR. Output-identical (exact f32 equality) to b-inary/postflop-solver. Supports PioSOLVER-compatible range syntax and bet size notation (pot-relative, previous-bet-relative, geometric, additive, all-in). Handles suit isomorphism for turn/river to skip redundant chance nodes.

**`range-solver-compare`** -- Test harness that generates random game configurations and verifies exact output identity between our range-solver and the original postflop-solver. Includes fast default tests (1000 river configs in ~21s) and slow soak tests for overnight validation.

**`cfvnet`** -- Neural network pipeline for learning river counterfactual values. Generates training data by solving random river situations with range-solver DCFR, trains a 7-layer MLP, and evaluates against exact solves. Enables depth-limited solving at river leaves.

**`agents/`** -- TOML files defining agent play styles (tight-aggressive, loose-aggressive, etc.) that map hand classes to action frequency distributions for simulation.

## Unit Convention

All internal values (pot, stacks, bet sizes, EVs) are in **chips**.

- 1 BB = 2 chips
- Config files use chips: `small_blind: 1`, `big_blind: 2`, `stack_depth: 200` (for 100 BB)
- Preflop action sizes use chip amounts with a `bb` suffix: `"5bb"` means raise to 5 chips (2.5 BB)
- Pot-fraction sizes (e.g. `0.67`) and multiplier sizes (e.g. `"3.0x"`) are unitless and unchanged
- Display to users converts chips to BB by dividing by 2, at the UI/CLI boundary only
- The range-solver (`TreeConfig`) also uses chips for `starting_pot` and `effective_stack`

## Blueprint V2 MCCFR Solver

**Algorithm:** External-sampling MCCFR with DCFR discounting. Samples random deals (hole cards + full board), traverses preflop through river, accumulates regrets at each information set, and stores the average strategy as action-frequency sums. The trainer can use either eager dense CFR storage or an opt-in sparse row backend; both feed the same traversal abstraction and export the same dense-compatible snapshot and strategy bundle formats.

**Key types:**
- `GameConfig` -- game structure: blinds, stacks, bet sizes, abstraction mode, DCFR params
- `HunlPostflop` -- implements the `Game` trait; manages game tree traversal with pre-dealt boards
- `MccfrSolver` -- external-sampling MCCFR traversal over the `BlueprintCfrStorage` abstraction
- `BlueprintStorage` -- eager dense flat buffers for regrets, strategy sums, optional baselines, and optional prediction values
- `SparseBlueprintStorage` -- opt-in HU lazy row storage keyed by stable decision-node identity, bucket, and action-schema fingerprint; missing rows read as zero/uniform and writes realize rows
- `BlueprintV2Strategy` -- serialized strategy extracted from solver; maps info set keys to action distributions

**Flow:**
1. Build `GameConfig` from YAML
2. Initialize `HunlPostflop` game with deal pool
3. Select `training.storage_backend` (`dense` by default, `sparse`/`lazy` opt-in)
4. Run MCCFR iterations with parallel batch processing
5. Extract `BlueprintV2Strategy` for exploration

**HU storage backends:**
- `dense` is the default and preserves historical behavior: every `(decision node, bucket, action)` regret and strategy-sum slot is allocated up front.
- `sparse`/`lazy` keeps the existing eager arena `GameTree` but realizes CFR rows only after traversal writes to a `(decision node, bucket)` pair. Reads of unrealized rows return zero regrets/sums/predictions/baselines and uniform current/average strategy.
- Sparse training uses the same optimizer, SAPCFR+ prediction, baseline, and regret-floor plumbing as dense storage. BRCFR+ remains dense-only in the current HU slice because its best-response prediction pass is still implemented against dense buffers.
- Snapshots and Explorer/Tauri bundles remain dense-compatible: sparse training projects to dense `regrets.bin` and `strategy.bin` at export/resume boundaries. There is no sparse on-disk snapshot format for HU `blueprint_v2`.
- Sparse progress logging includes realized rows/slots, dense-equivalent slots/bytes, approximate sparse resident bytes, inserts, and read/write probe counters.

**Shared training runtime:** `crates/core/src/training_runtime.rs` defines the backend-neutral runtime contract used to converge the HU and MP trainers. Runtime units are explicit (`Iteration` for HU blueprint_v2, `MetaIteration` for MP), and the runtime owns stop checks, pause/quit controls, snapshot/refresh/reload requests, elapsed-time limits, and counter updates. Backend adapters seed counters from restored/current trainer state but must not mutate runtime counters while running a batch.

**Universal dense blueprint format:** `docs/blueprint_format.md` specifies the versioned export format for HU, eager MP, and lazy sparse MP strategies. The format is row-oriented, records game/player/action/bucket provenance, distinguishes read-only strategy exports from resumable CFR snapshots, and keeps legacy HU bundles readable during migration. The core read/write module lives at `crates/core/src/blueprint_universal/` (manifest types, binary payload headers with CRC-64/XZ, row/action descriptors, f32 probability payloads, SHA-256 checksums, and a validating bundle writer/reader). The HU exporter (`blueprint_universal/hu_export.rs`, CLI `export-universal`) projects legacy `BlueprintV2Strategy` bundles into universal bundles with bitwise probability pass-through. The MP eager exporter (`blueprint_universal/mp_eager_export.rs`, CLI `export-universal-mp`) exports from live `MpStorage::average_strategy` or from saved snapshot dirs under the `mp_arena` namespace with explicit seats; both exporters share row machinery in `blueprint_universal/export_common.rs`. The MP lazy sparse exporter (`blueprint_universal/mp_lazy_export.rs`, via `export-universal-mp` kind dispatch) exports realized sparse rows under the `mp_semantic` namespace with verbatim semantic identity in a `strategy.semantic.bin` side table, gated by the `mp_semantic_rows_v1` required feature; rows realized by lazy traversal now carry stored action identity, so exports use concrete universal action descriptors, while synthetic rows without identity retain the safe `Opaque` fallback. The unified loader (`blueprint_universal/loader.rs`) provides `detect_bundle_kind` (cheap manifest-only detection of legacy HU vs universal HU/MP-eager/MP-lazy, `blueprint.json` taking precedence over a retained `config.yaml`, including native `snapshot_NNNN/universal/blueprint.json` output) and `load_bundle`, returning a `LoadedBundle` enum with a unified infoset query API; legacy and universal-HU loads return bitwise-identical results. The Explorer and devserver load via the unified loader (detecting `blueprint.json` before legacy `config.yaml`): universal HU bundles render through the existing HU views by reconstructing a `BlueprintV2Strategy` from the universal rows and rebuilding the tree from the `config.yaml` now retained inside each exported bundle; universal MP bundles load read-only (bundle info + manifest metadata), and the snapshot-specific `load_blueprint_v2` path delegates to nested universal loading before parsing MP `config.yaml` as HU config, with full N-player browsing UI deferred to follow-on work. Trainers write this format natively at snapshot time behind a `snapshots.format` config flag (`legacy` | `universal` | `both`, default `legacy`) for all three backends — HU, MP eager, MP lazy — with native output byte-identical to the post-hoc `export-universal` path; MP snapshot saves persist root `config.yaml` using the effective snapshot config so retained bundle config is available automatically. A `train` subcommand auto-detects HU-vs-MP config and dispatches. Remaining Phase 7 consolidation (one TUI, 2-10 players, retiring the HU/eager training paths) is tracked separately. Per user direction (2026-06-10): the end state is ONE lazy sparse training workflow supporting 2-10 players with ONE TUI; the only hard compatibility surface is the Tauri Explorer's ability to load exported strategies. Legacy-bundle conversion, HU blueprint_v2 training, and the MP eager dense backend are all transitional — retirable, not migration targets.

Universal MP-lazy loading uses a private owned-byte snapshot rather than
file-backed mappings. File-backed mappings cannot prevent another process from
truncating or rewriting a file, so the reader checks stable file identity,
length, and modification metadata before and after loading and validation, then
refuses queries when a source payload change is detectable. Strict eager
manifest, header,
SHA-256, CRC-64, structural, and probability-normalization validation remains
in place; row, action, probability, and semantic descriptors are materialized
during a query only as applicable. Queries use a compact sorted range locator
rather than a full row-key `HashMap`. Reader timings distinguish payload
loading, integrity checks, validation, and index setup. HU, eager MP, and legacy
readers retain their existing loading paths and behavior.

The mounted Tauri Game view now has a separate lazy-session adapter for exactly
two-player `universal_mp_lazy` bundles with a retained full MP config. It keeps
the semantic lazy row model, renders preflop rows by seat/street/bucket/history
identity, exposes a typed flop chance state while 0-2 cards are selected, and
renders completed-street matrices by enumerating canonical-hand combos, removing
board blockers, resolving each combo through trainer-compatible file-backed
`AllBuckets`, and averaging the matching sparse row probabilities. Bucket files
are resolved from bundle-local/ancestor `buckets/` directories or a valid
retained `training.cluster_path`. Relative configured paths are anchored at
the retained config directory and searched through its ancestors before
implicit `buckets/` candidates; missing sources, mappings, rows, and
incompatible action schemas are explicit errors. The narrow additive
`AllBuckets::try_get_bucket` API provides the same canonical/combo lookup with
errors instead of the trainer hot path's panic behavior. Card and action
updates are transactional, and back navigation replays the semantic action
path and retained board selection. The Universal MP lazy browser supports
two-player turn and river navigation, including dealing and rewinding those
streets. Its exact adapter supports non-terminal two-player Flop, Turn, and
River decision roots when the board is complete for the current street.
Preflop roots, incomplete-board chance states, terminal roots, Subgame solves,
eager MP bundles, and N-player exact solving remain unsupported. Eager MP and
N-player UI remain deferred.

**External baseline validation:**
- `training.baseline_validation` is an opt-in trainer diagnostic that compares learned average strategy frequencies against a pinned external preflop baseline JSON. It is separate from VR-MCCFR `use_baselines`; it does not change traversal, regrets, or strategy sums.
- The current validator is deliberately pinned to the 20bb HU cEV cash baseline at `local_data/baselines/cash_hu_20bb_cev.json`: stack 40 chips, blinds 1/2, no SB open limp, 169 preflop buckets, and preflop raise rows `2.5bb` then `5bb`.
- Integration passes `BaselineGamePreconditions` from the original `GameConfig` used to build the tree. The validator refuses scoring if trusted config metadata, tree shape, preflop bucket count, or baseline document schema do not match the pinned target.
- Reports read through the `BlueprintCfrStorage` provider boundary and call `average_strategy` on `active_storage()`. Sparse validation does not project the whole strategy to dense storage.
- Metrics are strategy-frequency distances, not EV pass/fail results: aggregate total variation, root and first-response total variation, worst-spot total variation, coverage, skipped zero-mass rows, invalid rows, unsupported spots/actions, and worst combo rows.

**Abstractions:**
- `HandClassV2` -- 19-class hand classification with intra-class strength and equity binning (28-bit hand field)
- `PotentialAwareEmd` -- True Pluribus-style potential-aware bucket abstraction (see below)
- Info set keys encode: hand (28 bits) | street (2) | SPR (5) | reserved (5) | actions (24)

**Key files:**
- Game: `crates/core/src/game/hunl_postflop.rs`
- Config: `crates/core/src/blueprint_v2/config.rs`
- MCCFR: `crates/core/src/blueprint_v2/mccfr.rs`
- Storage: `crates/core/src/blueprint_v2/storage.rs`
- Sparse storage: `crates/core/src/blueprint_v2/sparse_storage.rs`
- Trainer: `crates/core/src/blueprint_v2/trainer.rs`

### Potential-Aware Clustering Pipeline

The clustering pipeline computes card abstractions by running bottom-up from river to preflop. At each street, the feature vector for a (board, combo) pair is a distribution over the *next* street's bucket IDs — true potential-aware abstraction as described by Brown & Sandholm (Pluribus).

**Pipeline flow (all in memory, files written at the end):**

```
1. cluster_river()     → equity-based 1-D k-means           → river BucketFile
2. cluster_turn()      → histogram over river bucket IDs     → turn BucketFile
3. cluster_flop()      → histogram over turn bucket IDs      → flop BucketFile
4. cluster_preflop()   → histogram over flop bucket IDs      → preflop BucketFile
5. Write all 4 BucketFiles to disk
```

**Histogram construction (`build_bucket_histogram_u8`):** For each possible next-street card, extends the board, canonicalizes it via `canonical_key()`, looks up the board index in the previous street's `BucketFile` via `board_index_map()`, and increments the count for that combo's bucket ID. Returns raw u8 counts.

**Clustering:** Turn and flop use weighted EMD (Earth Mover's Distance) k-means over these bucket-ID histograms. By default, child-bucket ground distances are adjacent child centroid equity gaps, and sampled centroid training plus exhaustive assignment use the same metric. An opt-in per-street experimental metric can blend uniform potential movement, child centroid equity gaps, and sampled river nut-distance gaps through `<street>.metric.*` config weights. Enabled channels are normalized by their mean positive adjacent gap before blending. The nut-distance channel can additionally be capped and shaped (`linear`, `sqrt`, or `log1p`) after normalization so it works as a bounded hierarchy guardrail rather than overwhelming potential awareness. Clustering writes the observed scale factors to `metric_scales.json`. River uses equity-based 1-D k-means; preflop is a deterministic 169 canonical-hand map because the strategic abstraction starts postflop.

**Variants:** Each street has three clustering variants:
- **Canonical** (`cluster_*_canonical`): exhaustive enumeration of isomorphic boards with combinatorial weights
- **Sampling** (`cluster_*`): samples from canonical boards with weights
- **With-boards** (`cluster_*_with_boards`): raw random board sampling for testing

All variants store canonical `PackedBoard` entries in the `BucketFile.boards` field (version 2 format) for downstream lookup.

**Diagnostics** (`cluster_diagnostics.rs`):
- `cross_street_transition_matrix` -- counts (board, combo) transitions between adjacent streets
- `centroid_emd_report` -- pairwise EMD between reconstructed bucket centroids
- `sample_hands_for_bucket` -- sample hands from a specific bucket for inspection

**Key files:**
- Pipeline: `crates/core/src/blueprint_v2/cluster_pipeline.rs`
- BucketFile: `crates/core/src/blueprint_v2/bucket_file.rs`
- K-means: `crates/core/src/blueprint_v2/clustering.rs`
- Diagnostics: `crates/core/src/blueprint_v2/cluster_diagnostics.rs`
- Config: `crates/core/src/blueprint_v2/config.rs` (`ClusteringConfig`)

### Heuristic V3 Bucketing

An alternative card abstraction that uses two deterministic axes instead of EMD clustering:

- **Nut distance** (configurable, default 6 bits / 64 bins): fraction of possible opponent holdings that beat the hero's hand on the current board. 0 = absolute nuts, 63 = pure air.
- **Equity delta** (configurable, default 4 bits / 16 bins): expected change in equity from current street to next. Positive = draws improving, negative = vulnerable made hands. Zero on river (no future cards).

Default configuration produces **1,024 buckets per street** (64 on river where delta collapses to midpoint). Buckets are precomputed per-flop and stored in the existing `PerFlopBucketFile` format.

Key advantages over EMD clustering:
- **Deterministic**: same inputs always produce same buckets (no k-means convergence)
- **Fast precomputation**: direct bin assignment, no iterative clustering
- **Interpretable**: each bucket maps to a (nut_distance, equity_delta) pair

Select via config:
```yaml
clustering:
  algorithm: heuristic_v3
  nut_distance_bits: 6
  equity_delta_bits: 4
```

## N-Player Blueprint (`blueprint_mp`)

A clean-room N-player (2-10) MCCFR solver module alongside the existing `blueprint_v2`. Uses strong domain types and supports configurable blind/ante structures.

### Module Structure

```
crates/core/src/blueprint_mp/
├── types.rs            # Domain types: Seat, PlayerSet, Chips, Bucket, Street, Deal
├── config.rs           # BlueprintMpConfig with lead/raise split, ForcedBet blinds
├── game_tree.rs        # N-player game tree with fold-continuation
├── info_key.rs         # 128-bit InfoKey (seat + bucket + street + SPR + 22 actions)
├── terminal.rs         # Side pot resolution, showdown, fold payoffs
├── storage.rs          # Flat-buffer atomic regret/strategy storage
├── sparse_storage.rs   # Visited-infoset sparse atomic storage for lazy traversal
├── lazy_mccfr.rs       # Dynamic public-state traversal over sparse infoset storage
├── mccfr.rs            # External-sampling MCCFR traversal (Pluribus-style)
├── trainer.rs          # Training loop with per-seat traverser cycling, DCFR
├── training_runtime_adapter.rs # Shared-runtime adapter for lazy sparse MP
└── exploitability.rs   # Per-seat best-response diagnostic
```

### Key Design Decisions

- **2-10 players** with `MAX_PLAYERS = 10`
- **Configurable blinds**: SB, BB, ante, BB-ante, straddle via per-seat config
- **Lead/raise split**: Separate bet sizes for opening bets vs raises
- **Optional preflop flop-player cap**: `action_abstraction.max_flop_players` prunes non-closing preflop calls that would fill the last allowed flop seat, while preserving closing calls up to the cap
- **Full side pot resolution** at showdown terminals
- **128-bit info set keys** with 22 action slots (panics on overflow)
- **Pre-allocated eager storage** for the current backend: cumulative regrets use signed 32-bit atomics, and average-strategy sums use saturating unsigned 64-bit atomics
- **Sparse visited-infoset storage** for the planned lazy backend: unvisited infosets read as zero/uniform, visited infosets allocate sharded atomic regret and strategy counters, and snapshots export only touched entries
- **Symmetric integer regret discounting** across eager and sparse MP storage: discounted signed regrets are clamped to `i32` bounds and fractional results truncate toward zero, so both positive and negative values with magnitude below one become zero; unsigned strategy-sum discounting keeps its separate conversion semantics
- **Shared deterministic DCFR scheduler** across eager and lazy MP training: production runners own a monotonic `Instant` and feed completed meta-iterations plus elapsed `Duration` into pure scheduling state. Optional `training.dcfr_discount_interval_seconds` overrides the legacy iteration interval, arms when the scheduler starts at or beyond warmup or at the first later completed-batch observation that crosses warmup, and executes at most one pass per safe batch boundary. Missed wall-clock slots and crossed iteration boundaries are skipped without catch-up; wall-clock epochs count actual passes, while iteration mode retains the scheduled-boundary epoch.
- **Lazy public-state traversal** for 100bb migration: legal actions are generated on demand from compact betting state, chance/runout nodes are collapsed against the sampled full board, and sparse infoset keys combine seat, a street-namespaced abstract bucket, and action history
- **Experimental negative-action subtree purge** for lazy sparse traversal: aggressive action edges whose cumulative regret falls below a configured threshold are tracked in a sharded blocked-edge set; normal traversal masks blocked aggressive edges, while physical sparse-row deletion is deferred until the DCFR discount boundary, where post-discount regrets decide whether blocked child subtrees are purged or reactivated
- **External-sampling average strategy updates**: every visited decision infoset records the full current strategy vector; opponent actions are sampled only for recursion, not for average-strategy accounting
- **Shared-runtime lazy adapter**: `LazySparseMpTrainingRuntimeAdapter` wraps `LazyTrainContext` without changing lazy traversal or sparse key identity. Its unit is one MP meta-iteration: one sampled deal followed by one traversal per seat. `LazyMpTrainingStepper` preserves the old lazy loop's base-iteration, pruning RNG, chance-continuation, DCFR discount, and negative-action purge cadence while allowing the shared runtime to cap batches by `BatchBudget`. Snapshot, resume, and config reload hooks for lazy MP remain trainer-side work; the core adapter fails explicitly instead of faking support.
- **Universal dense export target**: the planned `dense_blueprint` bundle in `docs/blueprint_format.md` is the common strategy export contract for HU, eager MP, and lazy sparse MP. Lazy sparse exports are read-only until their blocked-edge purge state and runtime cadence are persisted.
- Shares `abstraction/`, `cfr/`, and `hand_eval` with `blueprint_v2`

### 100bb MP Scaling Plan

The current `blueprint_mp` backend eagerly materializes the full public betting tree and dense `(node, bucket, action)` storage. Normal 100bb 6-max configs with multiple preflop raise depths can exceed hundreds of millions of public nodes and hundreds of GB of virtual storage. The intended 100bb path is a lazy/sparse backend: traverse compact public states on demand, key regrets by stable infoset/action-history keys, and store only visited infosets. See `docs/plans/2026-05-07-blueprint-mp-100bb-design.md`.

### Lazy Sparse Negative-Action Purge

Lazy sparse MP training supports an opt-in sampled-prefix chance continuation
mode via `training.chance_continuation_mode`. The default `sampled_full_deal`
keeps the original full-board sampling behavior. `sampled_turn_exact_river`
samples private cards, flop, and turn as usual, precomputes every legal river
runout for that sampled turn prefix, and averages values over those river
runout for that sampled turn prefix. `sampled_flop_exact_turn_river` samples
private cards and flop, precomputes every legal turn/river runout for that flop
prefix, and averages values at flop-to-turn chance boundaries, turn-to-river
chance boundaries, and pre-river showdown terminals. Regret updates use the
averaged value, not the sum, so DCFR and pruning thresholds remain on the same
scale as sampled training.

The negative-action subtree purge is an opt-in experiment layered on the lazy sparse backend. It is not part of eager `blueprint_mp` storage, and it remains inactive until the configured warmup boundary (`meta_iter >= prune_after_iterations`). Ordinary regret-threshold traversal pruning is also opt-in via `traversal_pruning_enabled`; it only skips eligible traverser-side action branches for a batch and does not physically remove sparse rows or strategy sums. During post-warmup traversal, aggressive action edges can be gated by cumulative regret when the negative-action experiment is enabled. Passive actions (`Fold`, `Check`, `Call`, and all-in calls that do not increase the current max bet) are never persistent subtree-purge candidates, because their child histories can contain other players' future decisions. If an aggressive parent action regret drops below `negative_action_prune_below`, the edge is inserted into a sharded blocked-edge set with its packed child action-history prefix. Traversal skips blocked aggressive edges before child allocation, but it does not physically delete already visited descendant rows during ordinary traversal.

Physical purge runs immediately after lazy DCFR discounting. The boundary sweep scans the currently blocked edge set, rereads each parent action regret after discounting, and gives DCFR the first chance to soften or reactivate the edge. Edges whose regret reaches `negative_action_reactivate_at` are removed from the blocked set without deleting their child subtree. Remaining blocked child prefixes are batched into one sparse-storage scan for the discount boundary; matching rows at or below any stored child prefix are removed, preserving sibling histories while dropping already visited descendants below blocked actions.

The MP wall-clock discount scheduler is trainer-local and intentionally has no
checkpoint metadata in the current architecture. Its monotonic process-up time
includes training pauses, snapshots, and discount sweeps. A new process starts
a new timer, so process downtime is not counted. Persisting the pass epoch and
remaining deadline requires an atomic MP resume design and is deferred; the
optional interval likewise does not implement a maximum-pass stopping rule.

Persistent negative-action subtree purge only blocks and purges aggressive edges, because passive routing edges can contain later players' decision nodes. `prune_threshold` and `prune_explore_pct` control only ordinary traversal pruning when `traversal_pruning_enabled` is true; they do not enable physical subtree deletion.

## Range Solver (Exact Postflop Solver)

A self-contained postflop solver that computes Nash equilibrium strategies for specific hero/villain ranges on a given board. Unlike the blueprint solver (which uses hand abstraction), the range solver works with concrete hand combinations and produces exact strategies.

**Algorithm:** Discounted CFR (DCFR) with a=1.5, b=0.5, g=3.0. Strategy resets at power-of-4 iterations (4, 16, 64, ...).

**Key features:**
- PioSOLVER-compatible range syntax (AA, AKs, QQ-88, TT+, weights)
- Bet size notation: pot-relative (50%), previous-bet-relative (2.5x), geometric (2e), additive (100c), all-in (a)
- Suit isomorphism detection on turn/river to skip redundant chance nodes
- Rooted postflop trees for in-street subgame solves, including custom acting player, street stacks, current bet amount, and prior aggressive action metadata
- Arena-allocated game tree with `MutexLike` for lock-free interior mutability
- Two-pass O(n) terminal evaluation using sorted hand strength arrays

**CLI:** `cargo run -p poker-solver-trainer --release -- range-solve` -- see `docs/training.md` for full usage.

**Files:**
- `crates/range-solver/src/` -- solver, action tree, game tree, evaluation, isomorphism, hand evaluator
- `crates/range-solver-compare/` -- comparison harness and identity tests

## CFVnet (Deep Counterfactual Value Network)

A neural network pipeline for learning river-street counterfactual values, enabling depth-limited solving without computing full river subtrees at runtime.

**Crate:** `crates/cfvnet`

### Pipeline

```
generate -> train -> evaluate -> compare
```

1. **Generate** (`datagen`): Sample river situations, solve each with range-solver DCFR, extract pot-relative CFVs for both players, and write binary training records. Default river sampling uses random boards/pot/stack and RSP or precomputed blueprint preflop ranges. With `datagen.sampled_river_spots`, datagen instead samples a river board, walks a full blueprint bundle through preflop/flop/turn, and uses the reached river pot, stack, and line-conditioned normalized ranges. Datagen bet sizes are parsed into typed range-solver sizes: pot-relative entries may be fuzzed per sample, while explicit `a` entries remain all-in actions. Nested `game.bet_sizes` rows map to successive bet/raise rounds; raise depths beyond the configured rows are all-in only.

2. **Train** (`model`): Train a 7-layer MLP (2720->500->...->1326) using Huber loss + auxiliary game-value consistency loss. Framework: burn (wgpu/ndarray backends).

3. **Evaluate** (`eval`): Compute MAE, max error, and mbb/hand metrics on held-out data.

4. **Compare** (`eval`): Generate fresh river spots, solve exactly, compare network predictions against ground truth.

### Network Architecture

Two model variants share the same MLP architecture (`HiddenBlock`: Linear -> BatchNorm -> PReLU):

**CfvNet** (pot-relative output):
```
Input(2720) -> [Linear(500) -> BatchNorm -> PReLU] x 7 -> Linear(1326)
```
- Input: OOP range (1326) + IP range (1326) + board one-hot (52) + rank presence (13) + pot/400 + stack/400 + player
- Output: 1326 pot-relative counterfactual values

**BoundaryNet** (solver-native bcfv output):
```
Input(2720) -> [Linear(500) -> BatchNorm -> PReLU] x 7 -> Linear(1326)
```
- Input: OOP range (1326) + IP range (1326) + board one-hot (52) + rank presence (13) + `pot/(pot+stack)` + `stack/(pot+stack)` + player
- Range contract: canonical 1326-combo order; public-board-blocked combos zeroed; remaining finite non-negative mass normalized to sum 1 after blockers. River-enumerated adapters repeat this normalization after each candidate river blocker is removed.
- Output: 1326 normalized EVs (`chip_ev / (pot + effective_stack)`)
- At inference: `chip_ev[h] = normalized_ev[h] * (pot + effective_stack)`
- Dataset records may store CFVs as `chip_ev / pot`; that is a storage format only. Training loaders convert those records to the BoundaryNet target unit above.

Both use: Huber loss (masked for board-blocked combos) + lambda x auxiliary game-value constraint. ~2.9M parameters (default 7x500).

### Integration Point

**CfvNet** provides standalone river value predictions for evaluation and comparison.

**BoundaryNet** is wired into the range-solver as a depth-boundary evaluator via `NeuralBoundaryEvaluator`. The evaluator supports three explicit inference modes. `direct` sends the supplied boundary board directly to the model, which is the canonical contract for direct flop/turn/river BoundaryNet models and expects normalized EV output (`chip_ev / (pot + effective_stack)`). `river_enumerated_turn` is a legacy river-model adapter: on a 4-card turn board it evaluates every valid river runout and averages the river outputs. `direct_normalized_legacy` also sends the supplied boundary board directly to ONNX, but adapts the current Python-exported scaled-bcfv checkpoint output (`bcfv * pot / (pot + effective_stack)`) into the units required by the selected evaluator path. The ONNX evaluator batches OOP/IP rows together for each boundary cache fill, so direct modes perform one session run per boundary rather than one run per player.

Depth-boundary evaluators have two value conventions. Conditional evaluators return half-pot BCFV units (`chip_cfv / (pot / 2)`) and the range-solver converts them through the standard half-pot and blocker-aware opponent-reach formula before regret updates. This is the CFVNet/BoundaryNet path: the model receives normalized OOP/IP ranges and predicts a value conditional on the opponent reaching the boundary, so range-solver must still integrate the opponent's unnormalized reach. Raw CFV evaluators are reserved for exact/oracle evaluators that return already reach-integrated per-combination chip CFVs. For raw evaluators, each boundary visit caches only the current traverser's value slot; the opposite player is computed on that player's own traversal after their current opponent reach has been recorded.

### Key Files

- Config: `crates/cfvnet/src/config.rs`
- Range generator: `crates/cfvnet/src/datagen/range_gen.rs`
- Situation sampler: `crates/cfvnet/src/datagen/sampler.rs`
- Solve wrapper: `crates/cfvnet/src/datagen/solver.rs`
- CfvNet model: `crates/cfvnet/src/model/network.rs`
- BoundaryNet model: `crates/cfvnet/src/model/boundary_net.rs`
- BoundaryNet dataset encoding: `crates/cfvnet/src/model/boundary_dataset.rs`
- BoundaryNet training: `crates/cfvnet/src/model/boundary_training.rs`
- Loss functions: `crates/cfvnet/src/model/loss.rs`
- Training loop: `crates/cfvnet/src/model/training.rs`
- Boundary evaluator (range-solver integration): `crates/cfvnet/src/eval/boundary_evaluator.rs`
- CLI: `crates/cfvnet/src/main.rs`
- Sample config: `sample_configurations/river_cfvnet.yaml`

## Sampled Rollout Evaluator

The default boundary evaluator for subgame re-solving is a **depth-gated MCCFR sampling rollout**. When the range-solver hits a depth boundary (e.g., at river during a turn solve), it queries the blueprint strategy to estimate continuation values for each hero combo.

**Hybrid algorithm:** The evaluator uses exhaustive enumeration at shallow decision depths and Monte Carlo sampling at deeper ones. At decision depth < `enumerate_decision_depth` (default 2), all children are enumerated exactly weighted by the biased blueprint strategy. At decision depth >= the threshold, a single action is sampled from the biased strategy distribution and recursed into. Chance nodes always sample `num_rollouts` random cards (with a 3x sample boost at the first two chance levels for variance reduction).

This follows the approach described in **Modicum** (Brown, Sandholm & Amos, NeurIPS 2018): the first 1-2 decision levels carry the most entropy and have low branching cost, so exhaustive enumeration there preserves accuracy; deeper levels contribute geometrically less to the final value, making sampling sufficient. The stochastic noise at deeper levels is absorbed by DCFR's across-iteration averaging in the outer solver -- the same convergence property that Libratus, Pluribus, and Modicum rely on.

**Configurable knobs** (tunable via Tauri settings or CLI flags):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rollout_enumerate_depth` | 2 | Decision levels to enumerate before sampling. Set to 255 for fully exhaustive rollouts (old behavior). |
| `rollout_opponent_samples` | 8 | Opponent hands sampled per hero combo. Higher = less variance, slower. |
| `rollout_num_samples` | 3 | Chance-node samples (random runout cards) per evaluation. |

**Performance:** The sampling pivot yields ~100-200x speedup over exhaustive enumeration (e.g., 50ms vs 8.2s per evaluator call on a 1176-combo flop scenario) with < 1 mbb/hand mean error, validated by the `validate-rollout` CLI harness.

**Key files:**
- Rollout logic: `crates/core/src/blueprint_v2/continuation.rs`
- Evaluator construction: `crates/tauri-app/src/postflop.rs` (`build_rollout_evaluator`)
- Bench/validate CLI: `crates/trainer/src/bench_rollout.rs`, `crates/trainer/src/validate_rollout.rs`

## Safe Subgame Solving -- CFR-D Gadget (Option A2)

The subgame solver (`range-solver::PostFlopGame`) supports safe re-solving via a per-boundary CFR-D gadget per [Burch/Johanson/Bowling 2014 Section 3](https://webdocs.cs.ualberta.ca/~bowling/papers/14aaai-cfrd.pdf). When enabled (via the `--gadget` CLI flag or the Tauri `enable_gadget` command flag), a 4-node gadget subtree is injected at each cfvnet depth-boundary in the postflop tree:

```
Chance(river) -> G_IP  (Decision, owner=IP,  actions=[Terminate, Follow])
                  |-- Terminate_IP   (depth boundary terminal)
                  '-- Follow -> G_OOP (Decision, owner=OOP, actions=[Terminate, Follow])
                                 |-- Terminate_OOP  (depth boundary terminal)
                                 '-- Follow -> EXISTING cfvnet boundary
```

**Gadget Decision nodes** carry a `PLAYER_GADGET_FLAG` (bit 6 of the player byte) combined with the owner player bits. The `is_gadget()` and `gadget_owner()` methods on `PostFlopNode` decode these.

**Traverser-dependent activation (Option Y).** On a gadget Decision node, if `owner == traverser`, the solver runs standard CFR regret-matching -- the owner's own pass updates regrets and accumulates strategy sums. If `owner != traverser`, the gadget is disabled: the solver forces sigma=(0,1), skips directly to Follow, and performs no regret or strategy-sum update. Under the non-owner's pass the gadget behaves as if it does not exist. This matches the CFR-D semantics from [Burch 2014 Section 3](https://webdocs.cs.ualberta.ca/~bowling/papers/14aaai-cfrd.pdf) and [Brown & Sandholm 2017](https://arxiv.org/abs/1705.02955), where the opponent's gadget is visited but not regret-updated on the traverser's pass.

**Opt-out values** are computed once at game construction by `BlueprintCbvOptOut::from_cbv_context`. For each cfvnet depth-boundary (chance node descendant of the abstract tree root), the provider looks up the per-bucket CBV from the blueprint's `CbvTable`, normalizes by that boundary's own half-pot to produce bcfv units, and maps each concrete hand to its bucket. This yields per-boundary, per-player, per-hand opt-out values.

**Ordinal layout.** The original cfvnet boundaries retain ordinals 0..N. New gadget Terminate terminals occupy ordinals N..3N (for original boundary `b`: ordinal `N+2*b` = Terminate_IP, ordinal `N+2*b+1` = Terminate_OOP). Gadget terminal CFVs are pre-populated at injection time: the gadget player receives opt-out values; the non-gadget player receives zero. The zero is inert — under Option Y's traverser-disable semantics the non-owner's traversal short-circuits at the gadget Decision above each Terminate and never queries the non-gadget-player CFV at that terminal, so no zero-sum complement is needed.

**Safety invariant.** Regret-matching at each active gadget ensures `avg_realized_CFV[h] >= opt_out[h]` per gadget owner, per boundary, per hand. This is the Burch 2014 Section 3 sufficiency condition. Verified by the test `per_boundary_safety_invariant_avg_realized_cfv_geq_opt_out` at 0.01 tolerance after 1000 iterations.

**Tree invariant.** `game.root()` returns the real subgame root (not a gadget layer). Explorer, blueprint seeding, and all external arena-index consumers require no special handling.

**Key files:**
- Gadget config and tree injection: `crates/range-solver/src/game/gadget.rs` (`GadgetConfigPerBoundary`, `inject_per_boundary_gadgets`)
- Solver traverser-disable logic: `crates/range-solver/src/solver.rs` (gadget activation check near line 245)
- Compose helper and opt-out provider: `crates/tauri-app/src/gadget.rs` (`make_per_boundary_gadget_game`, `BlueprintCbvOptOut::from_cbv_context`)
- CLI routing: `crates/trainer/src/compare_solve.rs` (`--gadget` / `--gadget-clamp`, `build_gadget_tree_game`)

### Supersedes

Option A (root-level gadget, merged at `50799416`) placed two nested Decision nodes at the subgame root (arena indices 0--3). Option A2 replaces this with per-boundary placement, matching the standard CFR-D structural prescription. The post-clamp `GadgetEvaluator` (boundary-wrapper path) is retained for A/B diagnostic comparison via `--gadget-clamp`. Retirement tracked in a follow-up bean.

### Design History

- Original Option A design: `docs/plans/2026-04-24-option-a-deepstack-gadget-design.md` (documents the root-gadget design, now superseded by Option A2).
- Option A2 pivot rationale: `docs/plans/2026-04-24-option-a2-per-boundary-gadget-addendum.md`.

## Known Limitations

- **No real-time subgame solving yet:** The blueprint is a static strategy. Pluribus-style real-time search is planned but not implemented.
