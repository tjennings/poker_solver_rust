# Universal Dense Blueprint Format

This document specifies the planned universal dense blueprint bundle. The format
is intentionally row-oriented: every strategy row is a normalized action
distribution plus enough identity metadata to make the row meaningful across
heads-up `blueprint_v2`, eager N-player `blueprint_mp`, and lazy sparse
N-player training.

Phase 0 is a specification only. Existing HU bundles remain readable, and no
trainer is required to write this format until the later implementation phases.

## Goals

- Use one directory format for HU and N-player exported strategies.
- Preserve game, player, action, abstraction, bucket, and training provenance.
- Support read-only exports from dense, sparse, and lazy sparse trainers.
- Detect incompatible bucket/action/game mismatches before a strategy is used.
- Keep resume/checkpoint semantics separate from read-only strategy loading.

## Non-Goals

- Do not change traversal semantics, CFR updates, pruning, or storage identity.
- Do not make lazy sparse MP snapshots resumable in this phase.
- Do not delete support for legacy HU `strategy.bin` bundles.
- Do not require Explorer to support N-player navigation before its integration
  phase.
- Do not store every unrealized lazy sparse row. Missing rows are represented by
  policy, not by synthetic rows.

## Directory Layout

Universal dense bundles are directory-first. A trainer may write this layout at
the bundle root, inside `snapshot_NNNN/`, or inside `final/`.

```text
bundle_or_snapshot/
  blueprint.json
  strategy.rows.bin
  strategy.actions.bin
  strategy.probs.f32.bin
  checksums.json
  cfr.snapshot.bin          # optional; present only for complete resumable state
  config.yaml               # optional retained source config / legacy context
  bucket_refs.json          # optional references to external bucket files
  buckets/                  # optional embedded bucket files
```

`blueprint.json` is the authoritative manifest. Binary files are payloads named
from manifest entries, not magic paths baked into readers.

## Manifest

The manifest is UTF-8 JSON. Unknown optional fields must be ignored. Unknown
required features must reject the bundle.

Required top-level fields:

| Field | Meaning |
|-|-|
| `format_name` | Must be `"dense_blueprint"` |
| `format_version` | Writer format version, starting at `1` |
| `compat_min_reader` | Oldest reader version expected to understand all required features |
| `created_at` | RFC3339 timestamp |
| `producer` | Crate/tool name and semantic version when available |
| `producer_git` | Git commit or `"unknown"` |
| `required_features` | Feature strings that a reader must support |
| `optional_features` | Feature strings safe to ignore |
| `game` | Game identity and player metadata |
| `training` | Iteration/unit/provenance metadata |
| `strategy` | Strategy payload metadata |
| `layout` | Row/action/probability counts and ordering |
| `actions` | Action abstraction metadata and fingerprints |
| `buckets` | Bucket counts and bucket-file identity |
| `compatibility` | Legacy/resume/missing-row behavior |
| `files` | Binary file names, sizes, and checksums |

### Game Metadata

`game` must identify the game independently of trainer internals:

- `game_kind`: currently `"holdem_no_limit"`.
- `num_players`: 2 through 10.
- `seats`: ordered seat descriptors with seat id, label, blind/ante role, and
  starting stack.
- `button_seat` and dealer/blind conventions used by the trainer.
- `small_blind`, `big_blind`, antes, straddles, and stack units in chips.
- `rake`: rate/cap metadata, even when zero.
- `max_flop_players` when a preflop cap is configured.

### Training Metadata

`training` records source and progress:

- `source_backend`: one of `hu_dense`, `hu_sparse_projected`,
  `mp_eager_dense`, `mp_lazy_sparse_projected`.
- `unit_kind`: `Iteration` for HU or `MetaIteration` for MP.
- `units_completed`: iteration/meta-iteration count.
- `elapsed_minutes`.
- `strategy_unit`: `average_strategy`, `current_regret_matched_strategy`, or
  `purified_average_strategy`.
- `command`, `config_path`, and relevant config fingerprint when available.

### Strategy Layout Metadata

`layout` must include:

- `row_count`.
- `action_descriptor_count`.
- `probability_count`.
- `row_sort_order`: stable order used in `strategy.rows.bin`.
- `row_namespace`: allowed namespaces in the row file.
- `missing_row_policy`: default behavior for absent rows.

Rows are sorted by `(namespace, seat, street, source_node_idx, global_bucket,
row_key_fingerprint)` unless a future version declares a different required
ordering. Readers must reject duplicate row identities.

## Binary File Header

Every binary payload starts with a fixed little-endian header:

```text
magic[8]        # file-specific ASCII magic
format_version  u16
header_version  u16
endianness      u8    # 1 = little endian
reserved[7]
header_len      u32
record_count    u64
payload_len     u64
payload_crc64   u64
```

The manifest also records file byte length and SHA-256. Readers must validate
magic, version compatibility, length, and checksum before exposing rows.

Initial magic values:

| File | Magic |
|-|-|
| `strategy.rows.bin` | `BPROW001` |
| `strategy.actions.bin` | `BPACT001` |
| `strategy.probs.f32.bin` | `BPPRO001` |
| `cfr.snapshot.bin` | `BPCFR001` |

## Row Descriptors

`strategy.rows.bin` stores fixed-width descriptors. Version 1 row descriptors
must contain these logical fields:

```text
row_id                     u64
namespace                  u16
seat                       u8
street                     u8
local_bucket               u16
reserved0                  u16
global_bucket              u32
source_node_idx            u32
action_offset              u64
action_count               u16
reserved1                  u16
prob_offset                u64
row_key_fingerprint        u64
action_schema_fingerprint  u64
semantic_key_kind          u16
semantic_key_offset        u64
```

`source_node_idx` is the arena node index for HU and eager MP rows. It is
`u32::MAX` for semantic-only lazy sparse rows. `global_bucket` must be stable
inside the bundle; for lazy sparse MP it expands the current packed
street-namespaced bucket into a full-width field.

Namespaces:

| Namespace | Meaning |
|-|-|
| `hu_arena` | HU `blueprint_v2` arena node plus bucket |
| `mp_arena` | Eager MP public-tree node plus seat plus bucket |
| `mp_semantic` | Lazy MP seat, street bucket, action-history semantic key |

Semantic key payloads are stored in a future side table when fixed fields are
not enough. Version 1 lazy MP semantic identity must at minimum preserve seat,
street, local/global bucket, action history hash, action history length, and
the high/low history words currently used by sparse storage.

## Action Descriptors

`strategy.actions.bin` stores the ordered legal actions for all rows. A row's
`action_offset..action_offset + action_count` slice points into this file.

Version 1 action descriptors must contain:

- `kind`: fold, check, call, bet, raise, all-in call, all-in bet/raise, or
  future required feature.
- `amount_chips`: resolved total amount in chips when applicable.
- `size_key`: canonical abstraction key, such as pot fraction, multiplier, or
  explicit all-in marker.
- `label`: display label, not part of strategy semantics.
- `is_aggressive`: true for bets, raises, and aggressive all-ins.
- `source_action_index`: original action index from the exporting trainer.

`action_schema_fingerprint` is computed from the ordered action kinds and
resolved amounts, plus the semantic size key. It must not depend on display
labels alone.

## Probability Payload

`strategy.probs.f32.bin` is a dense `f32` payload. A row's
`prob_offset..prob_offset + action_count` slice gives the action probabilities
for that row. Each row must be finite, non-negative, and normalized to a sum of
1 within the reader tolerance recorded in the manifest.

Rows exported from raw strategy sums use average strategy normalization. If the
source row has zero strategy-sum mass, the exporter writes the row's declared
missing/default strategy, normally uniform across legal actions, and marks the
row as zero-mass in optional row flags.

## Buckets

`buckets` records the abstraction identity needed to interpret row buckets:

- Per-street bucket counts.
- Bucket mode: embedded files, referenced external files, or preflop-only
  canonical classes.
- File name/path, byte size, and SHA-256 for each bucket file.
- Bucket generator version and semantic fingerprint.
- Per-flop bucket mode and canonicalization parameters when used.

Readers must reject a bundle when required bucket files are absent, checksums do
not match, or the requested runtime bucket abstraction fingerprint differs from
the bundle fingerprint.

## Checksums and Fingerprints

`checksums.json` duplicates the manifest file checksums for simple external
validation. The manifest remains authoritative when the two disagree.

Fingerprints are stable 64-bit hashes for internal identity checks, not
cryptographic integrity. Cryptographic integrity uses SHA-256 file checksums.
Fingerprint inputs must be versioned in the manifest so future changes do not
silently collide semantically.

Required fingerprints:

- Game fingerprint.
- Action abstraction fingerprint.
- Per-row action schema fingerprint.
- Bucket semantic fingerprint.
- Source config fingerprint when available.

## Compatibility Policy

The only hard compatibility obligation is the Tauri Explorer's ability to load
and browse exported strategies (user direction, 2026-06-10). Everything below
the first bullet is transitional convenience, droppable once the Explorer
reads universal bundles.

- Universal readers first look for `blueprint.json`; if absent, they may fall
  back to legacy HU loading.
- Legacy HU `config.yaml` plus `strategy.bin` bundles remain readable during
  the transition window.
- Writers may continue writing legacy `strategy.bin` during a compatibility
  window.
- MP eager and MP lazy exports must not masquerade as legacy HU bundles.
- `cfr.snapshot.bin` is optional. Its absence means the bundle is analysis-only.
- Lazy MP universal exports are analysis-only until blocked-edge state, purge
  cadence, RNG/runtime cadence, and sparse storage metadata are persisted.

## Missing Rows

Missing rows are legal only when `compatibility.missing_row_policy` allows them.
Initial policies:

| Policy | Meaning |
|-|-|
| `reject` | Any requested missing row is an error |
| `uniform_legal` | Return uniform probabilities over runtime legal actions |
| `zero_mass_uniform` | Return uniform and report the row as zero strategy-sum mass |

Dense HU and eager MP full exports should use `reject`. Lazy sparse projected
exports should use `uniform_legal` or `zero_mass_uniform` for unrealized rows,
depending on whether the caller needs to distinguish absent rows from present
zero-mass rows.

## Reader and Writer Architecture

The implementation should keep export sources separate from bundle IO:

```rust
trait StrategyRowSource {
    fn metadata(&self) -> BlueprintExportMetadata;
    fn rows(&self) -> Box<dyn Iterator<Item = StrategyRow> + '_>;
}

trait StrategyRowLookup {
    fn row_by_identity(&self, key: RowIdentity) -> Option<RowView<'_>>;
    fn default_for_missing(&self, key: RowIdentity) -> MissingRowPolicy;
}
```

Planned exporters:

- HU dense and HU sparse projected from `BlueprintCfrStorage::average_strategy`.
- MP eager from `MpStorage::average_strategy` plus public tree node metadata.
- MP lazy from `SparseSnapshotEntry`, normalizing `strategy_sums` and exporting
  realized semantic rows only.

Planned import flow:

1. Validate manifest, required features, file lengths, and checksums.
2. Memory-map or read binary payloads.
3. Validate sorted unique row identities, offsets, action counts, and row
   probability normalization.
4. Expose a read-only lookup.
5. Only if `cfr.snapshot.bin` is present and complete, expose a separate resume
   capability.

## Validation Plan

Phase implementation must include focused tests:

- Legacy HU `strategy.bin` to universal export to universal load preserves
  strategy probabilities row-for-row.
- MP eager export matches `MpStorage::average_strategy` for sampled and known
  public tree rows.
- MP lazy export matches `SparseSnapshotEntry` normalization, including
  zero-sum uniform fallback.
- Known row identity tests cover HU arena, MP eager arena, and MP lazy long
  action-history semantic keys.
- Readers reject wrong player counts, bucket mismatches, action schema
  mismatches, unknown required features, bad checksums, truncated binaries,
  duplicate row identities, invalid offsets, and non-normalized probabilities.
- Property tests cover row offset arithmetic, sorted unique identities, action
  descriptor round trips, and normalization tolerance.

Binary fuzzing is useful after the initial reader exists, but the first
implementation must prioritize deterministic rejection tests over broad fuzzing.
