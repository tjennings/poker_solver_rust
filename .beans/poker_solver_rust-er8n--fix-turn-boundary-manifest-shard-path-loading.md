---
# poker_solver_rust-er8n
title: Fix turn-boundary manifest shard path loading
status: completed
type: bug
priority: high
created_at: 2026-05-13T23:45:30Z
updated_at: 2026-05-13T23:54:06Z
---

Training turn-boundary v2 data fails because Python manifest loading resolves a shard as <dataset>/<dataset>/<shard>. Diagnose manifest path normalization and fix loader/writer compatibility so existing datasets train successfully.

## Implementation Checklist

- [x] Fix Python manifest shard path resolution for duplicated dataset prefixes
- [x] Fix Rust manifest writer lexical relative-prefix normalization
- [x] Add Python and Rust regression tests
- [x] Run focused verification

## Summary of Changes

- Added Python manifest shard resolution fallback for legacy relative shard paths that duplicate the dataset directory prefix.
- Updated Rust manifest path normalization so future manifests strip relative shard paths that already include the dataset directory.
- Added Python and Rust regressions for the duplicated-prefix case.

## Verification

- env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_manifest.py
- cargo test -p cfvnet datagen::manifest::tests

## Additional Verification

- Exact failing dataset path now resolves: `_resolve_bin_files(Path("../../../local_data/cfvnet/turn_boundary/v2"), expected_street="turn_boundary", expected_board_size=4)` returned 1000 shards with `a_BVZnf` first.
- `cargo test -p cfvnet` passed.
- Full Python `uv run pytest` reached `tests/test_e2e.py` after earlier tests passed, then hung for several minutes and was stopped; focused manifest/path tests passed.
