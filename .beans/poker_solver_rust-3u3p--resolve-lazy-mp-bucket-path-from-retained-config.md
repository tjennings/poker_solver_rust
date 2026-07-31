---
# poker_solver_rust-3u3p
title: Resolve lazy MP bucket path from retained config
status: completed
type: bug
priority: high
created_at: 2026-07-17T20:58:10Z
updated_at: 2026-07-17T21:45:18Z
parent: poker_solver_rust-osss
---

The Tauri universal MP flop navigator fails to find the bucket files even though the retained MP config points to an existing cluster directory. The config uses a relative training.cluster_path such as ./local_data/buckets/500f_100t_100r_nut_high_cap_0p5_v1. The resolver currently searches the bundle and ancestors plus an unanchored relative path, so it misses the repository/config-root path and reports a false missing bucket source.

Use the retained config location as the anchor for relative training.cluster_path values. Preserve absolute paths, bundle-local/ancestor discovery, and explicit missing-source errors. Add a regression fixture with a relative cluster_path and a real or deterministic flop.buckets artifact, and verify the Tauri flop transition succeeds.

## Checklist

- [x] Research config path ownership and current bucket candidate resolution.
- [x] Anchor relative training.cluster_path to the retained config directory or equivalent project root.
- [x] Preserve absolute paths and existing bundle-local/ancestor candidates.
- [x] Add regression coverage for relative cluster_path resolution.
- [x] Run focused Tauri tests and verification.
- [x] Update docs if path behavior is user-visible.

## Summary of Changes

- Anchored relative `training.cluster_path` values to the retained config directory and its ancestors, while preserving absolute and process-CWD compatibility fallbacks.
- Prioritized configured bucket sources, validated flop bucket metadata, and prevented invalid implicit candidates from masking valid configured sources.
- Added nested snapshot regression coverage for configured-path resolution and invalid configured-source errors; documented the behavior in the Explorer and architecture guides.

Parent: unified HU/MP trainer runtime epic.
