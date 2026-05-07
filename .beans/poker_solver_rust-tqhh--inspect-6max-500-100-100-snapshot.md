---
# poker_solver_rust-tqhh
title: Inspect 6max 500-100-100 snapshot
status: completed
type: task
priority: normal
created_at: 2026-05-07T03:24:51Z
updated_at: 2026-05-07T03:26:00Z
---

Inspect the snapshot written from the running 6max 500/100/100 Blueprint MP training run and summarize artifact contents and any immediate concerns.



## Summary of Changes

Inspected local snapshot artifacts after the user reported writing a snapshot from the running 6max 500/100/100 Blueprint MP training run.

Findings:
- No snapshot directory or recent strategy/regret file was found under the configured output_dir local_data/blueprints/mp_6max_20bb_simplified_actions_500f_100t_100r_v1.
- No recent snapshot/strategy/regret artifact was found elsewhere under local_data or the repo-level snapshot search.
- In the MP TUI path, pressing s calls metrics.request_snapshot(), but bridge_mp_iterations only pushes telemetry and strategy grids; unlike the HU Blueprint V2 trainer path, it does not consume the snapshot trigger or write a persisted bundle/snapshot.
- ps is blocked in this sandbox, so the live process command line could not be confirmed.

Conclusion: if the snapshot was requested via the MP TUI s key, it likely did not persist to disk. This should be tracked as an MP snapshot persistence bug/feature if persisted snapshots are needed for evaluation.
