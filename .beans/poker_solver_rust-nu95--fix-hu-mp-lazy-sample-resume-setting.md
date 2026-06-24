---
# poker_solver_rust-nu95
title: Fix HU MP lazy sample resume setting
status: in-progress
type: bug
priority: high
created_at: 2026-06-24T14:38:45Z
updated_at: 2026-06-24T14:38:45Z
---

The new HU MP lazy sample inherited snapshots.resume=true from the 6-max base, but lazy_sparse resume is explicitly unsupported. Set resume=false, verify the config inspector still passes, smoke the train command far enough to confirm it no longer exits with the resume error, and commit the config plus tracker.
