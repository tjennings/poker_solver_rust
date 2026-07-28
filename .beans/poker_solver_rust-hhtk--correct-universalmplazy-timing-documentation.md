---
# poker_solver_rust-hhtk
title: Correct UniversalMpLazy timing documentation
status: completed
type: task
priority: normal
created_at: 2026-07-28T14:49:32Z
updated_at: 2026-07-28T15:32:46Z
parent: poker_solver_rust-osss
---

Review follow-up: docs/explorer.md should distinguish [universal-reader] phase logs from the aggregate [explorer-load] log emitted by Tauri. Document the actual prefixes and timing availability accurately.

## Summary of Changes\n\n- Corrected docs/explorer.md to distinguish detailed [universal-reader] phase logs from aggregate [explorer-load] command timings.\n- Documented staged payload loading, integrity/validation/index phases, and deferred descriptor queries.\n\n## Verification\n\n- git diff --check passed.\n- Representative load emitted universal-reader phases: loading, integrity, validation, reader_ready, and index.
