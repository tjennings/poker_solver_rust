---
# poker_solver_rust-9w7c
title: Bump CFVNet GPU buffer refresh to 50 percent
status: completed
type: task
priority: normal
created_at: 2026-05-14T00:30:06Z
updated_at: 2026-05-14T00:31:30Z
---

Update boundary training GPU ring buffer refresh from 10% of the active pool per epoch to 50%, so new training runs cycle through the dataset faster.

## Summary of Changes

Updated CFVNet boundary training so the GPU ring buffer refreshes 50% of the active pool per epoch instead of 10%. Added a small helper and regression test for the refresh count.

## Verification

- env UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_train.py
