---
# poker_solver_rust-qhtw
title: Harden universal bundle header and normalization validation
status: todo
type: bug
priority: high
created_at: 2026-07-28T14:49:32Z
updated_at: 2026-07-28T14:49:32Z
parent: poker_solver_rust-osss
---

Review follow-up outside the mmap optimization: manifest probability normalization tolerance currently accepts arbitrary f64 values, and binary header parsing does not reject invalid version, endianness, header length, or reserved bytes despite the format contract. Add bounded validation and corruption tests without changing valid bundles.
