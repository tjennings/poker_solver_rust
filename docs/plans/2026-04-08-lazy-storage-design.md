# Lazy Storage Allocation — Design Document

**Date:** 2026-04-08
**Status:** Approved

## Overview

Replace `Vec<AtomicI16>` / `Vec<AtomicI32>` pre-allocation in `MpStorage` with `mmap`-backed lazy buffers. Physical pages are only committed on first write, saving ~38% memory (matching Pluribus's 62% visit rate). Zero changes to the hot path.

## Approach

Anonymous `mmap` with platform-specific flags:
- **Linux:** `MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE`
- **macOS:** `MAP_PRIVATE | MAP_ANONYMOUS` (lazy by default)

Wrapped in `MmapBuffer<T>` that implements `Deref<Target=[T]>`.

## Changes

### New file: `crates/core/src/blueprint_mp/mmap_buffer.rs`

`MmapBuffer<T>` — allocates via mmap, zero-initialized, implements `Deref`/`DerefMut`, `Drop` calls `munmap`. ~60 lines total.

### Modified: `crates/core/src/blueprint_mp/storage.rs`

- `regrets: Vec<AtomicI16>` → `regrets: MmapBuffer<AtomicI16>`
- `strategy_sums: Vec<AtomicI32>` → `strategy_sums: MmapBuffer<AtomicI32>`
- `new()` constructor: `MmapBuffer::new(total)` instead of collect loop
- Size logging: show virtual size + estimated physical at 60% visit rate
- All accessors, indexing, DCFR discount, telemetry scan — unchanged (via Deref)

### No changes needed:
- `mccfr.rs`, `trainer.rs`, `exploitability.rs` — they access storage through `&[AtomicI16]` / `&[AtomicI32]` slices
- `mp_tui.rs` — telemetry scan uses `&storage.regrets` which derefs to `&[AtomicI16]`
- All tests — `MmapBuffer` derefs identically to Vec
