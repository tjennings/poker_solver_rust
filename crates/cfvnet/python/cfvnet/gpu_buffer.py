"""GPU ring buffer with bulk parallel refill.

Architecture:
  - GPU buffer: large contiguous tensors on GPU, sampled for training
  - Refill: between epochs (or async), reader processes fill pinned CPU
    chunks in parallel, then one bulk copy moves them to random GPU slots

No staging ring, no per-record upload. Bulk operations only.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from cfvnet.constants import INPUT_SIZE, NUM_COMBOS
from cfvnet.data import (
    RECORD_SIZE_RIVER,
    _count_records_in_file,
    _decode_single_record,
    _resolve_bin_files,
)

_FLOATS_PER_RECORD = INPUT_SIZE + NUM_COMBOS * 3 + 2  # 6722


class GpuRingBuffer:
    """GPU-resident training buffer with bulk parallel refresh.

    Usage:
        buf = GpuRingBuffer(data_path, capacity=1_000_000, device=device)
        for epoch in range(epochs):
            for step in range(steps):
                batch = buf.sample_batch(batch_size)
                ...
            buf.refresh(num_records=100_000)  # replace 10% of pool
    """

    def __init__(
        self,
        data_path: Path,
        capacity: int,
        device: torch.device,
        num_workers: int = 8,
    ) -> None:
        self._device = device
        self._capacity = capacity
        self._num_workers = num_workers

        # Build file index.
        files = _resolve_bin_files(data_path)
        self._file_index: list[tuple[str, int]] = []
        skipped = 0
        for f in files:
            n = _count_records_in_file(f)
            if n <= 0:
                if n < 0:
                    skipped += 1
                continue
            for i in range(n):
                self._file_index.append((str(f), i * RECORD_SIZE_RIVER))

        if skipped > 0:
            print(f"  Skipped {skipped} non-standard files")
        self._total_records = len(self._file_index)
        print(f"  Indexed {self._total_records:,} records")

        # GPU tensors.
        gb = capacity * _FLOATS_PER_RECORD * 4 / 1024**3
        print(f"  Allocating GPU buffer: {capacity:,} records ({gb:.1f} GB)...")
        self.inputs = torch.zeros(capacity, INPUT_SIZE, device=device)
        self.targets = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.masks = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.ranges = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.game_values = torch.zeros(capacity, device=device)
        self.sample_weights = torch.zeros(capacity, device=device)

        self._filled = 0
        self._initial_fill()

    def _initial_fill(self) -> None:
        """Fill GPU buffer using parallel workers."""
        n = min(self._capacity, self._total_records)
        print(f"  Initial fill: {n:,} records using {self._num_workers} workers...")
        t0 = time.time()
        self._bulk_fill_slots(list(range(n)), list(range(n)))
        self._filled = n
        elapsed = time.time() - t0
        rate = n / max(elapsed, 0.001)
        print(f"  Initial fill complete: {n:,} records in {elapsed:.1f}s ({rate:.0f} rec/s)")

    def _bulk_fill_slots(self, gpu_slots: list[int], record_indices: list[int]) -> None:
        """Fill specific GPU slots with specific records using parallel workers.

        Workers fill a shared CPU buffer in parallel, then one bulk copy
        moves everything to GPU.
        """
        n = len(gpu_slots)
        if n == 0:
            return

        # Shared CPU buffer for all workers to write into.
        shared_buf = mp.Array(ctypes.c_float, n * _FLOATS_PER_RECORD, lock=False)

        # Split work across workers.
        chunk_size = (n + self._num_workers - 1) // self._num_workers
        processes: list[mp.Process] = []
        for i in range(self._num_workers):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            if start >= n:
                break
            p = mp.Process(
                target=_fill_chunk,
                args=(
                    self._file_index,
                    self._total_records,
                    record_indices[start:end],
                    shared_buf,
                    start,
                    end - start,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        # Bulk copy: shared CPU buffer → GPU.
        cpu_flat = np.frombuffer(shared_buf, dtype=np.float32).reshape(n, _FLOATS_PER_RECORD)
        gpu_indices = torch.tensor(gpu_slots, dtype=torch.long, device=self._device)

        off = 0
        self.inputs[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + INPUT_SIZE].copy()).to(self._device)
        off += INPUT_SIZE
        self.targets[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + NUM_COMBOS].copy()).to(self._device)
        off += NUM_COMBOS
        self.masks[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + NUM_COMBOS].copy()).to(self._device)
        off += NUM_COMBOS
        self.ranges[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + NUM_COMBOS].copy()).to(self._device)
        off += NUM_COMBOS
        self.game_values[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off].copy()).to(self._device)
        self.sample_weights[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off + 1].copy()).to(self._device)

    def sample_batch(self, batch_size: int) -> tuple:
        """Sample a random batch from GPU (no I/O, no transfer)."""
        n = min(self._filled, self._capacity)
        indices = torch.randint(0, n, (batch_size,), device=self._device)
        return (
            self.inputs[indices],
            self.targets[indices],
            self.masks[indices],
            self.ranges[indices],
            self.game_values[indices],
            self.sample_weights[indices],
        )

    def prepare_refresh(self, num_records: int) -> dict:
        """Read records from disk into CPU buffer (no GPU access).

        Can run in a background thread while GPU trains.

        Args:
            num_records: Number of records to prepare.

        Returns:
            Refresh context to pass to apply_refresh().
        """
        n = min(num_records, self._capacity)
        rng = np.random.default_rng()

        gpu_slots = rng.integers(0, self._filled, size=n).tolist()
        record_indices = rng.integers(0, self._total_records, size=n).tolist()

        # Parallel CPU read + encode into shared buffer.
        shared_buf = mp.Array(ctypes.c_float, n * _FLOATS_PER_RECORD, lock=False)
        chunk_size = (n + self._num_workers - 1) // self._num_workers
        processes: list[mp.Process] = []
        for i in range(self._num_workers):
            start = i * chunk_size
            end = min(start + chunk_size, n)
            if start >= n:
                break
            p = mp.Process(
                target=_fill_chunk,
                args=(
                    self._file_index, self._total_records,
                    record_indices[start:end], shared_buf, start, end - start,
                ),
                daemon=True,
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        return {"shared_buf": shared_buf, "gpu_slots": gpu_slots, "n": n}

    def apply_refresh(self, ctx: dict) -> None:
        """Bulk copy prepared CPU buffer to GPU (fast, ~milliseconds).

        Call from main thread between training steps.

        Args:
            ctx: Context dict from prepare_refresh().
        """
        n = ctx["n"]
        gpu_slots = ctx["gpu_slots"]
        cpu_flat = np.frombuffer(ctx["shared_buf"], dtype=np.float32).reshape(
            n, _FLOATS_PER_RECORD,
        )
        gpu_indices = torch.tensor(gpu_slots, dtype=torch.long, device=self._device)

        off = 0
        self.inputs[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + INPUT_SIZE].copy()).to(self._device)
        off += INPUT_SIZE
        self.targets[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + NUM_COMBOS].copy()).to(self._device)
        off += NUM_COMBOS
        self.masks[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + NUM_COMBOS].copy()).to(self._device)
        off += NUM_COMBOS
        self.ranges[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off:off + NUM_COMBOS].copy()).to(self._device)
        off += NUM_COMBOS
        self.game_values[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off].copy()).to(self._device)
        self.sample_weights[gpu_indices] = torch.from_numpy(
            cpu_flat[:, off + 1].copy()).to(self._device)

    def stop(self) -> None:
        """No-op (no persistent background workers)."""
        pass


def _fill_chunk(
    file_index: list[tuple[str, int]],
    total_records: int,
    record_indices: list[int],
    shared_buf: mp.Array,
    buf_offset: int,
    count: int,
) -> None:
    """Worker process: read + encode records into shared buffer.

    Each worker writes to its own slice of the shared buffer — no contention.
    """
    buf_np = np.frombuffer(shared_buf, dtype=np.float32).reshape(-1, _FLOATS_PER_RECORD)
    handles: OrderedDict = OrderedDict()

    # Scratch buffers.
    inp = np.zeros(INPUT_SIZE, dtype=np.float32)
    tgt = np.zeros(NUM_COMBOS, dtype=np.float32)
    msk = np.zeros(NUM_COMBOS, dtype=np.float32)
    rng = np.zeros(NUM_COMBOS, dtype=np.float32)
    gv = np.zeros(1, dtype=np.float32)
    sw = np.zeros(1, dtype=np.float32)

    for i in range(count):
        rec_idx = record_indices[i]
        fstr, byte_offset = file_index[rec_idx]

        fh = _get_handle(handles, fstr, 128)
        fh.seek(byte_offset)
        raw = np.frombuffer(fh.read(RECORD_SIZE_RIVER), dtype=np.uint8).copy()
        _decode_single_record(raw, 0, inp, tgt, msk, rng, gv, sw, 0)

        # Write to shared buffer (this worker's slice, no contention).
        row = buf_np[buf_offset + i]
        off = 0
        row[off:off + INPUT_SIZE] = inp
        off += INPUT_SIZE
        row[off:off + NUM_COMBOS] = tgt
        off += NUM_COMBOS
        row[off:off + NUM_COMBOS] = msk
        off += NUM_COMBOS
        row[off:off + NUM_COMBOS] = rng
        off += NUM_COMBOS
        row[off] = gv[0]
        row[off + 1] = sw[0]

    for fh in handles.values():
        fh.close()


def _get_handle(handles: OrderedDict, path: str, max_handles: int):
    """Get or open a file handle with LRU eviction."""
    fh = handles.get(path)
    if fh is not None:
        handles.move_to_end(path)
        return fh
    if len(handles) >= max_handles:
        _, old_fh = handles.popitem(last=False)
        old_fh.close()
    fh = open(path, "rb")  # noqa: SIM115
    handles[path] = fh
    return fh
