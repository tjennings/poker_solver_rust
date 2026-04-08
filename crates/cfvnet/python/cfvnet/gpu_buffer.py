"""GPU ring buffer with shared-memory refill pipeline.

Architecture:
  - GPU buffer: 1M+ records as contiguous tensors on GPU
  - CPU staging ring: shared-memory numpy arrays (no pickle)
  - Reader processes: continuously read random records, encode, write to staging
  - Upload thread: batch-copies ready staging slots to GPU

Readers run independently — they don't wait for training to consume records.
They continuously replace random GPU slots with fresh data from disk.
"""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import threading
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

# Staging slot states (int32).
_EMPTY = 0
_READY = 1

# Floats per record in staging buffer.
_FLOATS_PER_RECORD = INPUT_SIZE + NUM_COMBOS * 3 + 2  # 6722


class GpuRingBuffer:
    """GPU-resident training buffer with continuous background refresh."""

    def __init__(
        self,
        data_path: Path,
        capacity: int,
        device: torch.device,
        num_workers: int = 8,
        staging_size: int = 4096,
    ) -> None:
        self._device = device
        self._capacity = capacity
        self._num_workers = num_workers
        self._staging_size = staging_size

        # Build file index (str paths for cross-process compatibility).
        files = _resolve_bin_files(data_path)
        self._file_index: list[tuple[str, int]] = []
        skipped = 0
        for f in files:
            n = _count_records_in_file(f)
            if n <= 0:
                if n < 0:
                    skipped += 1
                continue
            fstr = str(f)
            for i in range(n):
                self._file_index.append((fstr, i * RECORD_SIZE_RIVER))

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

        # Shared-memory staging: data + status + gpu_slot per staging slot.
        self._staging_data = mp.Array(
            ctypes.c_float, staging_size * _FLOATS_PER_RECORD, lock=False,
        )
        self._staging_meta = mp.Array(
            ctypes.c_int32, staging_size * 2, lock=False,
        )  # [status, gpu_slot] per staging slot

        # Tracking.
        self._filled = 0
        self._stop_event = mp.Event()
        self._refill_count = mp.Value(ctypes.c_int64, 0)

        self._initial_fill()

    def _initial_fill(self) -> None:
        """Fill GPU buffer sequentially (single-threaded)."""
        n = min(self._capacity, self._total_records)
        print(f"  Initial fill: {n:,} records...")

        handles: OrderedDict = OrderedDict()
        inp = np.zeros(INPUT_SIZE, dtype=np.float32)
        tgt = np.zeros(NUM_COMBOS, dtype=np.float32)
        msk = np.zeros(NUM_COMBOS, dtype=np.float32)
        rng = np.zeros(NUM_COMBOS, dtype=np.float32)
        gv = np.zeros(1, dtype=np.float32)
        sw = np.zeros(1, dtype=np.float32)

        for slot in range(n):
            fstr, offset = self._file_index[slot % self._total_records]
            fh = _get_handle(handles, fstr, 128)
            fh.seek(offset)
            raw = np.frombuffer(fh.read(RECORD_SIZE_RIVER), dtype=np.uint8).copy()
            _decode_single_record(raw, 0, inp, tgt, msk, rng, gv, sw, 0)

            self.inputs[slot] = torch.from_numpy(inp)
            self.targets[slot] = torch.from_numpy(tgt)
            self.masks[slot] = torch.from_numpy(msk)
            self.ranges[slot] = torch.from_numpy(rng)
            self.game_values[slot] = float(gv[0])
            self.sample_weights[slot] = float(sw[0])

            if slot > 0 and slot % 100_000 == 0:
                print(f"    [{100 * slot // n}%] {slot:,}/{n:,}", flush=True)

        for fh in handles.values():
            fh.close()
        self._filled = n
        print(f"  Initial fill complete: {n:,} records on GPU")

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

    def start_refill(self) -> None:
        """Start continuous background refresh."""
        self._stop_event.clear()

        # Reader processes: read + encode → shared staging.
        self._processes: list[mp.Process] = []
        for i in range(self._num_workers):
            p = mp.Process(
                target=_reader_worker,
                args=(
                    self._file_index,
                    self._total_records,
                    self._capacity,
                    self._staging_data,
                    self._staging_meta,
                    self._staging_size,
                    self._stop_event,
                    i,  # worker_id for slot assignment
                    self._num_workers,
                ),
                name=f"reader-{i}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        # Upload thread: staging → GPU.
        self._upload_thread = threading.Thread(
            target=self._upload_worker, name="gpu-upload", daemon=True,
        )
        self._upload_thread.start()
        print(f"  Started {self._num_workers} reader processes + 1 upload thread")

    def _upload_worker(self) -> None:
        """Continuously scan staging for READY slots, batch-copy to GPU."""
        staging_np = np.frombuffer(self._staging_data, dtype=np.float32).reshape(
            self._staging_size, _FLOATS_PER_RECORD,
        )
        meta = np.frombuffer(self._staging_meta, dtype=np.int32).reshape(
            self._staging_size, 2,
        )

        while not self._stop_event.is_set():
            uploaded = 0
            for i in range(self._staging_size):
                if meta[i, 0] != _READY:
                    continue

                gpu_slot = meta[i, 1]
                row = staging_np[i]

                off = 0
                self.inputs[gpu_slot] = torch.from_numpy(row[off:off + INPUT_SIZE].copy())
                off += INPUT_SIZE
                self.targets[gpu_slot] = torch.from_numpy(row[off:off + NUM_COMBOS].copy())
                off += NUM_COMBOS
                self.masks[gpu_slot] = torch.from_numpy(row[off:off + NUM_COMBOS].copy())
                off += NUM_COMBOS
                self.ranges[gpu_slot] = torch.from_numpy(row[off:off + NUM_COMBOS].copy())
                off += NUM_COMBOS
                self.game_values[gpu_slot] = float(row[off])
                self.sample_weights[gpu_slot] = float(row[off + 1])

                meta[i, 0] = _EMPTY
                uploaded += 1

            if uploaded > 0:
                with self._refill_count.get_lock():
                    self._refill_count.value += uploaded
            else:
                time.sleep(0.001)

    def refill_count(self) -> int:
        """Return records replaced since last call, reset counter."""
        with self._refill_count.get_lock():
            count = self._refill_count.value
            self._refill_count.value = 0
        return count

    def stop(self) -> None:
        """Stop all background workers."""
        self._stop_event.set()
        if hasattr(self, "_upload_thread"):
            self._upload_thread.join(timeout=5.0)
        for p in getattr(self, "_processes", []):
            p.terminate()
            p.join(timeout=2.0)
        self._processes = []


def _reader_worker(
    file_index: list[tuple[str, int]],
    total_records: int,
    gpu_capacity: int,
    staging_data_raw: mp.Array,
    staging_meta_raw: mp.Array,
    staging_size: int,
    stop_event: mp.Event,
    worker_id: int,
    num_workers: int,
) -> None:
    """Reader process: continuously read records and write to staging.

    Each worker owns a stripe of staging slots (worker_id, worker_id + N, ...).
    No contention between workers — each slot has exactly one writer.
    """
    staging_np = np.frombuffer(staging_data_raw, dtype=np.float32).reshape(
        staging_size, _FLOATS_PER_RECORD,
    )
    meta = np.frombuffer(staging_meta_raw, dtype=np.int32).reshape(staging_size, 2)

    rng_gen = np.random.default_rng(seed=worker_id * 12345 + 1)
    handles: OrderedDict = OrderedDict()

    # Scratch buffers.
    inp = np.zeros(INPUT_SIZE, dtype=np.float32)
    tgt = np.zeros(NUM_COMBOS, dtype=np.float32)
    msk = np.zeros(NUM_COMBOS, dtype=np.float32)
    range_buf = np.zeros(NUM_COMBOS, dtype=np.float32)
    gv = np.zeros(1, dtype=np.float32)
    sw = np.zeros(1, dtype=np.float32)

    # This worker's staging slots: [worker_id, worker_id + N, worker_id + 2N, ...]
    my_slots = list(range(worker_id, staging_size, num_workers))

    while not stop_event.is_set():
        for slot_idx in my_slots:
            if stop_event.is_set():
                break

            # Wait for slot to be empty (uploaded or initial).
            if meta[slot_idx, 0] != _EMPTY:
                continue

            # Pick random record and GPU destination.
            rec_idx = rng_gen.integers(0, total_records)
            gpu_slot = rng_gen.integers(0, gpu_capacity)

            fstr, byte_offset = file_index[rec_idx]
            fh = _get_handle(handles, fstr, 128)
            fh.seek(byte_offset)
            raw = np.frombuffer(fh.read(RECORD_SIZE_RIVER), dtype=np.uint8).copy()

            _decode_single_record(raw, 0, inp, tgt, msk, range_buf, gv, sw, 0)

            # Write to shared staging (this worker owns this slot).
            row = staging_np[slot_idx]
            off = 0
            row[off:off + INPUT_SIZE] = inp
            off += INPUT_SIZE
            row[off:off + NUM_COMBOS] = tgt
            off += NUM_COMBOS
            row[off:off + NUM_COMBOS] = msk
            off += NUM_COMBOS
            row[off:off + NUM_COMBOS] = range_buf
            off += NUM_COMBOS
            row[off] = gv[0]
            row[off + 1] = sw[0]

            meta[slot_idx, 1] = gpu_slot
            meta[slot_idx, 0] = _READY

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
