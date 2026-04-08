"""GPU ring buffer for training data with shared-memory refill pipeline.

Architecture:
  - GPU buffer: large contiguous tensors on GPU, sampled for training
  - CPU staging ring: shared-memory numpy arrays, workers write encoded records
  - Reader processes: read from disk, encode, write to staging (no pickle)
  - Upload thread: batch-copies staging → GPU, replaces consumed slots

No per-record serialization. Workers write directly to shared memory.
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

# Staging slot states.
_SLOT_EMPTY = 0
_SLOT_WRITING = 1
_SLOT_READY = 2

# Size of each field per record.
_RECORD_FLOATS = INPUT_SIZE + NUM_COMBOS * 3 + 2  # inp + target + mask + range + gv + sw


class GpuRingBuffer:
    """GPU-resident ring buffer with shared-memory refill pipeline.

    Usage:
        buf = GpuRingBuffer(data_path, capacity=1_000_000, device=device)
        buf.start_refill()
        for step in range(total_steps):
            batch = buf.sample_batch(batch_size)
            ...
        buf.stop()
    """

    def __init__(
        self,
        data_path: Path,
        capacity: int,
        device: torch.device,
        num_workers: int = 8,
        staging_size: int = 10_000,
    ) -> None:
        self._device = device
        self._capacity = capacity
        self._num_workers = num_workers

        # Build file index.
        files = _resolve_bin_files(data_path)
        self._file_index: list[tuple[str, int]] = []  # str paths for pickling
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
            print(f"  Skipped {skipped} files with non-standard record sizes")
        self._total_records = len(self._file_index)
        print(f"  Indexed {self._total_records:,} records")

        # GPU tensors.
        gb = capacity * _RECORD_FLOATS * 4 / 1024**3
        print(f"  Allocating GPU buffer for {capacity:,} records ({gb:.1f} GB)...")
        self.inputs = torch.zeros(capacity, INPUT_SIZE, device=device)
        self.targets = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.masks = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.ranges = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.game_values = torch.zeros(capacity, device=device)
        self.sample_weights = torch.zeros(capacity, device=device)

        # Shared-memory staging ring.
        self._staging_size = staging_size
        self._staging_data = mp.Array(ctypes.c_float, staging_size * _RECORD_FLOATS, lock=False)
        self._staging_slots = mp.Array(ctypes.c_int32, staging_size * 2, lock=False)
        # Each staging slot has: [status, gpu_slot_index]
        # Initialize all as empty.
        for i in range(staging_size):
            self._staging_slots[i * 2] = _SLOT_EMPTY
            self._staging_slots[i * 2 + 1] = -1

        # Consumed GPU slots queue — training puts slot indices here.
        self._consumed_queue: mp.Queue = mp.Queue()

        # Tracking.
        self._filled = 0
        self._stop_event = mp.Event()
        self._refill_count = 0
        self._refill_count_lock = threading.Lock()

        # Initial fill.
        self._initial_fill()

    def _initial_fill(self) -> None:
        """Fill GPU buffer with first `capacity` records (single-threaded)."""
        n = min(self._capacity, self._total_records)
        print(f"  Initial fill: loading {n:,} records to GPU...")

        handles: OrderedDict = OrderedDict()
        chunk = 10_000
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            self._fill_range_direct(start, end, handles)
            if start > 0 and (start // chunk) % 10 == 0:
                print(f"    [{100.0 * end / n:.0f}%] {end:,}/{n:,}", flush=True)

        for fh in handles.values():
            fh.close()
        self._filled = n
        print(f"  Initial fill complete: {n:,} records on GPU")

    def _fill_range_direct(self, start: int, end: int, handles: OrderedDict) -> None:
        """Fill GPU slots directly (for initial fill, no staging)."""
        inp = np.zeros(INPUT_SIZE, dtype=np.float32)
        target = np.zeros(NUM_COMBOS, dtype=np.float32)
        mask = np.zeros(NUM_COMBOS, dtype=np.float32)
        rng = np.zeros(NUM_COMBOS, dtype=np.float32)
        gv = np.zeros(1, dtype=np.float32)
        sw = np.zeros(1, dtype=np.float32)

        for slot in range(start, end):
            rec_idx = slot % self._total_records
            fstr, offset = self._file_index[rec_idx]
            fh = _get_cached_handle(handles, fstr, 128)
            fh.seek(offset)
            raw = np.frombuffer(fh.read(RECORD_SIZE_RIVER), dtype=np.uint8).copy()

            _decode_single_record(raw, 0, inp, target, mask, rng, gv, sw, 0)

            self.inputs[slot] = torch.from_numpy(inp)
            self.targets[slot] = torch.from_numpy(target)
            self.masks[slot] = torch.from_numpy(mask)
            self.ranges[slot] = torch.from_numpy(rng)
            self.game_values[slot] = float(gv[0])
            self.sample_weights[slot] = float(sw[0])

    def sample_batch(self, batch_size: int) -> tuple:
        """Sample a random batch from GPU. Queues consumed slots for refill."""
        n = min(self._filled, self._capacity)
        indices = torch.randint(0, n, (batch_size,), device=self._device)

        # Queue consumed indices for refill (batch put).
        idx_list = indices.cpu().tolist()
        for idx in idx_list:
            self._consumed_queue.put_nowait(idx)

        return (
            self.inputs[indices],
            self.targets[indices],
            self.masks[indices],
            self.ranges[indices],
            self.game_values[indices],
            self.sample_weights[indices],
        )

    def start_refill(self) -> None:
        """Start refill pipeline: dispatcher + reader processes + upload thread."""
        self._stop_event.clear()

        # Dispatcher thread: moves consumed GPU slots to staging slots.
        self._dispatcher = threading.Thread(
            target=self._dispatch_worker, name="dispatcher", daemon=True,
        )
        self._dispatcher.start()

        # Reader processes: read from disk, encode, write to staging.
        self._processes: list[mp.Process] = []
        for i in range(self._num_workers):
            p = mp.Process(
                target=_reader_worker,
                args=(
                    self._file_index,
                    self._total_records,
                    self._staging_data,
                    self._staging_slots,
                    self._staging_size,
                    self._stop_event,
                ),
                name=f"reader-{i}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        # Upload thread: copies ready staging slots to GPU.
        self._upload_thread = threading.Thread(
            target=self._upload_worker, name="gpu-upload", daemon=True,
        )
        self._upload_thread.start()

        print(f"  Started {self._num_workers} reader processes + upload pipeline")

    def _dispatch_worker(self) -> None:
        """Move consumed GPU slot indices into staging ring for refill."""
        while not self._stop_event.is_set():
            try:
                gpu_slot = self._consumed_queue.get(timeout=0.1)
            except Exception:
                continue

            # Find an empty staging slot.
            placed = False
            for _ in range(self._staging_size * 2):
                for i in range(self._staging_size):
                    base = i * 2
                    if self._staging_slots[base] == _SLOT_EMPTY:
                        self._staging_slots[base + 1] = gpu_slot
                        self._staging_slots[base] = _SLOT_WRITING
                        placed = True
                        break
                if placed:
                    break
                time.sleep(0.001)
            # If we couldn't place it, drop it (buffer stays stale for this slot).

    def _upload_worker(self) -> None:
        """Batch-copy ready staging slots to GPU."""
        staging_np = np.frombuffer(self._staging_data, dtype=np.float32).reshape(
            self._staging_size, _RECORD_FLOATS,
        )

        while not self._stop_event.is_set():
            uploaded = 0
            for i in range(self._staging_size):
                base = i * 2
                if self._staging_slots[base] != _SLOT_READY:
                    continue

                gpu_slot = self._staging_slots[base + 1]
                row = staging_np[i]

                # Unpack: inp(2720) + target(1326) + mask(1326) + range(1326) + gv(1) + sw(1)
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

                # Mark slot empty for reuse.
                self._staging_slots[base] = _SLOT_EMPTY
                uploaded += 1

            if uploaded > 0:
                with self._refill_count_lock:
                    self._refill_count += uploaded
            else:
                time.sleep(0.001)

    def refill_count(self) -> int:
        """Return records replaced since last call and reset counter."""
        with self._refill_count_lock:
            count = self._refill_count
            self._refill_count = 0
        return count

    def stop(self) -> None:
        """Stop all background workers."""
        self._stop_event.set()
        if hasattr(self, "_dispatcher"):
            self._dispatcher.join(timeout=5.0)
        if hasattr(self, "_upload_thread"):
            self._upload_thread.join(timeout=5.0)
        for p in getattr(self, "_processes", []):
            p.terminate()
            p.join(timeout=2.0)
        self._processes = []


def _reader_worker(
    file_index: list[tuple[str, int]],
    total_records: int,
    staging_data_raw: mp.Array,
    staging_slots_raw: mp.Array,
    staging_size: int,
    stop_event: mp.Event,
) -> None:
    """Reader process: find WRITING slots, read+encode from disk, mark READY.

    Writes directly to shared memory — no serialization.
    """
    staging_np = np.frombuffer(staging_data_raw, dtype=np.float32).reshape(
        staging_size, _RECORD_FLOATS,
    )
    staging_slots = np.frombuffer(staging_slots_raw, dtype=np.int32).reshape(staging_size, 2)

    rng_gen = np.random.default_rng()
    handles: OrderedDict = OrderedDict()

    # Scratch buffers (reused per record).
    inp = np.zeros(INPUT_SIZE, dtype=np.float32)
    target = np.zeros(NUM_COMBOS, dtype=np.float32)
    mask = np.zeros(NUM_COMBOS, dtype=np.float32)
    range_buf = np.zeros(NUM_COMBOS, dtype=np.float32)
    gv = np.zeros(1, dtype=np.float32)
    sw = np.zeros(1, dtype=np.float32)

    while not stop_event.is_set():
        # Scan for a WRITING slot to claim.
        found = -1
        for i in range(staging_size):
            if staging_slots[i, 0] == _SLOT_WRITING:
                found = i
                break

        if found < 0:
            time.sleep(0.001)
            continue

        # Read a random record from disk.
        rec_idx = rng_gen.integers(0, total_records)
        fstr, byte_offset = file_index[rec_idx]
        fh = _get_cached_handle(handles, fstr, 128)
        fh.seek(byte_offset)
        raw = np.frombuffer(fh.read(RECORD_SIZE_RIVER), dtype=np.uint8).copy()

        # Encode into scratch buffers.
        _decode_single_record(raw, 0, inp, target, mask, range_buf, gv, sw, 0)

        # Write to shared memory staging slot (zero-copy within process).
        row = staging_np[found]
        off = 0
        row[off:off + INPUT_SIZE] = inp
        off += INPUT_SIZE
        row[off:off + NUM_COMBOS] = target
        off += NUM_COMBOS
        row[off:off + NUM_COMBOS] = mask
        off += NUM_COMBOS
        row[off:off + NUM_COMBOS] = range_buf
        off += NUM_COMBOS
        row[off] = gv[0]
        row[off + 1] = sw[0]

        # Mark ready for GPU upload.
        staging_slots[found, 0] = _SLOT_READY

    for fh in handles.values():
        fh.close()


def _get_cached_handle(handles: OrderedDict, path: str, max_handles: int):
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
