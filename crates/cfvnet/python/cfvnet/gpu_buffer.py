"""GPU ring buffer for training data.

Maintains a large buffer of training records on GPU memory. Background
threads continuously read from disk and replace consumed records. The
training loop samples batches directly from GPU — no per-batch I/O or
CPU-GPU transfer.
"""

from __future__ import annotations

import multiprocessing as mp
import threading
import time
from pathlib import Path
from queue import Queue

import numpy as np
import torch

from cfvnet.constants import INPUT_SIZE, NUM_COMBOS
from cfvnet.data import (
    RECORD_SIZE_RIVER,
    _count_records_in_file,
    _encode_raw_to_tensors,
    _resolve_bin_files,
)


class GpuRingBuffer:
    """GPU-resident ring buffer with async disk refill.

    The buffer holds `capacity` records as contiguous GPU tensors.
    Training samples random indices from the buffer. Background threads
    read fresh records from disk and replace consumed slots.

    Usage:
        buf = GpuRingBuffer(data_path, capacity=500_000, device=device)
        buf.start_refill(num_workers=4)

        for step in range(total_steps):
            batch = buf.sample_batch(batch_size)
            # ... train on batch ...

        buf.stop()
    """

    def __init__(
        self,
        data_path: Path,
        capacity: int,
        device: torch.device,
        num_workers: int = 4,
    ) -> None:
        """Initialize the GPU buffer.

        Args:
            data_path: Path to training data (file or directory).
            capacity: Number of records to hold on GPU.
            device: Torch device (must be CUDA).
            num_workers: Number of background reader threads.
        """
        self._device = device
        self._capacity = capacity
        self._num_workers = num_workers

        # Build file index for random access.
        files = _resolve_bin_files(data_path)
        self._file_index: list[tuple[Path, int]] = []
        skipped = 0
        for f in files:
            n = _count_records_in_file(f)
            if n <= 0:
                if n < 0:
                    skipped += 1
                continue
            for i in range(n):
                self._file_index.append((f, i * RECORD_SIZE_RIVER))

        if skipped > 0:
            print(f"  Skipped {skipped} files with non-standard record sizes")
        self._total_records = len(self._file_index)
        print(f"  Indexed {self._total_records:,} records")

        # Allocate GPU tensors.
        print(f"  Allocating GPU buffer for {capacity:,} records "
              f"({capacity * (INPUT_SIZE + NUM_COMBOS * 3 + 2) * 4 / 1024**3:.1f} GB)...")
        self.inputs = torch.zeros(capacity, INPUT_SIZE, device=device)
        self.targets = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.masks = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.ranges = torch.zeros(capacity, NUM_COMBOS, device=device)
        self.game_values = torch.zeros(capacity, device=device)
        self.sample_weights = torch.zeros(capacity, device=device)

        # Track which slots have been consumed and need refilling.
        self._filled = 0  # How many slots have valid data.
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._refill_queue: mp.Queue = mp.Queue()  # Slot indices to refill.
        self._workers: list[threading.Thread] = []

        # Refill rate tracking.
        self._refill_count = 0
        self._refill_count_lock = threading.Lock()
        self._last_refill_time = time.time()

        # File handle cache per thread (thread-local).
        self._local = threading.local()

        # Initial fill.
        self._initial_fill()

    def _initial_fill(self) -> None:
        """Fill the buffer with the first `capacity` records."""
        n = min(self._capacity, self._total_records)
        print(f"  Initial fill: loading {n:,} records to GPU...")

        # Read in batches to avoid holding too much CPU memory.
        chunk_size = 10_000
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            self._fill_range(start, end)
            if (start // chunk_size) % 10 == 0 and start > 0:
                print(f"    [{100.0 * end / n:.0f}%] {end:,}/{n:,}", flush=True)

        self._filled = n
        print(f"  Initial fill complete: {n:,} records on GPU")

    def _fill_range(self, slot_start: int, slot_end: int) -> None:
        """Fill buffer slots [slot_start, slot_end) from the file index."""
        for slot in range(slot_start, slot_end):
            record_idx = slot % self._total_records
            self._fill_slot(slot, record_idx)

    def _fill_slot(self, slot: int, record_idx: int) -> None:
        """Read one record from disk, encode, and upload to GPU slot."""
        file_path, byte_offset = self._file_index[record_idx]
        raw = self._read_cached(file_path, byte_offset)
        inp, target, mask, rng, gv, sw = _encode_raw_to_tensors(raw)

        # Upload to GPU (individual slot assignment).
        self.inputs[slot] = torch.from_numpy(inp)
        self.targets[slot] = torch.from_numpy(target)
        self.masks[slot] = torch.from_numpy(mask)
        self.ranges[slot] = torch.from_numpy(rng)
        self.game_values[slot] = float(gv)
        self.sample_weights[slot] = float(sw)

    _MAX_OPEN_FILES = 128  # Per-thread file handle cache limit.

    def _read_cached(self, path: Path, offset: int) -> np.ndarray:
        """Read record bytes using thread-local LRU file handle cache."""
        cache = getattr(self._local, "handles", None)
        if cache is None:
            from collections import OrderedDict
            cache = OrderedDict()
            self._local.handles = cache

        fh = cache.get(path)
        if fh is not None:
            cache.move_to_end(path)
        else:
            if len(cache) >= self._MAX_OPEN_FILES:
                _, old_fh = cache.popitem(last=False)
                old_fh.close()
            fh = open(path, "rb")  # noqa: SIM115
            cache[path] = fh

        fh.seek(offset)
        buf = fh.read(RECORD_SIZE_RIVER)
        return np.frombuffer(buf, dtype=np.uint8).copy()

    def sample_batch(self, batch_size: int) -> tuple:
        """Sample a random batch from the buffer.

        Returns GPU tensors ready for training (no transfer needed).

        Args:
            batch_size: Number of records per batch.

        Returns:
            Tuple of (input, target, mask, range, game_value, sample_weight).
        """
        n = min(self._filled, self._capacity)
        indices = torch.randint(0, n, (batch_size,), device=self._device)

        # Queue consumed indices for refill.
        for idx in indices.cpu().tolist():
            self._refill_queue.put(idx)

        return (
            self.inputs[indices],
            self.targets[indices],
            self.masks[indices],
            self.ranges[indices],
            self.game_values[indices],
            self.sample_weights[indices],
        )

    def start_refill(self) -> None:
        """Start background refill: reader processes + GPU upload thread.

        Reader processes read and encode records on CPU (bypasses GIL).
        A single GPU upload thread drains the encoded queue and writes
        to GPU tensors.
        """
        self._stop_event.clear()
        self._mp_stop = mp.Event()

        # Queue for encoded records: (slot, inp, target, mask, range, gv, sw)
        self._encoded_queue: mp.Queue = mp.Queue(maxsize=self._num_workers * 64)

        # Start reader processes.
        self._processes: list[mp.Process] = []
        for i in range(self._num_workers):
            p = mp.Process(
                target=_reader_process,
                args=(
                    self._file_index,
                    self._total_records,
                    self._refill_queue,
                    self._encoded_queue,
                    self._mp_stop,
                ),
                name=f"refill-{i}",
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        # Start single GPU upload thread.
        self._upload_thread = threading.Thread(
            target=self._gpu_upload_worker,
            name="gpu-upload",
            daemon=True,
        )
        self._upload_thread.start()
        print(f"  Started {self._num_workers} refill processes + 1 GPU upload thread")

    def _gpu_upload_worker(self) -> None:
        """Single thread that uploads encoded records to GPU tensors."""
        while not self._stop_event.is_set():
            try:
                item = self._encoded_queue.get(timeout=0.1)
            except Exception:
                continue

            slot, inp, target, mask, rng_arr, gv, sw = item
            self.inputs[slot] = torch.from_numpy(inp)
            self.targets[slot] = torch.from_numpy(target)
            self.masks[slot] = torch.from_numpy(mask)
            self.ranges[slot] = torch.from_numpy(rng_arr)
            self.game_values[slot] = gv
            self.sample_weights[slot] = sw

            with self._refill_count_lock:
                self._refill_count += 1

    def refill_rate(self) -> float:
        """Return records/second refill rate since last call, and reset counter."""
        now = time.time()
        with self._refill_count_lock:
            count = self._refill_count
            self._refill_count = 0
        elapsed = now - self._last_refill_time
        self._last_refill_time = now
        return count / max(elapsed, 0.001)

    def stop(self) -> None:
        """Stop background refill processes and upload thread."""
        self._stop_event.set()
        if hasattr(self, "_mp_stop"):
            self._mp_stop.set()
        if hasattr(self, "_upload_thread"):
            self._upload_thread.join(timeout=5.0)
        for p in getattr(self, "_processes", []):
            p.terminate()
            p.join(timeout=2.0)
        self._processes = []


def _reader_process(
    file_index: list[tuple[Path, int]],
    total_records: int,
    slot_queue: Queue,
    encoded_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    """Subprocess: read records from disk, encode, put on queue.

    Runs in a separate process to bypass the GIL. Each process has its
    own file handle cache.
    """
    from collections import OrderedDict

    rng = np.random.default_rng()
    handles: OrderedDict = OrderedDict()
    max_handles = 128

    while not stop_event.is_set():
        try:
            slot = slot_queue.get(timeout=0.1)
        except Exception:
            continue

        record_idx = rng.integers(0, total_records)
        file_path, byte_offset = file_index[record_idx]

        # Read with LRU file handle cache.
        fh = handles.get(file_path)
        if fh is not None:
            handles.move_to_end(file_path)
        else:
            if len(handles) >= max_handles:
                _, old_fh = handles.popitem(last=False)
                old_fh.close()
            fh = open(file_path, "rb")  # noqa: SIM115
            handles[file_path] = fh

        fh.seek(byte_offset)
        buf = fh.read(RECORD_SIZE_RIVER)
        raw = np.frombuffer(buf, dtype=np.uint8).copy()

        # Encode on CPU (no GIL contention with other processes).
        inp, target, mask, rng_arr, gv, sw = _encode_raw_to_tensors(raw)

        try:
            encoded_queue.put((slot, inp, target, mask, rng_arr, float(gv), float(sw)), timeout=1.0)
        except Exception:
            continue

    # Clean up file handles.
    for fh in handles.values():
        fh.close()
