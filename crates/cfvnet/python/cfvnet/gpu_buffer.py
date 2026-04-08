"""GPU ring buffer for training data.

Maintains a large buffer of training records on GPU memory. Background
threads continuously read from disk and replace consumed records. The
training loop samples batches directly from GPU — no per-batch I/O or
CPU-GPU transfer.
"""

from __future__ import annotations

import threading
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
        self._refill_queue: Queue[int] = Queue()  # Slot indices to refill.
        self._workers: list[threading.Thread] = []

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
        self.game_values[slot] = gv
        self.sample_weights[slot] = sw

    def _read_cached(self, path: Path, offset: int) -> np.ndarray:
        """Read record bytes using thread-local file handle cache."""
        cache = getattr(self._local, "handles", None)
        if cache is None:
            cache = {}
            self._local.handles = cache

        fh = cache.get(path)
        if fh is None:
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
        """Start background refill threads."""
        self._stop_event.clear()
        for i in range(self._num_workers):
            t = threading.Thread(
                target=self._refill_worker,
                name=f"refill-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)
        print(f"  Started {self._num_workers} refill workers")

    def _refill_worker(self) -> None:
        """Background worker that replaces consumed buffer slots."""
        rng = np.random.default_rng()
        while not self._stop_event.is_set():
            try:
                slot = self._refill_queue.get(timeout=0.1)
            except Exception:
                continue

            # Pick a random record from the full dataset.
            record_idx = rng.integers(0, self._total_records)
            self._fill_slot(slot, record_idx)

    def stop(self) -> None:
        """Stop background refill threads."""
        self._stop_event.set()
        for t in self._workers:
            t.join(timeout=5.0)
        self._workers.clear()

        # Close cached file handles.
        cache = getattr(self._local, "handles", {})
        for fh in cache.values():
            fh.close()
