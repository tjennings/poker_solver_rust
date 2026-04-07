# tests/test_dataset.py
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from cfvnet.constants import INPUT_SIZE, NUM_COMBOS, OUTPUT_SIZE
from cfvnet.data import BoundaryDataset


def _write_test_file(path: Path, n: int = 4) -> None:
    """Write n test records to a binary file."""
    with open(path, "wb") as f:
        for i in range(n):
            board = [0, 4, 8, 12, 16]
            f.write(struct.pack("B", len(board)))
            f.write(bytes(board))
            f.write(struct.pack("<f", 100.0))  # pot
            f.write(struct.pack("<f", 50.0 + i * 10))  # stack
            f.write(struct.pack("B", i % 2))  # player
            f.write(struct.pack("<f", 0.05 * i))  # game_value
            oop = np.zeros(NUM_COMBOS, dtype=np.float32)
            oop[0] = 0.5
            f.write(oop.tobytes())
            ip = np.zeros(NUM_COMBOS, dtype=np.float32)
            ip[0] = 0.5
            f.write(ip.tobytes())
            cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
            cfvs[0] = 0.1 * (i + 1)
            f.write(cfvs.tobytes())
            mask = np.ones(NUM_COMBOS, dtype=np.uint8)
            f.write(mask.tobytes())


def test_dataset_from_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.bin"
        _write_test_file(path, n=4)
        ds = BoundaryDataset.from_path(path)

    assert len(ds) == 4
    inp, target, mask, rng, gv, sw = ds[0]
    assert inp.shape == (INPUT_SIZE,)
    assert target.shape == (OUTPUT_SIZE,)
    assert mask.shape == (OUTPUT_SIZE,)
    assert rng.shape == (OUTPUT_SIZE,)
    assert isinstance(gv, float) or gv.shape == ()
    assert isinstance(sw, float) or sw.shape == ()


def test_dataset_from_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_test_file(Path(tmpdir) / "a.bin", n=3)
        _write_test_file(Path(tmpdir) / "b.bin", n=2)
        ds = BoundaryDataset.from_path(Path(tmpdir))

    assert len(ds) == 5


def test_dataset_works_with_dataloader():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.bin"
        _write_test_file(path, n=8)
        ds = BoundaryDataset.from_path(path)

    loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    inp, target, mask, rng, gv, sw = batch
    assert inp.shape == (4, INPUT_SIZE)
    assert target.shape == (4, OUTPUT_SIZE)
    assert gv.shape == (4,)
    assert sw.shape == (4,)
