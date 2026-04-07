"""Tests for the BoundaryNet training loop."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from cfvnet.config import TrainConfig
from cfvnet.constants import NUM_COMBOS
from cfvnet.train import train_boundary


def _write_test_data(path: Path, n: int = 32) -> None:
    """Write synthetic binary training records matching the Rust format.

    Each record has a 5-card board, pot=100, stack=50, and sparse
    ranges/CFVs so the network can learn a simple mapping.

    Args:
        path: File path to write.
        n: Number of records to generate.
    """
    with open(path, "wb") as f:
        for i in range(n):
            board = [0, 4, 8, 12, 16]
            f.write(struct.pack("B", len(board)))
            f.write(bytes(board))
            f.write(struct.pack("<f", 100.0))
            f.write(struct.pack("<f", 50.0))
            f.write(struct.pack("B", i % 2))
            f.write(struct.pack("<f", 0.1 * i))
            oop = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                oop[j] = 0.1
            f.write(oop.tobytes())
            ip = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                ip[j] = 0.1
            f.write(ip.tobytes())
            cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                cfvs[j] = (i + j) * 0.01
            f.write(cfvs.tobytes())
            mask = np.ones(NUM_COMBOS, dtype=np.uint8)
            f.write(mask.tobytes())


def test_training_reduces_loss():
    """Verify that 50 epochs on tiny data drives loss below 0.1."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = Path(tmpdir) / "train.bin"
        _write_test_data(data_path, n=32)

        config = TrainConfig(
            hidden_layers=2, hidden_size=64,
            batch_size=16, epochs=50,
            learning_rate=0.001, lr_min=0.001,
            huber_delta=1.0, aux_loss_weight=0.0,
            validation_split=0.0, grad_clip_norm=1.0,
        )

        result = train_boundary(
            data_path=data_path,
            config=config,
            output_dir=None,
            device=torch.device("cpu"),
            num_workers=0,
        )

    assert result.final_train_loss < 0.1, f"expected loss < 0.1, got {result.final_train_loss}"
