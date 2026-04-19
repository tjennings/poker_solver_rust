"""Tests for the BoundaryNet training loop."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import torch

from cfvnet.config import TrainConfig
from cfvnet.constants import NUM_COMBOS
from cfvnet.train import _maybe_resume, train_boundary


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


def test_maybe_resume_picks_numerically_latest_checkpoint():
    """Checkpoint resume must sort numerically, not lexicographically.

    Alphabetic sort of `checkpoint_epoch{N}.pt` picks epoch 9 over epoch 200,
    which silently corrupts LR schedule and optimizer state past epoch 9.
    """
    from torch.optim import Adam
    from torch.optim.lr_scheduler import CosineAnnealingLR

    from cfvnet.model import BoundaryNet

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)

        # Save small checkpoints for each epoch in {1, 2, 9, 10, 11, 100, 200}.
        template = BoundaryNet(num_layers=1, hidden_size=8)
        t_opt = Adam(template.parameters(), lr=1e-3)
        t_sched = CosineAnnealingLR(t_opt, T_max=10)
        t_scaler = torch.amp.GradScaler(enabled=False)
        for epoch in [1, 2, 9, 10, 11, 100, 200]:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": template.state_dict(),
                    "optimizer_state_dict": t_opt.state_dict(),
                    "scheduler_state_dict": t_sched.state_dict(),
                    "scaler_state_dict": t_scaler.state_dict(),
                },
                output_dir / f"checkpoint_epoch{epoch}.pt",
            )

        # Resume should load epoch 200.
        model = BoundaryNet(num_layers=1, hidden_size=8)
        opt = Adam(model.parameters(), lr=1e-3)
        sched = CosineAnnealingLR(opt, T_max=10)
        scaler = torch.amp.GradScaler(enabled=False)
        epoch = _maybe_resume(model, opt, sched, scaler, output_dir)

    assert epoch == 200, f"expected resume from epoch 200, got {epoch}"
