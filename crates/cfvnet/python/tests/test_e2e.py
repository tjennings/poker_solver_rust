"""End-to-end test: create data -> train -> export -> infer with onnxruntime."""

import struct
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cfvnet.config import TrainConfig
from cfvnet.constants import INPUT_SIZE, NUM_COMBOS, OUTPUT_SIZE
from cfvnet.export import export_onnx
from cfvnet.model import BoundaryNet
from cfvnet.train import train_boundary


def _write_test_data(path: Path, n: int = 64) -> None:
    """Write n test records in Rust binary format."""
    with open(path, "wb") as f:
        for i in range(n):
            board = [0, 4, 8, 12, 16]
            f.write(struct.pack("B", len(board)))
            f.write(bytes(board))
            f.write(struct.pack("<f", 100.0))
            f.write(struct.pack("<f", 50.0))
            f.write(struct.pack("B", i % 2))
            f.write(struct.pack("<f", 0.0))
            oop = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                oop[j] = 0.1
            f.write(oop.tobytes())
            f.write(oop.tobytes())  # ip same as oop
            cfvs = np.zeros(NUM_COMBOS, dtype=np.float32)
            for j in range(10):
                cfvs[j] = 0.01 * (i + j)
            f.write(cfvs.tobytes())
            mask = np.ones(NUM_COMBOS, dtype=np.uint8)
            f.write(mask.tobytes())


def test_full_pipeline():
    """Train -> export -> ONNX inference end-to-end."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_path = tmpdir / "data.bin"
        output_dir = tmpdir / "model"
        _write_test_data(data_path)

        # Train.
        config = TrainConfig(
            hidden_layers=2, hidden_size=32,
            batch_size=16, epochs=20,
            learning_rate=0.001, lr_min=0.001,
            huber_delta=1.0, aux_loss_weight=0.0,
            validation_split=0.0, grad_clip_norm=1.0,
            checkpoint_every_n_epochs=20,
        )
        result = train_boundary(data_path, config, output_dir, torch.device("cpu"))
        assert result.final_train_loss < 1.0

        # Load trained model and export.
        model = BoundaryNet(config.hidden_layers, config.hidden_size)
        ckpt = torch.load(output_dir / "checkpoint_epoch20.pt", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])

        onnx_path = output_dir / "model.onnx"
        export_onnx(model, onnx_path)

        # Infer with onnxruntime.
        session = ort.InferenceSession(str(onnx_path))
        x = np.random.randn(4, INPUT_SIZE).astype(np.float32)
        out = session.run(None, {"input": x})[0]
        assert out.shape == (4, OUTPUT_SIZE)
