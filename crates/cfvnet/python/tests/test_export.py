"""Tests for ONNX export functionality."""

import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cfvnet.constants import INPUT_SIZE, OUTPUT_SIZE
from cfvnet.export import export_onnx
from cfvnet.model import BoundaryNet


def test_export_creates_onnx_file():
    """Verify export_onnx creates a non-empty .onnx file."""
    model = BoundaryNet(num_layers=2, hidden_size=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        export_onnx(model, path)
        assert path.exists()
        assert path.stat().st_size > 0


def test_onnx_output_matches_pytorch():
    """Verify ONNX model output matches PyTorch within tolerance."""
    model = BoundaryNet(num_layers=2, hidden_size=64)
    model.eval()

    x = torch.randn(4, INPUT_SIZE)
    with torch.no_grad():
        pytorch_out = model(x).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        export_onnx(model, path)

        session = ort.InferenceSession(str(path))
        onnx_out = session.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(pytorch_out, onnx_out, rtol=1e-5, atol=1e-5)


def test_onnx_supports_dynamic_batch():
    """Verify exported ONNX model supports variable batch sizes."""
    model = BoundaryNet(num_layers=2, hidden_size=64)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "model.onnx"
        export_onnx(model, path)

        session = ort.InferenceSession(str(path))

        input_meta = session.get_inputs()[0]
        batch_dim = input_meta.shape[0]
        assert not isinstance(batch_dim, int), (
            f"ONNX batch dim should be dynamic, got literal {batch_dim!r}"
        )

        for batch_size in [1, 7, 32]:
            x = np.random.randn(batch_size, INPUT_SIZE).astype(np.float32)
            out = session.run(None, {"input": x})[0]
            assert out.shape == (batch_size, OUTPUT_SIZE)
