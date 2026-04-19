"""ONNX export for BoundaryNet."""

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cfvnet.constants import INPUT_SIZE
from cfvnet.model import BoundaryNet


def export_onnx(model: BoundaryNet, path: Path) -> None:
    """Export a trained BoundaryNet to ONNX format.

    Sets model to eval mode, exports with dynamic batch axis,
    and verifies the exported model produces matching outputs.

    Args:
        model: Trained BoundaryNet model.
        path: Output path for .onnx file.
    """
    model.eval()
    dummy = torch.zeros(1, INPUT_SIZE)

    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )

    _verify_export(model, path)


def _verify_export(model: BoundaryNet, path: Path) -> None:
    """Verify ONNX output matches PyTorch within tolerance.

    Raises:
        AssertionError: If outputs diverge beyond tolerance.
    """
    model.eval()
    x = torch.randn(4, INPUT_SIZE)

    with torch.no_grad():
        pytorch_out = model(x).numpy()

    session = ort.InferenceSession(str(path))
    onnx_out = session.run(None, {"input": x.numpy()})[0]

    np.testing.assert_allclose(
        pytorch_out, onnx_out, rtol=1e-4, atol=1e-4,
        err_msg="ONNX output does not match PyTorch",
    )
    print(f"  ONNX export verified: {path} ({path.stat().st_size / 1024:.0f} KB)")
