"""Write a copy of the published MiDaS graph that accepts any batch size.

The published file declares a batch dimension of 1. The copy differs only in
its declared shapes: dimension 0 of every graph input and output becomes
symbolic. Nodes and weights are untouched, so both files compute the same
function.

Runs at image build time, after the download. A copy that fails the ONNX
checker fails the build.

Usage: python scripts/patch_midas.py <published.onnx> <batched.onnx>
"""

from __future__ import annotations

import sys
from pathlib import Path

import onnx

BATCH_DIMENSION_NAME = "batch"


def main(argv: list[str]) -> None:
    if len(argv) != 3:
        raise SystemExit(__doc__)
    source, destination = Path(argv[1]), Path(argv[2])

    model = onnx.load(source)
    graph = model.graph
    for value in (*graph.input, *graph.output):
        value.type.tensor_type.shape.dim[0].dim_param = BATCH_DIMENSION_NAME
    # Shapes recorded for intermediate tensors were inferred for the fixed batch size.
    del graph.value_info[:]

    onnx.checker.check_model(model)
    onnx.save(model, destination)
    print(f"{destination} ok ({destination.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv)
