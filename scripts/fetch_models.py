"""Download the two ONNX models and verify their SHA-256 digests.

Runs at image build time. A failed download or a digest mismatch fails the
build, so a container never starts with a model other than the benchmarked one.

Usage: python scripts/fetch_models.py <destination directory>
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import NamedTuple

CHUNK_BYTES = 1 << 20
TIMEOUT_S = 60


class Model(NamedTuple):
    filename: str
    url: str
    sha256: str


MODELS = (
    Model(
        "mobilenetv3_small.onnx",
        "https://huggingface.co/onnx-community/mobilenetv3_small_100.lamb_in1k/resolve/main/onnx/model.onnx",
        "d36eea8c3298d339c232581dff18beed6651eacd607d0a531b7bdb100f8dc713",
    ),
    Model(
        "midas_small.onnx",
        "https://github.com/isl-org/MiDaS/releases/download/v2_1/model-small.onnx",
        "2d8c6cb8f415229daf1eb041024208e2608c9f98e17c81cc7c6ecb449c56fd58",
    ),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(model: Model, dest_dir: Path) -> Path:
    """Download one model into dest_dir unless a verified copy is already there."""
    dest = dest_dir / model.filename
    if dest.exists() and sha256_of(dest) == model.sha256:
        return dest

    partial = dest.with_name(dest.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(model.url, headers={"User-Agent": "fetch-models"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response, partial.open("wb") as out:
        while chunk := response.read(CHUNK_BYTES):
            digest.update(chunk)
            out.write(chunk)

    if digest.hexdigest() != model.sha256:
        partial.unlink()
        raise SystemExit(
            f"{model.filename}: sha256 mismatch, expected {model.sha256}, got {digest.hexdigest()}"
        )
    partial.replace(dest)
    return dest


def main(argv: list[str]) -> None:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    dest_dir = Path(argv[1])
    dest_dir.mkdir(parents=True, exist_ok=True)
    for model in MODELS:
        path = fetch(model, dest_dir)
        print(f"{path} ok ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main(sys.argv)
