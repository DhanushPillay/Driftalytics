"""Real dataset loaders used by benchmarks and tests."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

DATASET_NAME = "insects_incremental_abrupt_balanced"
DOWNLOAD_URL = (
    "https://drive.google.com/uc?export=download&id=1-J5WIBN8_F_tomdcrOaiLCxk9nzxtFsf&confirm=t"
)
SHA256 = "c1cd19d429844349009311fa372ee700d5bf0e21545fb61d0d3fba724123188b"
EXPECTED_ROWS = 79_986
EXPECTED_CLASSES = (2, 3, 4, 5, 11, 12)
FEATURES = [f"f{index}" for index in range(1, 34)]
COLUMNS = [*FEATURES, "class"]
CACHE_PATH = Path("data/raw/insects/insects_incremental_abrupt_balanced.csv")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(DOWNLOAD_URL, timeout=120) as response:
        with temporary.open("wb") as handle:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                handle.write(chunk)
    if _sha256(temporary) != SHA256:
        temporary.unlink(missing_ok=True)
        raise ValueError(f"checksum mismatch for {DATASET_NAME}")
    temporary.replace(destination)


def load_insects(path: str | Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load the Insects incremental-abrupt stream.

    The CSV has no header and contains 33 numeric features followed by the
    integer class label. When ``path`` is omitted, the dataset is downloaded
    to ``data/raw/insects`` and checked against the published SHA-256.
    """
    if path is None:
        source = CACHE_PATH
        if not source.exists():
            _download(source)
        if _sha256(source) != SHA256:
            raise ValueError(f"checksum mismatch for {source}; delete it and retry")
    else:
        source = Path(path)

    if not source.exists():
        raise FileNotFoundError(f"dataset not found: {source}")

    frame = pd.read_csv(source, header=None, names=COLUMNS)
    if validate:
        if frame.shape != (EXPECTED_ROWS, len(COLUMNS)):
            raise ValueError(
                f"expected {EXPECTED_ROWS} rows and {len(COLUMNS)} columns, got {frame.shape}"
            )
        classes = tuple(sorted(int(value) for value in frame["class"].unique()))
        if classes != EXPECTED_CLASSES:
            raise ValueError(f"unexpected classes: {classes}")
    return frame
