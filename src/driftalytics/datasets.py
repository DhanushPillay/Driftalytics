"""Real dataset loaders used by benchmarks and tests."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

import pandas as pd

DATASET_NAME = "cicids2017"
DOWNLOAD_URL = (
    "https://huggingface.co/datasets/rdpahalavan/CIC-IDS2017/resolve/main/"
    "Network-Flows/CICIDS_Flow.parquet"
)
SHA256 = "e64f6adfbcb7f96b018cf36c731be64ef4de3353ccd7340b190590734a0dda35"
EXPECTED_ROWS = 2_827_677
EXPECTED_ATTACKS = 556_555
CACHE_PATH = Path("data/raw/cicids2017/CICIDS_Flow.parquet")

FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
]
COLUMNS = ["timestamp", "label", "attack_label", *FEATURES]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(DOWNLOAD_URL, timeout=600) as response:
            with temporary.open("wb") as handle:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
        if _sha256(temporary) != SHA256:
            raise ValueError(f"checksum mismatch for {DATASET_NAME}")
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def load_cicids(path: str | Path | None = None, *, validate: bool = True) -> pd.DataFrame:
    """Load CICIDS2017 network flows as a timestamped binary classification frame.

    The parquet file is downloaded on demand and checked against the published
    SHA-256. Features are stored as ``float32`` to keep the 2.8M-row frame
    practical; ``label`` is 0 for benign flows and 1 for attacks.
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

    frame = pd.read_parquet(source)
    required = ["Timestamp", "attack_label", *FEATURES]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"dataset is missing required columns: {missing}")

    timestamps = pd.to_datetime(frame["Timestamp"], format="mixed", dayfirst=True, errors="coerce")
    invalid = int(timestamps.isna().sum())
    if invalid:
        raise ValueError(f"dataset contains {invalid} unparseable timestamps")

    attack_labels = frame["attack_label"].astype(str)
    result = pd.DataFrame(
        {
            "timestamp": timestamps,
            "label": (attack_labels != "BENIGN").astype("int8"),
            "attack_label": attack_labels.astype("category"),
        }
    )
    for feature in FEATURES:
        result[feature] = frame[feature].astype("float32")
    result = result.sort_values("timestamp").reset_index(drop=True)

    if validate:
        if len(result) != EXPECTED_ROWS:
            raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(result)}")
        attacks = int(result["label"].sum())
        if attacks != EXPECTED_ATTACKS:
            raise ValueError(f"expected {EXPECTED_ATTACKS} attacks, got {attacks}")
    return result
