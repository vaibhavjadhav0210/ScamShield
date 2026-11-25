"""CSV-backed check history logger for ScamShield.

OOP Concept Demonstrated:
    - Encapsulation: File path and CSV schema management are encapsulated in
      `HistoryLog`, exposing clean `record()` and `load()` methods.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from scamshield.models.message import DetectionResult, Message

DEFAULT_HISTORY_PATH = Path("data") / "history.csv"
HISTORY_COLUMNS: list[str] = [
    "timestamp",
    "channel",
    "verdict",
    "scam_type",
    "risk_score",
    "detector",
    "preview",
]


class HistoryLog:
    """Persists ScamShield check results to a local CSV file."""

    def __init__(self, csv_path: str | Path = DEFAULT_HISTORY_PATH) -> None:
        self._csv_path: Path = Path(csv_path)

    @property
    def csv_path(self) -> Path:
        """Return the path to the history CSV file."""
        return self._csv_path

    def record(self, message: Message, result: DetectionResult) -> None:
        """Append a single detection check to the CSV history."""
        self._csv_path.parent.mkdir(parents=True, exist_ok=True)
        preview = message.text.replace("\n", " ").strip()[:80]
        row = pd.DataFrame(
            [
                {
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    "channel": message.channel,
                    "verdict": result.verdict,
                    "scam_type": result.scam_type,
                    "risk_score": result.risk_score,
                    "detector": result.detector_name,
                    "preview": preview,
                }
            ]
        )
        write_header = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
        row.to_csv(self._csv_path, mode="a", header=write_header, index=False)

    def load(self, limit: int = 20) -> pd.DataFrame:
        """Load the most recent `limit` history entries as a DataFrame."""
        if not self._csv_path.exists() or self._csv_path.stat().st_size == 0:
            return pd.DataFrame(columns=HISTORY_COLUMNS)
        df = pd.read_csv(self._csv_path)
        return df.tail(limit).iloc[::-1].reset_index(drop=True)

    def clear(self) -> None:
        """Remove the history CSV file if it exists."""
        if self._csv_path.exists():
            self._csv_path.unlink()
