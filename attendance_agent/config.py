"""Settings and tiny CSV helpers shared by all modules."""
from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Optional

import pandas as pd

# ---- paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ATTENDANCE_CSV = DATA_DIR / "attendance.csv"
ENGAGEMENT_CSV = DATA_DIR / "engagement.csv"
SUGGESTIONS_CSV = DATA_DIR / "suggestions.csv"

ATTENDANCE_COLUMNS = ["person_id", "first_seen", "marked_present_at"]
ENGAGEMENT_COLUMNS = ["window_end", "face_score", "movement_score",
                      "audio_score", "engagement", "label"]
SUGGESTION_COLUMNS = ["created_at", "suggestion"]

# ---- video
FRAME_WIDTH = 640          # downscale frames to this width
TARGET_FPS = 10            # frames processed per second
PRESENT_SECONDS = 5.0      # continuous seconds before marking Present
MAX_GAP_SECONDS = 1.0      # a track survives this long without a detection
MAX_MATCH_DISTANCE = 90.0  # max centroid jump (pixels) to keep the same ID

# ---- engagement
WINDOW_SECONDS = 10        # engagement window length
W_FACE = 0.5
W_MOVE = 0.2
W_AUDIO = 0.3
HIGH_THRESHOLD = 66        # score >= 66 -> High
MEDIUM_THRESHOLD = 33      # score >= 33 -> Medium, else Low
MOVEMENT_FULL_SCALE = 10.0   # mean gray difference (0-255) that maps to 1.0
AUDIO_RMS_FULL_SCALE = 0.1   # microphone RMS that maps to 1.0

# ---- suggestion rules
DROP_WINDOWS = 3             # consecutive drops that trigger advice
LOW_ENGAGEMENT_WINDOWS = 3
HIGH_ENGAGEMENT_WINDOWS = 3
LOW_FACE_SCORE = 0.3
LOW_FACE_WINDOWS = 3
LOW_AUDIO_LEVEL = 0.15
LOW_AUDIO_MINUTES = 5
LOW_AUDIO_WINDOWS = int(LOW_AUDIO_MINUTES * 60 / WINDOW_SECONDS)
HISTORY_SIZE = LOW_AUDIO_WINDOWS + 5

FILE_LOCK = threading.Lock()


def ensure_data_dir() -> None:
    """Create the data folder if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def append_csv(path: Path, columns: list[str], row: dict) -> None:
    """Append one row, writing the header when the file is new or empty."""
    ensure_data_dir()
    with FILE_LOCK:
        new_file = (not path.exists()) or path.stat().st_size == 0
        with open(path, "a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            if new_file:
                writer.writeheader()
            writer.writerow({c: row.get(c, "") for c in columns})


def reset_csv(path: Path, columns: list[str]) -> None:
    """Empty a CSV file but keep its header."""
    ensure_data_dir()
    with FILE_LOCK:
        with open(path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(columns)


def read_csv_records(path: Path, tail: Optional[int] = None) -> list[dict]:
    """Read a CSV into a list of dicts (empty list on any problem)."""
    try:
        with FILE_LOCK:
            if not path.exists() or path.stat().st_size == 0:
                return []
            df = pd.read_csv(path)
        if tail:
            df = df.tail(tail)
        return json.loads(df.to_json(orient="records"))
    except Exception:
        return []
