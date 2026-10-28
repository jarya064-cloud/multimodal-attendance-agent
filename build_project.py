"""Builds the 'Multimodal Attendance & Engagement Agent' project and zips it.

Usage: python build_project.py
Creates ./attendance_agent/ and ./attendance_agent.zip
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = "attendance_agent"

FILES: dict[str, str] = {}

# ---------------------------------------------------------------- requirements
FILES["attendance_agent/requirements.txt"] = r'''opencv-python>=4.9
numpy>=1.26
pandas>=2.1
flask>=3.0
sounddevice>=0.4.6
'''

# ---------------------------------------------------------------- config.py
FILES["attendance_agent/config.py"] = r'''"""Settings and tiny CSV helpers shared by all modules."""
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
'''

# ---------------------------------------------------------------- vision.py
FILES["attendance_agent/vision.py"] = r'''"""Face detection (Haar cascade), centroid tracking and attendance logic."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config

Box = tuple[int, int, int, int]  # x, y, w, h


def fmt_time(ts: float) -> str:
    """Format a unix timestamp for the CSV files."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Downscale a frame to config.FRAME_WIDTH pixels wide."""
    h, w = frame.shape[:2]
    if w <= config.FRAME_WIDTH:
        return frame
    scale = config.FRAME_WIDTH / float(w)
    return cv2.resize(frame, (config.FRAME_WIDTH, int(h * scale)),
                      interpolation=cv2.INTER_AREA)


class FaceDetector:
    """OpenCV built-in Haar cascade frontal face detector."""

    def __init__(self) -> None:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError("Could not load Haar cascade: " + path)

    def detect(self, frame: np.ndarray) -> list[Box]:
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=5, minSize=(40, 40))
        return [tuple(int(v) for v in f) for f in faces]  # type: ignore


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    box: Box
    first_seen: float
    last_seen: float


class CentroidTracker:
    """Assign IDs to faces by nearest centroid."""

    def __init__(self, max_distance: float, max_gap: float, start_id: int = 1):
        self.max_distance = max_distance
        self.max_gap = max_gap
        self.next_id = start_id
        self.tracks: dict[int, Track] = {}

    def update(self, boxes: list[Box], now: float) -> list[Track]:
        """Match boxes to tracks; return tracks seen in this frame."""
        cents = [(x + w / 2.0, y + h / 2.0) for x, y, w, h in boxes]
        pairs = sorted(
            (math.hypot(t.centroid[0] - c[0], t.centroid[1] - c[1]), tid, bi)
            for tid, t in self.tracks.items() for bi, c in enumerate(cents))
        used_t: set[int] = set()
        used_b: set[int] = set()
        active: list[Track] = []
        for dist, tid, bi in pairs:
            if dist > self.max_distance:
                break
            if tid in used_t or bi in used_b:
                continue
            t = self.tracks[tid]
            t.centroid, t.box, t.last_seen = cents[bi], boxes[bi], now
            used_t.add(tid)
            used_b.add(bi)
            active.append(t)
        for bi, box in enumerate(boxes):
            if bi in used_b:
                continue
            t = Track(self.next_id, cents[bi], box, now, now)
            self.tracks[t.track_id] = t
            self.next_id += 1
            active.append(t)
        for tid in [k for k, t in self.tracks.items()
                    if now - t.last_seen > self.max_gap]:
            del self.tracks[tid]
        return active


class AttendanceLogger:
    """Mark a person Present after continuous presence; log to CSV."""

    def __init__(self, present_seconds: float) -> None:
        self.present_seconds = present_seconds
        self.marked: set[int] = set()

    def max_existing_id(self) -> int:
        ids = [int(r["person_id"])
               for r in config.read_csv_records(config.ATTENDANCE_CSV)
               if "person_id" in r]
        return max(ids) if ids else 0

    def check(self, tracks: list[Track], now: float) -> None:
        for t in tracks:
            if t.track_id in self.marked:
                continue
            if now - t.first_seen >= self.present_seconds:
                self.marked.add(t.track_id)
                config.append_csv(config.ATTENDANCE_CSV,
                                  config.ATTENDANCE_COLUMNS, {
                                      "person_id": t.track_id,
                                      "first_seen": fmt_time(t.first_seen),
                                      "marked_present_at": fmt_time(now)})

    def reset(self) -> None:
        self.marked.clear()


class VisionProcessor:
    """Detector + tracker + attendance, with box drawing."""

    def __init__(self, present_seconds: float, use_detector: bool = True):
        self.present_seconds = present_seconds
        self.detector: Optional[FaceDetector] = (
            FaceDetector() if use_detector else None)
        self.attendance = AttendanceLogger(present_seconds)
        self.tracker = self._new_tracker()

    def _new_tracker(self) -> CentroidTracker:
        return CentroidTracker(config.MAX_MATCH_DISTANCE,
                               config.MAX_GAP_SECONDS,
                               self.attendance.max_existing_id() + 1)

    def process(self, frame: np.ndarray,
                now: Optional[float] = None) -> tuple[np.ndarray, bool]:
        """Detect faces in a frame; return (annotated frame, face found)."""
        boxes = self.detector.detect(frame) if self.detector else []
        return self.process_boxes(frame, boxes, now), len(boxes) > 0

    def process_boxes(self, frame: np.ndarray, boxes: list[Box],
                      now: Optional[float] = None) -> np.ndarray:
        """Track given boxes, update attendance, draw them."""
        now = time.time() if now is None else now
        tracks = self.tracker.update(boxes, now)
        self.attendance.check(tracks, now)
        for t in tracks:
            x, y, w, h = t.box
            present = t.track_id in self.attendance.marked
            color = (0, 200, 0) if present else (0, 165, 255)
            text = "ID %d %s" % (t.track_id, "PRESENT" if present
                                 else "%.1fs" % (now - t.first_seen))
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return frame

    def reset(self) -> None:
        """Forget all tracks and attendance state (IDs restart at 1)."""
        self.attendance.reset()
        self.tracker = self._new_tracker()
'''

# ---------------------------------------------------------------- audio.py
FILES["attendance_agent/audio.py"] = r'''"""Microphone loudness (RMS) with a safe fallback to 0."""
from __future__ import annotations

import threading
from typing import Any

import numpy as np

import config


class AudioMeter:
    """Collects RMS loudness in the background; never raises."""

    def __init__(self, samplerate: int = 16000) -> None:
        self.samplerate = samplerate
        self.available = False
        self.error = ""
        self._stream: Any = None
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def start(self) -> bool:
        """Try to open the default microphone. Returns True on success."""
        try:
            import sounddevice as sd  # imported here so failures are caught
            self._stream = sd.InputStream(channels=1,
                                          samplerate=self.samplerate,
                                          blocksize=1024,
                                          callback=self._callback)
            self._stream.start()
            self.available = True
        except Exception as exc:
            self.available = False
            self.error = "Microphone unavailable: %s" % exc
        return self.available

    def _callback(self, indata: np.ndarray, frames: int,
                  time_info: Any, status: Any) -> None:
        try:
            rms = float(np.sqrt(np.mean(np.square(indata))))
            with self._lock:
                self._sum += rms
                self._count += 1
        except Exception:
            pass

    def read_level(self) -> float:
        """Mean RMS since the last call, normalized to 0-1 (0 if no mic)."""
        if not self.available:
            return 0.0
        with self._lock:
            mean = self._sum / self._count if self._count else 0.0
            self._sum, self._count = 0.0, 0
        return float(min(1.0, max(0.0, mean / config.AUDIO_RMS_FULL_SCALE)))

    def stop(self) -> None:
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        self.available = False
'''

# ---------------------------------------------------------------- engagement.py
FILES["attendance_agent/engagement.py"] = r'''"""Fuse face, movement and audio scores into an engagement score."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import cv2
import numpy as np

import config


class MotionMeter:
    """Mean frame difference between consecutive frames, scaled to 0-1."""

    def __init__(self) -> None:
        self.prev: Optional[np.ndarray] = None

    def update(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.GaussianBlur(cv2.resize(gray, (160, 120)), (5, 5), 0)
        if self.prev is None:
            self.prev = small
            return 0.0
        diff = float(cv2.absdiff(small, self.prev).mean())
        self.prev = small
        return min(1.0, diff / config.MOVEMENT_FULL_SCALE)


def compute_engagement(face: float, movement: float, audio: float) -> float:
    """Weighted sum scaled to 0-100."""
    score = 100.0 * (config.W_FACE * face + config.W_MOVE * movement
                     + config.W_AUDIO * audio)
    return round(max(0.0, min(100.0, score)), 1)


def label_for(score: float) -> str:
    if score >= config.HIGH_THRESHOLD:
        return "High"
    if score >= config.MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


class EngagementWindow:
    """Accumulates per-frame values for one time window."""

    def __init__(self, start: float, seconds: float) -> None:
        self.start = start
        self.seconds = seconds
        self.frames = 0
        self.face_frames = 0
        self.move_sum = 0.0

    def add(self, face_found: bool, movement: float) -> None:
        self.frames += 1
        self.face_frames += 1 if face_found else 0
        self.move_sum += movement

    def is_complete(self, now: float) -> bool:
        return now - self.start >= self.seconds

    def finish(self, audio_score: float) -> dict:
        """Return the finished window as a record dict."""
        n = max(1, self.frames)
        face = self.face_frames / n
        move = self.move_sum / n
        audio = max(0.0, min(1.0, audio_score))
        score = compute_engagement(face, move, audio)
        return {"window_end": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "face_score": round(face, 3),
                "movement_score": round(move, 3),
                "audio_score": round(audio, 3),
                "engagement": score,
                "label": label_for(score)}


def save_window(record: dict) -> None:
    config.append_csv(config.ENGAGEMENT_CSV, config.ENGAGEMENT_COLUMNS, record)
'''

# ---------------------------------------------------------------- agent.py
FILES["attendance_agent/agent.py"] = r'''"""Simple rule-based teaching suggestions (advice only)."""
from __future__ import annotations

from datetime import datetime

import config


def evaluate(history: list[dict]) -> list[str]:
    """Look at recent engagement windows and return suggestions."""
    if not history:
        return []
    scores = [float(h["engagement"]) for h in history]
    labels = [h["label"] for h in history]
    faces = [float(h["face_score"]) for h in history]
    audios = [float(h["audio_score"]) for h in history]
    tips: list[str] = []

    n = config.DROP_WINDOWS
    if len(scores) >= n + 1 and all(scores[-i] < scores[-i - 1]
                                    for i in range(1, n + 1)):
        tips.append("Engagement dropped for %d windows: try an interactive "
                    "activity" % n)

    n = config.LOW_AUDIO_WINDOWS
    if len(audios) >= n and all(a < config.LOW_AUDIO_LEVEL
                                for a in audios[-n:]):
        tips.append("Low audio for %d minutes: ask a question"
                    % config.LOW_AUDIO_MINUTES)

    n = config.LOW_FACE_WINDOWS
    if len(faces) >= n and all(f < config.LOW_FACE_SCORE for f in faces[-n:]):
        tips.append("Few faces visible for %d windows: check the camera "
                    "angle or ask learners to face the screen" % n)

    n = config.LOW_ENGAGEMENT_WINDOWS
    if len(labels) >= n and all(lb == "Low" for lb in labels[-n:]):
        tips.append("Engagement low for %d windows: take a short break or "
                    "change the topic" % n)

    n = config.HIGH_ENGAGEMENT_WINDOWS
    if len(labels) >= n and all(lb == "High" for lb in labels[-n:]):
        tips.append("Engagement high: keep the current pace")

    if not tips:
        tips.append("Engagement steady: no change needed")
    return tips


class SuggestionAgent:
    """Runs the rules and saves suggestions when they change."""

    def __init__(self) -> None:
        self.last_tips: list[str] = []

    def update(self, history: list[dict]) -> list[str]:
        tips = evaluate(history)
        if tips and tips != self.last_tips:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for tip in tips:
                config.append_csv(config.SUGGESTIONS_CSV,
                                  config.SUGGESTION_COLUMNS,
                                  {"created_at": now, "suggestion": tip})
            self.last_tips = tips
        return tips

    def reset(self) -> None:
        self.last_tips = []
'''

# ---------------------------------------------------------------- app.py
FILES["attendance_agent/app.py"] = r'''"""Flask dashboard + background worker for the attendance agent.

Run:  python app.py            (webcam)
      python app.py --video path/to/file.mp4
      python app.py --demo     (synthetic data, no camera or mic needed)
"""
from __future__ import annotations

import argparse
import math
import os
import random
import threading
import time
from typing import Optional

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

import config
from agent import SuggestionAgent
from audio import AudioMeter
from engagement import EngagementWindow, MotionMeter, save_window
from vision import Box, VisionProcessor, resize_frame

app = Flask(__name__)


class SharedState:
    """Data shared between the worker thread and Flask routes."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jpeg: Optional[bytes] = None
        self.engagement: Optional[dict] = None
        self.error = ""
        self.mode = "camera"
        self.audio_note = ""
        self.stop = threading.Event()
        self.reset_requested = threading.Event()
        self.reset_done = threading.Event()


state = SharedState()


# ------------------------------------------------------------ frame sources
def open_source(video: Optional[str]) -> tuple[Optional[cv2.VideoCapture], bool, str]:
    """Open webcam or video file. Returns (capture, is_file, error)."""
    try:
        if video:
            cap = cv2.VideoCapture(video)
            is_file = True
        else:
            is_file = False
            cap = (cv2.VideoCapture(0, cv2.CAP_DSHOW) if os.name == "nt"
                   else cv2.VideoCapture(0))
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            cap.release()
            return None, is_file, ("Could not open video file." if video else
                                   "Camera not available. Try --demo.")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, is_file, ""
    except Exception as exc:
        return None, False, "Camera error: %s" % exc


def placeholder_frame(text: str) -> np.ndarray:
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(frame, text[:60], (20, 180), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (200, 200, 200), 1)
    return frame


def demo_frame(t: float, rng: random.Random) -> tuple[np.ndarray, list[Box]]:
    """Synthetic frame with up to 3 moving fake faces."""
    frame = np.full((360, 640, 3), 40, dtype=np.uint8)
    cv2.putText(frame, "DEMO MODE (synthetic data)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    p_face = 0.55 + 0.4 * math.sin(t / 45.0)
    boxes: list[Box] = []
    for i in range(3):
        if t < i * 8 or rng.random() > p_face:  # people arrive one by one
            continue
        cx = int(120 + i * 200 + 25 * math.sin(t / 3.0 + i))
        cy = int(190 + 15 * math.cos(t / 2.0 + i))
        r = 48
        cv2.circle(frame, (cx, cy), r, (150, 180, 220), -1)
        cv2.circle(frame, (cx - 16, cy - 10), 5, (40, 40, 40), -1)
        cv2.circle(frame, (cx + 16, cy - 10), 5, (40, 40, 40), -1)
        boxes.append((cx - r, cy - r, 2 * r, 2 * r))
    return frame, boxes


def demo_audio(t: float, rng: random.Random) -> float:
    return float(min(1.0, max(0.0, 0.5 + 0.4 * math.sin(t / 30.0 + 1.0)
                              + rng.gauss(0, 0.05))))


# ------------------------------------------------------------ worker thread
def do_reset(vision: VisionProcessor, agent: SuggestionAgent) -> None:
    """Wipe CSV files and in-memory state (called inside the worker)."""
    config.reset_csv(config.ATTENDANCE_CSV, config.ATTENDANCE_COLUMNS)
    config.reset_csv(config.ENGAGEMENT_CSV, config.ENGAGEMENT_COLUMNS)
    config.reset_csv(config.SUGGESTIONS_CSV, config.SUGGESTION_COLUMNS)
    vision.reset()
    agent.reset()
    with state.lock:
        state.engagement = None


def worker(args: argparse.Namespace) -> None:
    """Capture, detect, track, score, advise. Runs until state.stop is set."""
    demo = bool(args.demo)
    cap: Optional[cv2.VideoCapture] = None
    meter = AudioMeter()
    try:
        vision = VisionProcessor(args.present_seconds, use_detector=not demo)
        motion = MotionMeter()
        agent = SuggestionAgent()
        history: list[dict] = []
        rng = random.Random()
        is_file = False

        if demo:
            state.mode = "demo"
        else:
            state.mode = "video" if args.video else "camera"
            cap, is_file, state.error = open_source(args.video)
            if not meter.start():
                state.audio_note = meter.error

        period = 1.0 / config.TARGET_FPS
        t0 = time.time()
        next_t = t0
        window = EngagementWindow(t0, config.WINDOW_SECONDS)

        while not state.stop.is_set():
            if state.reset_requested.is_set():
                do_reset(vision, agent)
                history.clear()
                window = EngagementWindow(time.time(), config.WINDOW_SECONDS)
                state.reset_requested.clear()
                state.reset_done.set()

            now = time.time()
            if now < next_t:
                if cap is not None and not is_file:
                    cap.grab()  # drop webcam frames to stay near target FPS
                else:
                    time.sleep(min(next_t - now, 0.05))
                continue
            next_t = max(next_t + period, now)

            try:
                face_found = False
                if demo:
                    frame, boxes = demo_frame(now - t0, rng)
                    movement = motion.update(frame)
                    frame = vision.process_boxes(frame, boxes, now)
                    face_found = len(boxes) > 0
                elif cap is None:
                    frame, movement = placeholder_frame(state.error), 0.0
                else:
                    ok, raw = cap.read()
                    if not ok and is_file:  # loop video files
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, raw = cap.read()
                    if not ok:
                        state.error = "Could not read from camera/video."
                        time.sleep(0.2)
                        continue
                    small = resize_frame(raw)
                    movement = motion.update(small)
                    frame, face_found = vision.process(small, now)

                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with state.lock:
                        state.jpeg = buf.tobytes()

                window.add(face_found, movement)
                if window.is_complete(now):
                    audio = (demo_audio(now - t0, rng) if demo
                             else meter.read_level())
                    record = window.finish(audio)
                    save_window(record)
                    history.append(record)
                    del history[:-config.HISTORY_SIZE]
                    agent.update(history)
                    with state.lock:
                        state.engagement = record
                    window = EngagementWindow(now, config.WINDOW_SECONDS)
            except Exception as exc:  # keep the worker alive
                state.error = "Processing error: %s" % exc
                time.sleep(0.2)
    except Exception as exc:
        state.error = "Worker stopped: %s" % exc
    finally:
        meter.stop()
        if cap is not None:
            cap.release()


# ------------------------------------------------------------ routes
@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/video_feed")
def video_feed() -> Response:
    def generate():
        while not state.stop.is_set():
            with state.lock:
                jpeg = state.jpeg
            if jpeg:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg + b"\r\n")
            time.sleep(0.1)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    with state.lock:
        engagement = state.engagement
    return jsonify({
        "mode": state.mode,
        "error": state.error,
        "audio_note": state.audio_note,
        "engagement": engagement,
        "attendance": config.read_csv_records(config.ATTENDANCE_CSV),
        "suggestions": config.read_csv_records(config.SUGGESTIONS_CSV, tail=6),
    })


@app.route("/api/delete", methods=["POST"])
def api_delete():
    state.reset_done.clear()
    state.reset_requested.set()
    done = state.reset_done.wait(timeout=3.0)
    return jsonify({"deleted": done})


# ------------------------------------------------------------ main
def main() -> None:
    parser = argparse.ArgumentParser(description="Attendance & Engagement Agent")
    parser.add_argument("--demo", action="store_true",
                        help="use synthetic data (no camera or mic)")
    parser.add_argument("--video", default=None, help="path to a video file")
    parser.add_argument("--present-seconds", type=float,
                        default=config.PRESENT_SECONDS,
                        help="seconds a face must be seen to be marked Present")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    config.ensure_data_dir()
    thread = threading.Thread(target=worker, args=(args,), daemon=True)
    thread.start()
    print("Open http://%s:%d  (Ctrl+C to stop)" % (args.host, args.port))
    try:
        app.run(host=args.host, port=args.port, threaded=True,
                use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop.set()
        thread.join(timeout=3.0)
        print("Stopped.")


if __name__ == "__main__":
    main()
'''

# ---------------------------------------------------------------- index.html
FILES["attendance_agent/templates/index.html"] = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Attendance &amp; Engagement Agent</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; margin: 0; background: #f3f4f6; color: #111827; }
  header { background: #1f2937; color: #fff; padding: 12px 20px; }
  main { padding: 16px 20px; max-width: 1100px; margin: 0 auto; }
  .notice { background: #fef3c7; border: 1px solid #f59e0b; padding: 10px 14px; border-radius: 6px; margin-bottom: 14px; }
  .grid { display: flex; flex-wrap: wrap; gap: 14px; }
  .card { background: #fff; border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.15); flex: 1 1 340px; margin-bottom: 14px; }
  img#video { width: 100%; border-radius: 6px; background: #000; }
  #score { font-size: 56px; font-weight: bold; }
  .High { color: #15803d; } .Medium { color: #b45309; } .Low { color: #b91c1c; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border-bottom: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
  .small { color: #6b7280; font-size: 13px; }
  .err { color: #b91c1c; }
  button { background: #b91c1c; color: #fff; border: 0; padding: 10px 16px; border-radius: 6px; cursor: pointer; }
</style>
</head>
<body>
<header><h1 style="margin:0;font-size:20px">Multimodal Attendance &amp; Engagement Agent</h1></header>
<main>
  <div class="notice"><b>Consent notice:</b> Video and audio are processed locally and not stored.
  Only attendance, engagement scores and suggestions are saved as CSV files on this computer.</div>

  <div class="grid">
    <div class="card">
      <h2>Live camera</h2>
      <img id="video" src="/video_feed" alt="Live video stream">
      <div class="small" id="mode"></div>
      <div class="small err" id="error"></div>
    </div>
    <div class="card">
      <h2>Engagement (last 10 s window)</h2>
      <div id="score">--</div>
      <div id="label" style="font-size:22px">Waiting for first window...</div>
      <div class="small" id="detail"></div>
      <h2>Suggestions</h2>
      <ul id="suggestions"><li class="small">No suggestions yet.</li></ul>
      <div class="small">Suggestions are advice only.</div>
    </div>
  </div>

  <div class="card">
    <h2>Attendance</h2>
    <table>
      <thead><tr><th>Person ID</th><th>First seen</th><th>Marked present at</th></tr></thead>
      <tbody id="attendance"><tr><td colspan="3" class="small">Nobody marked present yet.</td></tr></tbody>
    </table>
  </div>

  <button id="del">Delete all data</button>
  <span class="small"> Auto-refreshes every 5 seconds.</span>
</main>

<script>
function esc(s) {
  return String(s).replace(/[&<>"]/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" }[c];
  });
}

async function refresh() {
  try {
    var r = await fetch("/api/status");
    var d = await r.json();

    document.getElementById("mode").textContent = "Mode: " + d.mode + (d.audio_note ? " | " + d.audio_note : "");
    document.getElementById("error").textContent = d.error || "";

    var e = d.engagement;
    var scoreEl = document.getElementById("score");
    var labelEl = document.getElementById("label");
    if (e) {
      scoreEl.textContent = Math.round(e.engagement);
      scoreEl.className = e.label;
      labelEl.textContent = e.label + " engagement";
      labelEl.className = e.label;
      document.getElementById("detail").textContent =
        "face " + e.face_score + " | movement " + e.movement_score +
        " | audio " + e.audio_score + " | at " + e.window_end;
    } else {
      scoreEl.textContent = "--";
      scoreEl.className = "";
      labelEl.textContent = "Waiting for first window...";
      labelEl.className = "";
      document.getElementById("detail").textContent = "";
    }

    var sug = d.suggestions.slice().reverse();
    document.getElementById("suggestions").innerHTML = sug.length
      ? sug.map(function (s) { return "<li>" + esc(s.suggestion) + " <span class=\"small\">(" + esc(s.created_at) + ")</span></li>"; }).join("")
      : "<li class=\"small\">No suggestions yet.</li>";

    document.getElementById("attendance").innerHTML = d.attendance.length
      ? d.attendance.map(function (a) {
          return "<tr><td>" + esc(a.person_id) + "</td><td>" + esc(a.first_seen) +
                 "</td><td>" + esc(a.marked_present_at) + "</td></tr>";
        }).join("")
      : "<tr><td colspan=\"3\" class=\"small\">Nobody marked present yet.</td></tr>";
  } catch (err) {
    document.getElementById("error").textContent = "Dashboard cannot reach the server.";
  }
}

document.getElementById("del").addEventListener("click", async function () {
  if (!confirm("Delete all attendance, engagement and suggestion data?")) { return; }
  try { await fetch("/api/delete", { method: "POST" }); } catch (err) {}
  refresh();
});

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
'''

# ---------------------------------------------------------------- README.md
FILES["attendance_agent/README.md"] = r'''# Multimodal Attendance & Engagement Agent

A very basic, offline classroom helper:

- **Attendance**: Haar-cascade face detection + centroid tracking. A person is marked Present only after being seen continuously for 5 seconds (`--present-seconds` or `config.py`).
- **Engagement** (every 10 s): `0.5*face + 0.2*movement + 0.3*audio`, scaled 0-100. High >= 66, Medium 33-65, Low < 33.
- **Suggestions**: simple rules over recent windows. Advice only.
- **Dashboard**: live MJPEG video, score, attendance table, suggestions, consent notice, "Delete all data" button.

Video and audio are processed locally and are not stored. Only CSV files are saved in `data/`.

## Setup on Windows 10/11 (VS Code terminal)

```
py -3.11 -m venv .venv
.venv\Scripts\activate
python build_project.py
cd attendance_agent
pip install -r requirements.txt
```

If PowerShell blocks activation, run once:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

## Run

```
python app.py --demo                  (synthetic data, no camera or mic)
python app.py                         (webcam + microphone)
python app.py --video C:\path\clip.mp4
python app.py --present-seconds 3     (faster attendance)
```

Open http://127.0.0.1:5000 and press Ctrl+C in the terminal to stop.

## Notes

- Frames are downscaled to 640 px wide and processed at about 10 FPS.
- No microphone? The audio score is 0 and the app keeps running.
- No camera? The dashboard shows an error message; use `--demo` or `--video`.
- A face that disappears for more than 1 second loses its ID (see `MAX_GAP_SECONDS`).
- Data files: `data/attendance.csv`, `data/engagement.csv`, `data/suggestions.csv`.
- Haar cascades are basic: they work best with frontal, well-lit faces.
'''


def build() -> None:
    """Write all files, then zip the project folder."""
    for rel, content in FILES.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content.lstrip("\n"), encoding="utf-8", newline="\n")
        print("wrote", rel)

    zip_path = ROOT / (PROJECT + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in FILES:
            zf.write(ROOT / rel, arcname=rel)
    print("created", zip_path.name)

    print("\nNext steps:")
    print("  cd attendance_agent")
    print("  pip install -r requirements.txt")
    print("  python app.py --demo")
    print("  then open http://127.0.0.1:5000")


if __name__ == "__main__":
    build()
