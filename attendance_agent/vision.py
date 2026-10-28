"""Face detection (Haar cascade), centroid tracking and attendance logic."""
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
