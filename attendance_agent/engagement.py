"""Fuse face, movement and audio scores into an engagement score."""
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
