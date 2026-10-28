"""Simple rule-based teaching suggestions (advice only)."""
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
