"""Microphone loudness (RMS) with a safe fallback to 0."""
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
