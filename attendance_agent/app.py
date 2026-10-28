"""Flask dashboard + background worker for the attendance agent.

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
