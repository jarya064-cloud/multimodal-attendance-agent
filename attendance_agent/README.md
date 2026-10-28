# Multimodal Attendance & Engagement Agent

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
