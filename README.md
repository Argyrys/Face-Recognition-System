# VC Cabin — AI-Based Face Recognition for Entry (Prototype)

Prototype of the 15-day project: a software-only face recognition system for
visitor monitoring at a VC (Vice-Chancellor) cabin entry. The system detects,
recognizes, logs, and notifies — **the human makes the access decision**; no
door or lock is controlled.

## Stack

| Layer      | Choice |
|------------|--------|
| Face detection | OpenCV YuNet (ONNX, `face_detection_yunet_2023mar.onnx`) |
| Face recognition | OpenCV SFace (ONNX, `face_recognition_sface_2021dec.onnx`), cosine threshold 0.363 |
| Storage | SQLite (`database/vc_entry.db`) — persons, entry log, alerts |
| Dashboard | Flask web app (visitor alert card, entry log, enrollment) |
| Notifications | Windows toast (optional) + dashboard alerts |

## Setup

```powershell
python -m pip install -r requirements.txt
```

Models are already in `models/`. If missing, download from the OpenCV zoo
(**media** endpoint, not raw — raw returns Git LFS pointers):

```powershell
Invoke-WebRequest "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet_2023mar.onnx" -OutFile models/face_detection_yunet_2023mar.onnx
Invoke-WebRequest "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface_2021dec.onnx" -OutFile models/face_recognition_sface_2021dec.onnx
```

## Run

```powershell
python app.py                          # webcam 0, or sample_data if no webcam
python app.py --source video.mp4       # video file
python app.py --source sample_data     # image folder (dev, no camera needed)
python app.py --no-toast               # disable Windows notifications
```

Open http://127.0.0.1:5000

- **Dashboard** — pending visitor alert card (photo, Name, Category,
  Department, ID, Time, Recognition, Confidence %, Acknowledge button),
  live pipeline status, and the entry log.
- **Enrollment** — `/enroll`: consent-gated registration with one or more
  photos (embeddings averaged), list of enrolled persons, deactivate action.

`VC_COOLDOWN` (default 15 s) suppresses duplicate entries for the same
person/unknown. `VC_SOURCE` overrides the default frame source.

## Enroll from CLI (optional)

```powershell
python enrollment.py --image sample_data\lena.jpg --name "Lena" --role Faculty --department CSE --id EMP001
```

## Evaluation

```powershell
python evaluate.py --self-test
# or, probe images of a known identity against a gallery:
python evaluate.py --gallery sample_data --probe probe_data --label lena.jpg
```

Writes `reports/evaluation.csv` with TP/FP/FN/TN, accuracy, precision,
recall, and F1. For the report, also run the dashboard and note the mean
confidence of MATCHED entries (true-positive rate) from the entry log.

## Privacy / consent

Enrollment requires an explicit consent checkbox (or the CLI is only for
consented subjects). Data stays local (SQLite + snapshot JPGs in
`static/snapshots/`), persons can be deactivated, and deleting
`database/vc_entry.db` plus `static/snapshots/` erases everything.

## Layout

```
app.py            Flask dashboard + enrollment API
pipeline.py       background thread: frames -> detect -> match -> log -> notify
face_detection.py YuNet wrapper
face_recognition.py SFace embed + cosine best-match
enrollment.py     consented enrollment (single/multi image, CLI too)
database.py       SQLite: persons / entry_log / alerts
camera.py         frame source: webcam | video file | image directory
notification.py   Windows toast (fire-and-forget)
evaluate.py       confusion-matrix metrics -> reports/evaluation.csv
templates/        dashboard.html, enrollment.html
models/           YuNet + SFace ONNX weights
sample_data/      dev images (no camera required)
```
