# VC Cabin — AI-Based Face Recognition for Entry

A locally-run face recognition system for visitor monitoring at a VC
(Vice-Chancellor) cabin entry. The system detects, recognizes, logs, and
notifies — **the human makes the access decision**; no door or lock is
controlled.

## Stack

| Layer      | Choice |
|------------|--------|
| Face detection | OpenCV YuNet (ONNX, `face_detection_yunet_2023mar.onnx`) |
| Face recognition | OpenCV SFace (ONNX, `face_recognition_sface_2021dec.onnx`), cosine threshold 0.363 |
| Storage | SQLite (`database/vc_entry.db`) — persons, entry log, alerts, audit trail |
| Dashboard | Flask web app (login-gated: visitor alerts, entry log, enrollment, audit trail) |
| Server | Waitress (production) / Flask dev server (local) |
| Notifications | Windows toast (optional) + dashboard alerts |
| Tests | pytest (`tests/`, 17 tests) + GitHub Actions CI |

## Setup

```powershell
python -m pip install -r requirements.txt        # runtime
python -m pip install -r requirements-dev.txt    # + pytest for the test suite
```

Models are already in `models/`. If missing, download from the OpenCV zoo
(**media** endpoint, not raw — raw returns Git LFS pointers):

```powershell
Invoke-WebRequest "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet_2023mar.onnx" -OutFile models/face_detection_yunet_2023mar.onnx
Invoke-WebRequest "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface_2021dec.onnx" -OutFile models/face_recognition_sface_2021dec.onnx
```

## Authentication

Every page and API (except `/login`, `/health` and static assets) requires the
admin password.

- Set it with `VC_ADMIN_PASSWORD`; **if unset, a random password is generated
  on first run and printed to the log** (`logs/vc_cabin.log` + console) — the
  dashboard is never left open.
- 5 failed attempts trigger a 60-second per-IP lockout.
- Session cookie: HttpOnly, SameSite=Lax, 12h lifetime (`VC_SESSION_HOURS`).
- Mutating requests that browsers label cross-site (`Sec-Fetch-Site:
  cross-site`) are refused (CSRF defence).
- Every operator action — logins (incl. failures), acknowledge/reject,
  enrollment, deactivation, recognition toggles, retention purges — is written
  to an append-only **audit trail** viewable in the dashboard (`Audit Trail`)
  and via `GET /api/audit`.

## Run

### Local development

```powershell
$env:VC_ADMIN_PASSWORD = "your-password"
python app.py                          # webcam 0, or sample_data if no webcam
python app.py --source video.mp4       # video file
python app.py --source sample_data     # image folder (dev, no camera needed)
python app.py --no-toast               # disable Windows notifications
python app.py --host 0.0.0.0           # also serve other devices on the LAN
```

Open http://127.0.0.1:5000 and log in.

### Production (Waitress)

```powershell
$env:VC_ADMIN_PASSWORD = "your-password"
python serve.py                        # Waitress WSGI server (4 threads)
```

`serve.py` and `app.py` run the identical app and the same recognition
pipeline; Waitress replaces the Werkzeug dev server for deployment.

### Docker

```bash
docker build -t vc-cabin .
docker run -p 8000:8000 -e VC_ADMIN_PASSWORD=change-me vc-cabin
# webcam passthrough on Linux:
docker run --device=/dev/video0 -p 8000:8000 -e VC_ADMIN_PASSWORD=change-me vc-cabin
# or compose (reads VC_ADMIN_PASSWORD from a .env file):
docker compose up -d
```

The image health-checks `/health`; DB, snapshots and logs live in named
volumes so data survives restarts. For a headless server, point `VC_CAMERA`
at an IP Webcam / DroidCam stream URL.

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `VC_ADMIN_PASSWORD` | generated on first run | dashboard login password |
| `VC_SECRET_KEY` | generated, stored in DB | session signing key |
| `VC_HOST` / `VC_PORT` | `127.0.0.1` / `5000` | bind address (`0.0.0.0` = LAN) |
| `VC_CAMERA` / `VC_SOURCE` | webcam `0` | camera index, video file, folder, or stream URL |
| `VC_RETENTION_DAYS` | `30` | entry/alert/snapshot retention window |
| `VC_AUDIT_RETENTION_DAYS` | `90` | audit-log retention window |
| `VC_SESSION_HOURS` | `12` | login session lifetime |
| `VC_COOLDOWN` | `15` | seconds before the same person is re-logged |
| `VC_MAX_UPLOAD_MB` | `8` | upload size cap |
| `VC_LOG_LEVEL` / `VC_LOG_FILE` | `INFO` / `logs/vc_cabin.log` | logging (rotating, 2 MB × 3) |
| `VC_TOAST` | `1` | Windows desktop notifications |

## Tests & CI

```powershell
python -m pytest tests/ -q
```

The suite (17 tests) covers the login gate, lockout, CSRF defence, audit
trail, enrollment consent, deactivation, and retention purge against a
throwaway database — no camera or models needed. GitHub Actions
(`.github/workflows/ci.yml`) runs it on every push/PR for Python 3.11 & 3.12.

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
recall, and F1.

## Privacy / consent / retention

Enrollment requires an explicit consent checkbox (the CLI is only for
consented subjects). Data stays local (SQLite + snapshot JPGs in
`static/snapshots/`), persons can be deactivated, and deleting
`database/vc_entry.db` plus `static/snapshots/` erases everything. A
retention job (on startup + every 6h) deletes entry records, alerts, and
snapshot files older than `VC_RETENTION_DAYS`, and audit rows older than
`VC_AUDIT_RETENTION_DAYS`.

## Layout

```
app.py            Flask app: auth gate, dashboard/enrollment/audit API
serve.py          production entrypoint (Waitress)
config.py         env-driven configuration + rotating-file logging
pipeline.py       background thread: frames -> detect -> match -> log -> notify
face_detection.py YuNet wrapper
face_recognition.py SFace embed + cosine best-match
enrollment.py     consented enrollment (single/multi image, CLI too)
database.py       SQLite: persons / entry_log / alerts / audit_log / settings
camera.py         frame source: webcam | video file | image directory
notification.py   Windows toast (fire-and-forget)
evaluate.py       confusion-matrix metrics -> reports/evaluation.csv
templates/        login.html, dashboard.html, enrollment.html
tests/            pytest suite (no camera required)
models/           YuNet + SFace ONNX weights
sample_data/      dev images (no camera required)
Dockerfile / docker-compose.yml / .github/workflows/ci.yml
```
