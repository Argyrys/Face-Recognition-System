# VC Cabin — AI-Based Face Recognition for Entry

A locally-run face recognition system for visitor monitoring at a VC
(Vice-Chancellor) cabin entry. The system detects, recognizes, logs, and
notifies — **the human makes the access decision**; no door or lock is
controlled.

## Stack

| Layer      | Choice |
|------------|--------|
| Face detection | OpenCV YuNet (ONNX, `face_detection_yunet_2023mar.onnx`) |
| Face recognition | OpenCV SFace (ONNX, `face_recognition_sface_2021dec.onnx`), cosine threshold tunable (`VC_MIN_COSINE`, default 0.363) |
| Storage | SQLite (`database/vc_entry.db`) — persons, entry log, alerts, audit trail |
| Dashboard | Flask web app (login-gated: visitor alerts, entry log, enrollment, audit trail) |
| Server | Waitress (production) / Flask dev server (local); HTTPS via reverse proxy (see below) |
| Notifications | Windows toast + dashboard alerts + optional email/webhook to the guard |
| Tests | pytest (`tests/`, 43 tests) + GitHub Actions CI |

## Security appliance features

Beyond detect/recognize/log, the system behaves like a security appliance:

- **N-frame confirmation** — every face is tracked across frames (IoU
  tracker); a MATCHED or UNKNOWN entry is only logged after
  `VC_CONFIRM_FRAMES` (3) consecutive frames agree on the identity. One
  blurred or lucky frame can no longer create or deny a log entry.
- **Liveness / anti-spoof (passive PAD)** — per track the pipeline measures
  landmark motion and eye-region texture energy. A printed photo or frozen
  screen (no motion, no blinks) is recorded as a **presentation attack** and
  raises a critical alert; on live sources (`enforced` mode, the default) it
  can never become MATCHED. Modes: `enforced` | `monitor` | `off`
  (`VC_LIVENESS`). Honest limit: a high-quality *video replay* of a blinking
  person can pass passive checks — see `docs/SECURITY.md`.
- **Camera watchdog** — if a live source stops producing frames for
  `VC_STALL_SECONDS` (25 s), the pipeline is automatically restarted with
  backoff, the dashboard shows a red *camera outage* banner, the outage and
  recovery are written to the audit trail, and a critical alert goes out.
- **Out-of-band alerts** — SMTP email and/or a generic (Discord-compatible)
  webhook, configured by environment variables; per-channel rate limiting
  (`VC_ALERT_MIN_INTERVAL`) so a noisy stream can't flood the guard, while
  `critical` events (spoof, camera outage) always go through.
- **Evidence-based threshold** — `threshold_sweep.py` measures genuine vs
  imposter score distributions on your images and recommends operating
  points (see *Threshold sweep* below).
- **Backups** — `backup.py` makes a consistent, verified SQLite backup plus
  snapshot archive with pruning (see *Backups* below).
- **Hardening** — security headers (`nosniff`, `X-Frame-Options`,
  `Referrer-Policy`, `no-store` on APIs), login lockout, CSRF defence,
  append-only audit trail, retention purge, single-password admin gate.

## Setup

**Windows (one-click):** double-click `run.bat`. It finds your Python
(prefers 3.14, falls back to `python` / `py`), installs the dependencies from
`requirements.txt` on the first run only (checked via imports, so later runs
start instantly), verifies the model files are present, and starts the
dashboard. Any arguments are passed through (`run.bat --port 5001`).

Manual setup:

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
run.bat                                # Windows: setup + start in one double-click
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
| `VC_MIN_COSINE` | `0.363` | cosine match threshold (tune with `threshold_sweep.py`) |
| `VC_CONFIRM_FRAMES` / `VC_UNKNOWN_FRAMES` | `3` / `3` | N-frame confirmation window |
| `VC_LIVENESS` | `enforced` | anti-spoof: `enforced` \| `monitor` \| `off` |
| `VC_LIVENESS_MIN_MOTION` | `0.004` | landmark-motion floor for a "live" verdict |
| `VC_LIVENESS_HOLD_FRAMES` | `20` | frames to wait for liveness evidence before logging |
| `VC_STALL_SECONDS` | `25` | no frames on a live source => watchdog restart |
| `VC_ALERT_MIN_INTERVAL` | `60` | per-channel alert rate limit (critical bypasses) |
| `VC_SMTP_HOST` / `VC_SMTP_PORT` / `VC_SMTP_USER` / `VC_SMTP_PASS` / `VC_SMTP_FROM` / `VC_SMTP_STARTTLS` | – / `587` / – / – / `vc-cabin@localhost` / `1` | email alert channel |
| `VC_ALERT_EMAIL_TO` | – | guard mailbox (with SMTP set = email alerts on) |
| `VC_ALERT_WEBHOOK_URL` | – | generic/Discord webhook for alerts |
| `VC_SSL_PEM` | – | **reserved** — TLS is terminated by a reverse proxy, not Waitress (see *HTTPS*) |
| `VC_BACKUP_DIR` / `VC_BACKUP_KEEP` | `backups/` / `30` | backup archive location & retention |
| `VC_RETENTION_DAYS` | `30` | entry/alert/snapshot retention window |
| `VC_AUDIT_RETENTION_DAYS` | `90` | audit-log retention window |
| `VC_SESSION_HOURS` | `12` | login session lifetime |
| `VC_COOLDOWN` | `15` | seconds before the same person is re-logged |
| `VC_MAX_UPLOAD_MB` | `8` | upload size cap |
| `VC_LOG_LEVEL` / `VC_LOG_FILE` | `INFO` / `logs/vc_cabin.log` | logging (rotating, 2 MB × 3) |
| `VC_TOAST` | `1` | Windows desktop notifications |

### Alert routing (email / webhook)

```powershell
# email to the guard
$env:VC_SMTP_HOST = "smtp.example.com"; $env:VC_SMTP_USER = "user"
$env:VC_SMTP_PASS = "app-password";     $env:VC_ALERT_EMAIL_TO = "guard@example.com"
# or a Discord-compatible webhook
$env:VC_ALERT_WEBHOOK_URL = "https://discord.com/api/webhooks/...."
```

With neither set, behavior is exactly as before (on-screen + toast only).
Email needs `VC_SMTP_HOST` **and** `VC_ALERT_EMAIL_TO`; the webhook fires on
`VC_ALERT_WEBHOOK_URL` alone. Delivery is rate-limited per channel
(`VC_ALERT_MIN_INTERVAL`, default 60 s); `critical` alerts (suspected
spoofing, camera outage) always deliver. A failing channel is logged and
never crashes the pipeline.

### HTTPS

The production server is Waitress, which **does not implement TLS** — it is
designed to sit behind a TLS-terminating reverse proxy. To serve HTTPS:

**Caddy** (automatic certificates):

```
cabin.example.com {
    reverse_proxy 127.0.0.1:5000
}
```

**nginx**:

```nginx
server {
    listen 443 ssl;
    ssl_certificate     /etc/ssl/cabin.crt;
    ssl_certificate_key /etc/ssl/cabin.key;
    location / { proxy_pass http://127.0.0.1:5000; }
}
```

Bind the app to localhost only (`VC_HOST=127.0.0.1`, the default) so the
proxy is the only public entry point. The dashboard's MJPEG stream proxies
through both.

### Backups

```powershell
python backup.py            # consistent SQLite backup + snapshots -> backups/
python backup.py --keep 14  # keep only the newest 14 archives
python backup.py --verify   # integrity-check the live database
```

Safe while the app runs (SQLite backup API, no write race). Each archive is
a timestamped zip with `vc_entry.db`, `snapshots/`, and a `manifest.json`
(row counts + restore steps); old archives beyond `VC_BACKUP_KEEP` are
pruned. Schedule it hourly/daily via Task Scheduler:

```powershell
schtasks /create /tn "VC-Cabin-Backup" /sc daily /st 02:00 `
  /tr "powershell -Command `"cd 'C:\path\to\VC_Face_Recognition'; python backup.py`""
```

(or `cron` / a compose sidecar on Linux: `0 2 * * * python /app/backup.py`).

### Threshold sweep (evidence-based operating point)

```powershell
python threshold_sweep.py                 # sample_data with augmentations
python threshold_sweep.py --dir my_probes --max-far 0.01
```

Sweeps the cosine threshold 0.20–0.70 over genuine (same identity across
augmented variants) and imposter (cross-identity) pairs, and writes
`reports/threshold_sweep.csv` + `.md` with FAR/FRR/precision/recall/F1 per
threshold, the **proven-safe interval**, and a recommended operating point.
On the bundled sample data: safe interval **[0.21, 0.69]**, recommended
**0.450**, and the vendor default 0.363 sits inside the interval. Apply a
recommendation with `VC_MIN_COSINE`. Before freezing the threshold for a
real installation, re-run the sweep with probe photos taken by the actual
camera (augmented variants are a surrogate for a second capture session).

## Tests & CI

```powershell
python -m pytest tests/ -q
```

The suite (43 tests) covers the login gate, lockout, CSRF defence, audit
trail, enrollment consent, deactivation, retention purge, tracker identity
streaks, liveness verdicts, threshold-sweep scoring, alert rate limiting and
payloads, backup archives, schema migration, and security headers — all
against throwaway databases, no camera or models needed. GitHub Actions
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
`VC_AUDIT_RETENTION_DAYS`. See `docs/PRIVACY_NOTICE.md` for the data
practices notice and `docs/SECURITY.md` for the threat model.

## Layout

```
app.py            Flask app: auth gate, dashboard/enrollment/audit API, watchdog
serve.py          production entrypoint (Waitress)
config.py         env-driven configuration + rotating-file logging
pipeline.py       background thread: detect -> track -> confirm -> liveness -> match -> log -> notify
tracker.py        multi-face IoU tracker with per-visitor evidence accumulation
liveness.py       passive anti-spoof: landmark motion + blink detection
alerter.py        email + webhook alert channels (rate-limited)
backup.py         consistent SQLite + snapshot backups with pruning & verify
face_detection.py YuNet wrapper
face_recognition.py SFace embed + cosine best-match (tunable threshold)
enrollment.py     consented enrollment (single/multi image, CLI too)
database.py       SQLite: persons / entry_log / alerts / audit_log / settings
camera.py         frame source: webcam | video file | image directory | stream
notification.py   Windows toast (fire-and-forget)
evaluate.py       confusion-matrix metrics -> reports/evaluation.csv
threshold_sweep.py threshold sweep + ROC-style report -> reports/threshold_sweep.{csv,md}
docs/             SECURITY.md (threat model), PRIVACY_NOTICE.md
templates/        login.html, dashboard.html, enrollment.html
tests/            pytest suite (no camera required)
models/           YuNet + SFace ONNX weights
sample_data/      dev images (no camera required)
Dockerfile / docker-compose.yml / .github/workflows/ci.yml
```
