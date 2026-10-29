"""VC Cabin dashboard + enrollment web app (Flask).

Production notes:
- All pages and APIs (except /login, /health and static assets) require an
  admin login; the gate lives in a single before_request hook.
- Every operator action (login, ack/reject, enroll, deactivate, toggles) is
  written to an append-only audit_log table.
- Logging goes to console + rotating file (see config.setup_logging).
- `python serve.py` runs the same app under Waitress for production;
  `python app.py` keeps the dev server for local work.
"""

import argparse
import datetime
import logging
import os
import secrets
import threading
import time
import urllib.parse

import cv2
import numpy as np
from flask import (Flask, Response, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

from config import Config, setup_logging
from database import Database
from enrollment import SNAPSHOT_DIR, enroll
from face_detection import FaceDetector
from face_recognition import FaceRecognizer, best_match
from notification import show_toast
from pipeline import Pipeline

log = setup_logging()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

app = Flask(__name__)
app.config.update(
    SECRET_KEY=None,                       # resolved below (env or DB)
    MAX_CONTENT_LENGTH=Config.MAX_UPLOAD_MB * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    PERMANENT_SESSION_LIFETIME=datetime.timedelta(hours=Config.SESSION_HOURS),
)

db = Database()
pipeline = None

_detector = None
_recognizer = None


def _engine():
    global _detector, _recognizer
    if _detector is None:
        _detector = FaceDetector()
        _recognizer = FaceRecognizer()
    return _detector, _recognizer


def _redirect_msg(message: str, target="/enroll"):
    return redirect(f"{target}?{urllib.parse.urlencode({'msg': message})}")


# --------------------------------------------------------------------- auth

MAX_ATTEMPTS = 5          # failed logins per IP before a cooldown
LOCK_SECONDS = 60
LOGIN_ATTEMPTS = {}       # ip -> (fail_count, locked_until monotonic)


def _secret_key() -> str:
    """VC_SECRET_KEY if provided; otherwise a key generated once, kept in the DB
    so sessions survive restarts without extra setup."""
    if Config.SECRET_KEY:
        return Config.SECRET_KEY
    key = db.get_setting("secret_key")
    if not key:
        key = secrets.token_hex(32)
        db.set_setting("secret_key", key)
    return key


_PW_HASH = None


def _password_hash() -> str:
    """VC_ADMIN_PASSWORD if provided; otherwise a hash persisted in the DB.
    On the very first run a random password is generated and printed to the
    log so the dashboard is never left open."""
    global _PW_HASH
    if _PW_HASH:
        return _PW_HASH
    if Config.ADMIN_PASSWORD:
        _PW_HASH = generate_password_hash(Config.ADMIN_PASSWORD)
        db.set_setting("admin_password_hash", _PW_HASH)
        return _PW_HASH
    _PW_HASH = db.get_setting("admin_password_hash")
    if not _PW_HASH:
        password = secrets.token_urlsafe(12)
        _PW_HASH = generate_password_hash(password)
        db.set_setting("admin_password_hash", _PW_HASH)
        log.warning("First run: generated admin password -> %s "
                    "(set VC_ADMIN_PASSWORD to choose your own)", password)
    return _PW_HASH


def _check_password(attempt: str) -> bool:
    try:
        return check_password_hash(_password_hash(), attempt)
    except ValueError:
        return False


app.config["SECRET_KEY"] = _secret_key()

PUBLIC_PATHS = {"/login", "/health"}


@app.before_request
def _require_auth():
    """Single gate for every page and API. Static assets, /login and /health
    stay open; everything else needs a session. Mutating requests that the
    browser labels cross-site are refused outright (CSRF defence)."""
    if request.path in PUBLIC_PATHS or request.path.startswith("/static/"):
        return None
    if not session.get("auth"):
        if request.path.startswith("/api/"):
            return jsonify({"ok": False, "error": "authentication required"}), 401
        return redirect(url_for("login_page", next=request.path))
    if (request.method in ("POST", "PUT", "PATCH", "DELETE")
            and request.headers.get("Sec-Fetch-Site") == "cross-site"):
        return jsonify({"ok": False, "error": "cross-site request blocked"}), 403
    return None


@app.route("/login", methods=["GET", "POST"])
def login_page():
    if session.get("auth"):
        return redirect("/")
    error = None
    if request.method == "POST":
        ip = request.remote_addr or "unknown"
        fails, unlock_at = LOGIN_ATTEMPTS.get(ip, (0, 0.0))
        if time.monotonic() < unlock_at:
            wait = int(unlock_at - time.monotonic()) + 1
            error = f"Too many failed attempts. Try again in {wait}s."
        elif _check_password(request.form.get("password", "")):
            LOGIN_ATTEMPTS.pop(ip, None)
            session.clear()                 # avoid session fixation
            session.permanent = True
            session["auth"] = True
            db.audit("login.success", f"ip={ip}")
            log.info("login success from %s", ip)
            target = request.args.get("next", "/")
            if not target.startswith("/") or target.startswith("//"):
                target = "/"
            return redirect(target)
        else:
            fails += 1
            unlock_at = time.monotonic() + LOCK_SECONDS if fails >= MAX_ATTEMPTS else 0.0
            LOGIN_ATTEMPTS[ip] = (fails, unlock_at)
            db.audit("login.failed", f"ip={ip} attempt={fails}")
            log.warning("login failed from %s (attempt %d)", ip, fails)
            time.sleep(0.4)                 # slow brute-force attempts
            error = "Incorrect password."
    return render_template("login.html", error=error)


@app.route("/logout", methods=["GET", "POST"])
def logout():
    db.audit("logout", f"ip={request.remote_addr}")
    session.clear()
    return redirect(url_for("login_page"))


@app.route("/health")
def api_health():
    """Unauthenticated liveness probe for Docker / monitoring."""
    return jsonify({
        "status": "ok",
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
    })


# ---------------------------------------------------------------- dashboard

@app.route("/")
def dashboard():
    return render_template("dashboard.html", stats=db.stats())


@app.route("/api/status")
def api_status():
    status = dict(pipeline.status) if pipeline else {"running": False, "state": "not started"}
    status["stats"] = db.stats()
    return jsonify(status)


@app.route("/api/recognition", methods=["POST"])
def api_recognition():
    """Turn recognition on/off from the dashboard (preview keeps streaming)."""
    if pipeline is None:
        return jsonify({"ok": False, "error": "pipeline not started"}), 503
    data = request.get_json(silent=True) or {}
    if "enabled" not in data:
        return jsonify({"ok": False, "error": "missing 'enabled' field"}), 400
    pipeline.set_enabled(bool(data["enabled"]))
    db.audit("recognition.toggle", f"enabled={bool(data['enabled'])}")
    log.info("recognition %s by operator", "enabled" if data["enabled"] else "paused")
    return jsonify({"ok": True, "recognition": pipeline.status["recognition"]})


@app.route("/api/camera-stream")
def api_camera_stream():
    """MJPEG feed of whatever the pipeline is looking at (browser <img>)."""
    if pipeline is None:
        return "camera feed unavailable: pipeline not started", 503

    boundary = b"vcframe"

    def generate():
        last = None
        while True:
            jpeg = pipeline.latest_jpeg()
            if jpeg is None:
                time.sleep(0.25)
                continue
            if jpeg != last:
                last = jpeg
                yield (
                    b"--" + boundary + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                    + jpeg + b"\r\n"
                )
            else:
                time.sleep(0.05)

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=vcframe",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


@app.route("/api/alerts")
def api_alerts():
    alerts = db.pending_alerts()
    for a in alerts:
        a["snapshot_url"] = f"/static/snapshots/{a['snapshot']}" if a.get("snapshot") else None
        a["confidence_pct"] = round(a["confidence"] * 100, 1) if a.get("confidence") is not None else None
    return jsonify(alerts)


@app.route("/api/alerts/<int:alert_id>/ack", methods=["POST"])
def api_ack(alert_id):
    ok = db.ack_alert(alert_id)
    if ok:
        db.audit("alert.ack", f"alert=#{alert_id}")
        log.info("alert #%s acknowledged", alert_id)
    return jsonify({"ok": ok})


@app.route("/api/alerts/<int:alert_id>/reject", methods=["POST"])
def api_reject(alert_id):
    ok = db.reject_alert(alert_id)
    if ok:
        db.audit("alert.reject", f"alert=#{alert_id}")
        log.info("alert #%s rejected", alert_id)
    return jsonify({"ok": ok})


@app.route("/api/resolved")
def api_resolved():
    limit = min(int(request.args.get("limit", 20)), 100)
    alerts = db.resolved_alerts(limit)
    for a in alerts:
        a["snapshot_url"] = f"/static/snapshots/{a['snapshot']}" if a.get("snapshot") else None
        a["confidence_pct"] = round(a["confidence"] * 100, 1) if a.get("confidence") is not None else None
    return jsonify(alerts)


@app.route("/api/alerts/ack_all", methods=["POST"])
def api_ack_all():
    n = db.ack_all()
    if n:
        db.audit("alert.ack_all", f"count={n}")
        log.info("%s alerts acknowledged at once", n)
    return jsonify({"acked": n})


@app.route("/api/entries")
def api_entries():
    limit = min(int(request.args.get("limit", 50)), 200)
    entries = db.entries(limit)
    for e in entries:
        e["snapshot_url"] = f"/static/snapshots/{e['snapshot']}" if e.get("snapshot") else None
    return jsonify(entries)


@app.route("/api/audit")
def api_audit():
    """Recent operator actions (audit trail view)."""
    limit = min(int(request.args.get("limit", 200)), 500)
    return jsonify(db.recent_audit(limit))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """One-shot recognition of an uploaded visitor image."""
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify({"error": "No image uploaded."}), 400
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode the image."}), 400

    detector, recognizer = _engine()
    faces = detector.detect(img)
    face = FaceDetector.best(faces)
    if face is None:
        return jsonify({
            "found": False, "result": "NO_FACE",
            "message": "No face detected in the image.",
        })

    embedding = recognizer.embed(img, face)
    person_id, score = best_match(embedding, db.gallery())
    person = db.person(person_id) if person_id is not None else None
    matched = person is not None and person["status"] == "active"

    snapshot = _save_crop(img, face)
    if matched:
        entry_id = db.add_entry(
            name=person["name"], result="MATCHED", confidence=score,
            snapshot=snapshot, person_id=person["id"],
            role=person["role"], department=person["department"],
            external_id=person["external_id"],
        )
        show_toast(
            "VC Cabin Visitor Alert",
            f"{person['role']} - {person['department']} - MATCHED "
            f"{score * 100:.1f}%",
        )
    else:
        person = None
        entry_id = db.add_entry(
            name="Unknown", result="UNKNOWN", confidence=score,
            snapshot=snapshot,
        )
        show_toast("VC Cabin Visitor Alert",
                   "Unknown / Manual Verification Required")

    alert_id = db.add_alert(entry_id)
    result = "MATCHED" if matched else "UNKNOWN"
    db.audit("image.analyze",
             f"file={file.filename} result={result} conf={float(score):.3f}")
    log.info("analyze '%s' -> %s (%.3f)", file.filename, result, score)
    return jsonify({
        "found": True,
        "result": result,
        "name": person["name"] if matched else "Unknown",
        "role": person["role"] if matched else "Visitor",
        "department": person["department"] if matched else None,
        "external_id": person["external_id"] if matched else None,
        "confidence": round(float(score), 4),
        "confidence_pct": round(float(score) * 100, 1),
        "snapshot_url": f"/static/snapshots/{snapshot}",
        "entry_id": entry_id,
        "alert_id": alert_id,
        "faces": 0 if faces is None else int(len(faces)),
    })


def _save_crop(img, face) -> str:
    x, y, w, h = FaceDetector.box(face)
    margin = int(max(w, h) * 0.25)
    h_img, w_img = img.shape[:2]
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(w_img, x + w + margin), min(h_img, y + h + margin)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"entry_{stamp}.jpg"
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(SNAPSHOT_DIR, filename), img[y1:y2, x1:x2])
    return filename


# -------------------------------------------------------------- enrollment

@app.route("/enroll")
def enroll_page():
    return render_template(
        "enrollment.html",
        persons=db.persons(active_only=False),
        msg=request.args.get("msg", ""),
    )


@app.route("/enroll", methods=["POST"])
def enroll_submit():
    name = request.form.get("name", "").strip()
    role = request.form.get("role", "")
    department = request.form.get("department", "").strip()
    external_id = request.form.get("external_id", "").strip()
    consent = request.form.get("consent") == "on"
    files = [f for f in request.files.getlist("photos") if f and f.filename]

    if not (name and department and external_id):
        return _redirect_msg("Name, department and ID are required.")
    if not files:
        return _redirect_msg("Upload at least one photo of the person.")

    try:
        detector, recognizer = _engine()
        result = enroll(
            db, detector, recognizer,
            [f.read() for f in files],
            name=name, role=role or "Student", department=department,
            external_id=external_id, consent=consent,
        )
    except ValueError as exc:
        db.audit("person.enroll_failed", f"{name}: {exc}")
        log.warning("enroll failed for '%s': %s", name, exc)
        return _redirect_msg(str(exc))

    db.audit("person.enroll",
             f"#{result['id']} {name} ({role or 'Student'}, {department}) "
             f"photos={result['images']}")
    log.info("enrolled #%s %s from %s photo(s)", result["id"], name, result["images"])
    return _redirect_msg(
        f"Enrolled {name} (#{result['id']}) from {result['images']} photo(s)."
    )


@app.route("/api/persons")
def api_persons():
    return jsonify(db.persons(active_only=False))


@app.route("/api/persons/<int:person_id>/deactivate", methods=["POST"])
def api_deactivate(person_id):
    person = db.person(person_id)
    ok = db.deactivate_person(person_id)
    if ok:
        db.audit("person.deactivate",
                 f"#{person_id} {person['name'] if person else ''}".strip())
        log.info("person #%s deactivated", person_id)
    return jsonify({"ok": ok})


# --------------------------------------------------------------- retention

def run_retention():
    """Delete entry/alert/audit rows and snapshot files past their window."""
    try:
        removed = db.purge_older_than(
            Config.RETENTION_DAYS, Config.AUDIT_RETENTION_DAYS
        )
        cutoff = time.time() - Config.RETENTION_DAYS * 86400
        snaps = 0
        for fname in os.listdir(SNAPSHOT_DIR):
            if not fname.startswith("entry_"):
                continue
            path = os.path.join(SNAPSHOT_DIR, fname)
            try:
                if os.path.getmtime(path) < cutoff:
                    os.remove(path)
                    snaps += 1
            except OSError:
                pass
        removed["snapshots"] = snaps
        if any(removed.values()):
            log.info("retention purge (> %sd): %s", Config.RETENTION_DAYS, removed)
            db.audit("retention.purge", str(removed))
    except Exception:
        log.exception("retention purge failed")


def _retention_loop():
    while True:
        time.sleep(Config.RETENTION_INTERVAL_HOURS * 3600)
        run_retention()


# ------------------------------------------------------------------- boot

def _event_logger():
    """Mirror pipeline events (matched/unknown visitors) into the log file."""
    seen = None
    while True:
        ev = pipeline.status.get("last_event") if pipeline else None
        if ev is not None and ev is not seen:
            seen = ev
            log.info("event: %s %s conf=%.3f alert=#%s",
                     ev["name"], ev["result"], ev["confidence"], ev["alert_id"])
        time.sleep(1.0)


def start_backend(source: str = None, toast: bool = None):
    """Start the recognition pipeline + housekeeping threads (idempotent).
    Shared by the dev server (app.py) and production (serve.py)."""
    global pipeline
    if pipeline is not None:
        return pipeline
    spec = source or default_source()
    pipeline = Pipeline(spec, db, toast=Config.TOAST if toast is None else toast)
    pipeline.start()
    db.audit("system.start", f"source={pipeline.source.label}")
    log.info("recognition pipeline started (source=%s)", pipeline.source.label)
    run_retention()
    threading.Thread(target=_retention_loop, daemon=True,
                     name="vc-retention").start()
    threading.Thread(target=_event_logger, daemon=True,
                     name="vc-events").start()
    return pipeline


def default_source() -> str:
    if os.environ.get("VC_SOURCE"):
        return os.environ["VC_SOURCE"]
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ok, _ = cap.read()
        cap.release()
        if ok:
            return "0"
    if os.path.isdir(SAMPLE_DIR) and os.listdir(SAMPLE_DIR):
        return SAMPLE_DIR
    return "0"


def main():
    parser = argparse.ArgumentParser(description="VC Cabin face recognition dashboard")
    parser.add_argument("--source", default=None,
                        help="webcam index (0), video file, image directory, "
                             "or network stream URL (http://... from IP Webcam/DroidCam)")
    parser.add_argument("--host", default=Config.HOST,
                        help="bind address; use 0.0.0.0 to serve other devices (default: %(default)s)")
    parser.add_argument("--port", type=int, default=Config.PORT)
    parser.add_argument("--no-toast", action="store_true",
                        help="disable Windows desktop notifications")
    args = parser.parse_args()

    start_backend(args.source, toast=not args.no_toast)

    log.info("VC Cabin dashboard -> http://%s:%s", args.host, args.port)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False,
            threaded=True)


if __name__ == "__main__":
    main()
