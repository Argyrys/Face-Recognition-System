"""VC Cabin dashboard + enrollment web app (Flask)."""

import argparse
import datetime
import os
import time
import urllib.parse

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, render_template, request, url_for

from database import Database
from enrollment import SNAPSHOT_DIR, enroll
from face_detection import FaceDetector
from face_recognition import FaceRecognizer, best_match
from notification import show_toast
from pipeline import Pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")

app = Flask(__name__)
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


# ---------------------------------------------------------------- dashboard

@app.route("/")
def dashboard():
    return render_template("dashboard.html", stats=db.stats())


@app.route("/api/status")
def api_status():
    status = dict(pipeline.status) if pipeline else {"running": False, "state": "not started"}
    status["stats"] = db.stats()
    return jsonify(status)


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
    return jsonify({"ok": db.ack_alert(alert_id)})


@app.route("/api/alerts/ack_all", methods=["POST"])
def api_ack_all():
    return jsonify({"acked": db.ack_all()})


@app.route("/api/entries")
def api_entries():
    limit = min(int(request.args.get("limit", 50)), 200)
    entries = db.entries(limit)
    for e in entries:
        e["snapshot_url"] = f"/static/snapshots/{e['snapshot']}" if e.get("snapshot") else None
    return jsonify(entries)


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
    return jsonify({
        "found": True,
        "result": "MATCHED" if matched else "UNKNOWN",
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
        return _redirect_msg(str(exc))

    return _redirect_msg(
        f"Enrolled {name} (#{result['id']}) from {result['images']} photo(s)."
    )


@app.route("/api/persons")
def api_persons():
    return jsonify(db.persons(active_only=False))


@app.route("/api/persons/<int:person_id>/deactivate", methods=["POST"])
def api_deactivate(person_id):
    ok = db.deactivate_person(person_id)
    return _redirect_msg(
        f"Person #{person_id} deactivated." if ok else "Person not found.",
        target="/enroll",
    )


# -------------------------------------------------------------------- main

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
    global pipeline
    parser = argparse.ArgumentParser(description="VC Cabin face recognition dashboard")
    parser.add_argument("--source", default=default_source(),
                        help="webcam index (0), video file, image directory, "
                             "or network stream URL (http://... from IP Webcam/DroidCam)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-toast", action="store_true",
                        help="disable Windows desktop notifications")
    args = parser.parse_args()

    pipeline = Pipeline(args.source, db, toast=not args.no_toast)
    pipeline.start()

    print(f"VC Cabin dashboard -> http://127.0.0.1:{args.port}")
    print(f"Recognition source -> {pipeline.source.label}")
    app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
