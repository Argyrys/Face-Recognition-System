"""Background recognition pipeline: frames -> detect -> match -> log -> notify."""

import datetime
import os
import threading
import time

import cv2
import numpy as np

from camera import FrameSource
from database import Database
from enrollment import SNAPSHOT_DIR
from face_detection import FaceDetector
from face_recognition import FaceRecognizer, best_match
from notification import show_toast

COOLDOWN_SECONDS = float(os.environ.get("VC_COOLDOWN", "15"))
SNAPSHOT_KEEP = 240  # max recent snapshots kept on disk
PREVIEW_MAX_WIDTH = 720  # width of the JPEG shown in the dashboard preview
PROCESS_MAX_WIDTH = int(os.environ.get("VC_MAX_WIDTH", "960"))  # detection input


def _downscale(frame):
    """Shrink wide frames so detection and preview stay fast."""
    h, w = frame.shape[:2]
    if w > PROCESS_MAX_WIDTH:
        frame = cv2.resize(
            frame, (PROCESS_MAX_WIDTH, max(1, int(h * PROCESS_MAX_WIDTH / w)))
        )
    return frame


class Pipeline(threading.Thread):
    def __init__(self, source_spec: str, db: Database, toast: bool = True):
        super().__init__(daemon=True, name="vc-pipeline")
        self.db = db
        self.toast = toast
        self.source = FrameSource(source_spec)
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self._stop_event = threading.Event()
        self._gallery = db.gallery()
        self._gallery_version = db.version
        self._last_logged = {}  # key -> monotonic time
        self._frame_lock = threading.Lock()
        self._latest_jpeg = None  # latest preview frame, for the dashboard feed
        self._enabled = threading.Event()
        self._enabled.set()  # recognition starts ON
        self.status = {
            "running": False,
            "state": "starting",
            "recognition": "on",
            "source": self.source.label,
            "frames": 0,
            "detections": 0,
            "last_event": None,
            "last_frame_at": None,
        }

    # ------------------------------------------------------------- preview

    def latest_jpeg(self):
        """Most recent frame as JPEG bytes, or None if nothing seen yet."""
        with self._frame_lock:
            return self._latest_jpeg

    def _store_preview(self, frame):
        h, w = frame.shape[:2]
        if w > PREVIEW_MAX_WIDTH:
            frame = cv2.resize(
                frame, (PREVIEW_MAX_WIDTH, max(1, int(h * PREVIEW_MAX_WIDTH / w)))
            )
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self._frame_lock:
                self._latest_jpeg = buf.tobytes()

    # ------------------------------------------------------------------ run

    def run(self):
        self.status["running"] = True
        self.status["state"] = "running"
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
        while not self._stop_event.is_set():
            frame = self.source.read()
            if frame is None:
                if self.source.is_stream:
                    # A phone camera drops out (screen off, wifi blip).
                    # Keep retrying instead of killing the pipeline.
                    self.status["state"] = "reconnecting to camera..."
                    time.sleep(1.0)
                    continue
                self.status["state"] = "source ended"
                break
            self.status["frames"] += 1
            self.status["last_frame_at"] = datetime.datetime.now().isoformat(
                timespec="seconds"
            )
            frame = _downscale(frame)
            self._store_preview(frame)
            if not self._enabled.is_set():
                # Recognition paused: preview only, no detect/match/log.
                self.status["state"] = "recognition paused"
                continue
            self.status["state"] = "running"
            try:
                self._process(frame)
            except Exception as exc:  # keep the loop alive on bad frames
                self.status["state"] = f"error: {exc}"
        self.status["running"] = False
        self.source.release()

    def stop(self):
        self._stop_event.set()

    # -------------------------------------------------------- on/off toggle

    def set_enabled(self, on: bool):
        """Turn recognition (detect/match/log/notify) on or off from the dashboard.
        The camera preview keeps streaming either way."""
        if on:
            self._enabled.set()
        else:
            self._enabled.clear()
        self.status["recognition"] = "on" if on else "off"

    def enabled(self) -> bool:
        return self._enabled.is_set()

    # -------------------------------------------------------------- process

    def _process(self, frame: np.ndarray):
        if self.db.version != self._gallery_version:
            self._gallery = self.db.gallery()
            self._gallery_version = self.db.version

        faces = self.detector.detect(frame)
        face = self.detector.best(faces)
        if face is None:
            return
        self.status["detections"] += 1

        embedding = self.recognizer.embed(frame, face)
        person_id, score = best_match(embedding, self._gallery)
        now = time.monotonic()

        if person_id is not None:
            person = self.db.person(person_id)
            matched = person is not None and person["status"] == "active"
        else:
            person = None
            matched = False

        key = f"person:{person['id']}" if matched else "unknown"
        if not self._cooldown(key, now):
            return
        self._last_logged[key] = now

        if matched:
            entry_id = self.db.add_entry(
                name=person["name"], result="MATCHED", confidence=score,
                snapshot=self._save_snapshot(frame, face),
                person_id=person["id"], role=person["role"],
                department=person["department"],
                external_id=person["external_id"],
            )
            self._notify(
                entry_id, person["name"],
                f"{person['role']} - {person['department']} - MATCHED "
                f"{score * 100:.1f}%",
                score, "MATCHED", frame, face,
            )
            return

        # unknown / below threshold: never guess an identity
        entry_id = self.db.add_entry(
            name="Unknown", result="UNKNOWN", confidence=score,
            snapshot=self._save_snapshot(frame, face),
        )
        self._notify(entry_id, "Unknown visitor",
                     "Unknown / Manual Verification Required", score,
                     "UNKNOWN", frame, face)

    def _cooldown(self, key: str, now: float) -> bool:
        """True when this person/unknown may be logged again (cooldown passed)."""
        last = self._last_logged.get(key)
        return last is None or (now - last) >= COOLDOWN_SECONDS

    def _notify(self, entry_id, name, message, score, result, frame, face):
        alert_id = self.db.add_alert(entry_id)
        self.status["last_event"] = {
            "alert_id": alert_id,
            "entry_id": entry_id,
            "name": name,
            "result": result,
            "confidence": round(float(score), 4),
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        show_toast("VC Cabin Visitor Alert", message, enabled=self.toast)

    def _save_snapshot(self, frame, face) -> str:
        x, y, w, h = self.detector.box(face)
        margin = int(max(w, h) * 0.25)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w_img, x + w + margin), min(h_img, y + h + margin)
        crop = frame[y1:y2, x1:x2]
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"entry_{stamp}.jpg"
        cv2.imwrite(os.path.join(SNAPSHOT_DIR, filename), crop)
        self._prune_snapshots()
        return filename

    @staticmethod
    def _prune_snapshots():
        try:
            files = sorted(
                f for f in os.listdir(SNAPSHOT_DIR) if f.startswith("entry_")
            )
            for old in files[:-SNAPSHOT_KEEP]:
                os.remove(os.path.join(SNAPSHOT_DIR, old))
        except OSError:
            pass
