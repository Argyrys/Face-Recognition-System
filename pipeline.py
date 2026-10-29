"""Background recognition pipeline: frames -> detect -> track -> confirm ->
liveness gate -> match -> log -> notify.

Security changes vs. the original single-frame pipeline:
- every face is TRACKED across frames (tracker.py), not judged per frame;
- a MATCHED entry needs CONFIRM consecutive frames agreeing on the identity
  (N-frame confirmation) — one blurred frame can no longer log a wrong name;
- liveness (liveness.py) gates matches on live sources: a photo or still
  screen is recorded as UNKNOWN/PRESENTATION ATTACK, never MATCHED;
- unknown visitors get the same N-frame treatment, so detection flicker
  cannot spam the alert list.
"""

import datetime
import os
import threading
import time

import cv2
import numpy as np

from camera import FrameSource
from config import Config
from database import Database
from enrollment import SNAPSHOT_DIR
from face_detection import FaceDetector
from face_recognition import FaceRecognizer, best_match
from liveness import entry_label
from notification import show_toast
from tracker import MultiTracker

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

        # -- security layer ------------------------------------------------
        self.tracker = MultiTracker()
        self.confirm_frames = max(1, Config.CONFIRM_FRAMES)
        self.unknown_frames = max(1, Config.UNKNOWN_FRAMES)
        self.min_cosine = float(Config.MIN_COSINE)
        self.liveness_mode = str(Config.LIVENESS).lower()
        if self.liveness_mode not in ("enforced", "monitor", "off"):
            self.liveness_mode = "enforced"
        # recorded video / image folders ARE replay material by definition;
        # liveness enforcement only makes sense on a live camera.
        self.live_source = self.source.label.startswith(("webcam", "stream"))
        if self.liveness_mode == "enforced" and not self.live_source:
            self.liveness_mode = "monitor"

        self.status = {
            "running": False,
            "state": "starting",
            "recognition": "on",
            "liveness": self.liveness_mode,
            "source": self.source.label,
            "frames": 0,
            "detections": 0,
            "tracks": 0,
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

        frame_idx = self.status["frames"]
        faces = self.detector.detect(frame)
        if faces is not None and len(faces):
            self.status["detections"] += len(faces)
        gray = None
        if self.liveness_mode != "off":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        tracks = self.tracker.update(faces, frame_idx, gray)
        self.status["tracks"] = len(tracks)

        for tr in tracks:
            embedding = self.recognizer.embed(frame, tr.face)
            person_id, score = best_match(embedding, self._gallery,
                                          threshold=self.min_cosine)
            tr.record(person_id, score)
            self._maybe_log(frame, tr)

    def _maybe_log(self, frame, tr):
        """Emit at most one entry per track, and only on sufficient evidence."""
        if tr.logged:
            return
        state = ("unknown" if self.liveness_mode == "off"
                 else tr.liveness_state(Config.LIVENESS_MIN_MOTION))

        person = None
        if tr.ready_to_log(self.confirm_frames):
            candidate = self.db.person(tr.top_person)
            if candidate is not None and candidate["status"] == "active":
                person = candidate

        if person is not None:
            if self.liveness_mode == "enforced" and state != "live":
                if state == "spoof":
                    # a photo/screen of an ENROLLED person: reject loudly
                    tr.logged = True
                    self._log_spoof(frame, tr, state, of_person=person)
                    return
                # not enough evidence yet: give the visitor a moment
                tr.hold_frames += 1
                if tr.hold_frames <= Config.LIVENESS_HOLD_FRAMES:
                    return
                tr.logged = True
                self._log_unknown(frame, tr, "unknown",
                                  note="liveness unverified")
                return
            key = f"person:{person['id']}"
            now = time.monotonic()
            if not self._cooldown(key, now):
                tr.logged = True
                return
            self._last_logged[key] = now
            tr.logged = True
            self._log_matched(frame, tr, person, state)
            return

        if tr.ready_unknown(self.unknown_frames):
            now = time.monotonic()
            if not self._cooldown("unknown", now):
                tr.logged = True
                return
            self._last_logged["unknown"] = now
            tr.logged = True
            if self.liveness_mode == "enforced" and state == "spoof":
                self._log_spoof(frame, tr, state)
            else:
                self._log_unknown(frame, tr, state)

    # ------------------------------------------------------------ emitters

    def _log_matched(self, frame, tr, person, state):
        score = tr.last_score
        label = entry_label(self.liveness_mode, state)
        entry_id = self.db.add_entry(
            name=person["name"], result="MATCHED", confidence=score,
            snapshot=self._save_snapshot(frame, tr.box),
            person_id=person["id"], role=person["role"],
            department=person["department"],
            external_id=person["external_id"], liveness=label,
        )
        self._notify(
            entry_id, person["name"],
            f"{person['role']} - {person['department']} - MATCHED "
            f"{score * 100:.1f}%",
            score, "MATCHED", liveness=label,
        )

    def _log_unknown(self, frame, tr, state, note=None):
        label = entry_label(self.liveness_mode, state)
        score = tr.last_score
        entry_id = self.db.add_entry(
            name="Unknown", result="UNKNOWN", confidence=score,
            snapshot=self._save_snapshot(frame, tr.box), liveness=label,
        )
        message = note or "Unknown / Manual Verification Required"
        self._notify(entry_id, "Unknown visitor", message, score, "UNKNOWN",
                     liveness=label)

    def _log_spoof(self, frame, tr, state, of_person=None):
        """Presentation attack: photo/screen/replay in front of the camera."""
        label = entry_label(self.liveness_mode, state)
        score = tr.last_score
        entry_id = self.db.add_entry(
            name="Presentation attack", result="UNKNOWN", confidence=score,
            snapshot=self._save_snapshot(frame, tr.box), liveness=label,
        )
        detail = (f"possible replay of enrolled person {of_person['name']}"
                  if of_person else "unidentified subject")
        self._notify(
            entry_id, "Suspected presentation attack",
            f"PHOTO/REPLAY ATTACK? - {detail} - manual verification required",
            score, "UNKNOWN", liveness=label, critical=True,
        )

    def _cooldown(self, key: str, now: float) -> bool:
        """True when this person/unknown may be logged again (cooldown passed)."""
        last = self._last_logged.get(key)
        return last is None or (now - last) >= COOLDOWN_SECONDS

    def _notify(self, entry_id, name, message, score, result,
                liveness=None, critical=False):
        alert_id = self.db.add_alert(entry_id)
        self.status["last_event"] = {
            "alert_id": alert_id,
            "entry_id": entry_id,
            "name": name,
            "result": result,
            "liveness": liveness,
            "critical": bool(critical),
            "confidence": round(float(score), 4),
            "at": datetime.datetime.now().isoformat(timespec="seconds"),
        }
        show_toast("VC Cabin Visitor Alert", message, enabled=self.toast)

    def _save_snapshot(self, frame, box) -> str:
        x, y, w, h = (int(v) for v in box)
        margin = int(max(w, h) * 0.25)
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w_img, x + w + margin), min(h_img, y + h + margin)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
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
