"""Tests for the security-appliance layer: tracker, liveness, threshold
sweep, alerter, backup, schema migration, and response hardening."""

import json
import os
import sqlite3
import zipfile

import numpy as np
import pytest

from backup import backup, prune, verify
from database import Database
from liveness import (blink_count, decide, eye_energy, motion_score,
                      normalized_landmarks)
from threshold_sweep import summarize
from tracker import MultiTracker, Track, iou
from alerter import Alerter, WebhookChannel


# ------------------------------------------------------------------ helpers

def face_row(x, y, w=100, h=120):
    """Synthetic YuNet detection row (x, y, w, h + 5 landmarks + score)."""
    lm = []
    for fx, fy in ((0.3, 0.35), (0.7, 0.35), (0.5, 0.55),
                   (0.35, 0.8), (0.65, 0.8)):
        lm += [x + fx * w, y + fy * h]
    return np.array([x, y, w, h] + lm + [0.9], dtype=np.float32)


def jitter_face(base, dx=0, dy=0):
    """Move landmarks INSIDE a fixed box (expression / rotation motion)."""
    row = base.copy()
    for i in range(4, 14, 2):
        row[i] += dx
        row[i + 1] += dy
    return row


# -------------------------------------------------------------------- iou

def test_iou_geometry():
    a = (0, 0, 10, 10)
    assert iou(a, a) == pytest.approx(1.0)
    assert iou(a, (20, 20, 10, 10)) == 0.0
    # half overlap in x on identical squares -> 1/3
    assert iou(a, (5, 0, 10, 10)) == pytest.approx(1 / 3)


# ------------------------------------------------------------------ tracker

def test_tracker_stable_id_across_frames():
    tr = MultiTracker()
    base = face_row(50, 50)
    first = tr.update([base], 1)
    assert len(first) == 1
    tid = first[0].id
    for f in range(2, 6):
        tracks = tr.update([jitter_face(base, dx=f)], f)
        assert [t.id for t in tracks] == [tid]
    assert tracks[0].hits == 5


def test_tracker_second_face_gets_new_id():
    tr = MultiTracker()
    a, b = face_row(0, 0), face_row(400, 300)
    first = tr.update([a], 1)
    second = tr.update([a, b], 2)
    ids = sorted(t.id for t in second)
    assert ids == [first[0].id, max(ids)]
    assert len(set(ids)) == 2


def test_tracker_retires_long_missed_face():
    tr = MultiTracker(max_missed=3)
    tr.update([face_row(0, 0)], 1)
    for f in range(2, 20):
        tracks = tr.update([], f)
    assert tracks == []


# -------------------------------------------------- confirmation streaking

def make_track():
    return Track(1, face_row(10, 10), 1)


def test_match_needs_consecutive_agreement():
    tr = make_track()
    tr.record(7, 0.8)
    tr.record(7, 0.82)
    assert not tr.ready_to_log(3)          # only 2 agreeing frames
    tr.record(7, 0.79)
    assert tr.ready_to_log(3)
    assert not tr.ready_unknown(3)


def test_disagreement_resets_the_streak():
    tr = make_track()
    tr.record(7, 0.8)
    tr.record(7, 0.8)
    tr.record(None, 0.2)                   # flicker breaks the streak
    assert tr.unknown_streak == 1          # and starts an unknown streak
    tr.record(7, 0.8)
    tr.record(7, 0.8)
    assert not tr.ready_to_log(3)          # streak restarted from 1
    assert tr.unknown_streak == 0


def test_unknown_streak_and_single_emit():
    tr = make_track()
    for _ in range(3):
        tr.record(None, 0.1)
    assert tr.ready_unknown(3)
    tr.logged = True                       # pipeline marks it emitted
    assert not tr.ready_unknown(3)
    assert not tr.ready_to_log(3)


# ---------------------------------------------------------------- liveness

def test_static_face_is_spoof_but_not_prematurely():
    tr = make_track()
    row = face_row(100, 100)
    constant_energy = (1.2, 1.1)
    for _ in range(4):                     # enough for 'live', not 'spoof'
        tr.landmarks.append(normalized_landmarks(row))
        tr.energies.append(constant_energy)
    assert tr.liveness_state() == "unknown"
    for _ in range(12):                    # ~1 s of a perfectly still face
        tr.landmarks.append(normalized_landmarks(row))
        tr.energies.append(constant_energy)
    assert tr.liveness_state() == "spoof"


def test_moving_blinking_face_is_live():
    tr = make_track()
    base = face_row(100, 100)
    rng = np.random.default_rng(42)
    for i in range(20):
        row = jitter_face(base, dx=float(rng.normal(0, 4)),
                          dy=float(rng.normal(0, 4)))
        tr.landmarks.append(normalized_landmarks(row))
        tr.energies.append((1.2, 1.1) if i not in (7, 8)
                           else (0.3, 0.3))   # a blink dip
    assert tr.liveness_state() == "live"


def test_decide_matrix():
    assert decide(0.001, 0) == "spoof"
    assert decide(0.010, 1) == "live"
    assert decide(0.010, 0) == "live"       # motion >= 2x floor
    assert decide(0.003, 1) == "unknown"    # blink but sub-floor motion
    assert decide(0.003, 0) == "spoof"


def test_blink_detection_on_energy_history():
    hist = [(1.0, 1.0)] * 5 + [(0.2, 0.2), (0.2, 0.2)] + [(1.0, 1.0)] * 5
    assert blink_count(hist) == 1
    assert blink_count([(1.0, 1.0)] * 10) == 0


def test_eye_energy_sees_closed_eyes():
    """Flat eye region (closed lids) yields less texture than open eyes."""
    rng = np.random.default_rng(7)
    open_eyes = rng.integers(0, 255, (120, 120), dtype=np.uint8)
    closed = open_eyes.copy()
    # eyes are at 30% / 70% width, 35% height; smooth the eye boxes
    for cx in (36, 84):
        closed[28:50, cx - 12:cx + 12] = np.median(closed)  # smooth = closed
    f_open = face_row(10, 10)
    f_closed = face_row(10, 10)
    e_open = sum(eye_energy(open_eyes, f_open))
    e_closed = sum(eye_energy(closed, f_closed))
    assert e_closed < e_open


def test_landmark_normalization_is_scale_invariant():
    small = normalized_landmarks(face_row(0, 0, w=50, h=60))
    large = normalized_landmarks(face_row(0, 0, w=500, h=600))
    assert np.allclose(small, large)


def test_motion_score_grows_with_movement():
    still = [normalized_landmarks(face_row(0, 0))] * 5
    moving = [normalized_landmarks(jitter_face(face_row(0, 0), dx=i * 6,
                                               dy=i * 6))
              for i in range(5)]
    assert motion_score(moving) > motion_score(still)
    # gross translation cancels out by design (distance-invariance)
    translated = [normalized_landmarks(face_row(i * 40, i * 40))
                  for i in range(5)]
    assert motion_score(translated) == pytest.approx(0.0)


# ---------------------------------------------------------- threshold sweep

def test_sweep_on_perfectly_separated_scores():
    result = summarize(pos=[0.80, 0.85, 0.90], neg=[0.10, 0.15, 0.20],
                       thresholds=[0.10, 0.30, 0.50, 0.70], max_far=0.01)
    sec = result["security"]
    assert sec["far"] == 0.0
    assert sec["recall"] == 1.0
    lo, hi = result["security_interval"]
    assert lo <= sec["threshold"] <= hi
    # midpoint logic must not recommend an edge of the safe interval
    assert (lo, hi) == (0.30, 0.70)
    assert sec["threshold"] == 0.50         # middle of the tied pool


def test_sweep_reports_overlap_honestly():
    result = summarize(pos=[0.5, 0.6], neg=[0.55, 0.1],
                       thresholds=[0.2, 0.4, 0.6], max_far=0.5)
    rows = {r["threshold"]: r for r in result["rows"]}
    assert rows[0.6]["far"] == 0.0          # strict threshold: no false accepts
    assert rows[0.6]["frr"] == 0.5          # ...but genuine pairs rejected
    assert rows[0.2]["far"] == 0.5


# ------------------------------------------------------------------ alerter

class FakeChannel:
    name = "fake"

    def __init__(self, min_interval=60):
        self.min_interval = min_interval
        self.sent = []
        self._last = 0.0

    def ready(self, level):
        import time
        if level == "critical":
            return True
        if time.time() - self._last < self.min_interval:
            return False
        self._last = time.time()
        return True

    def send(self, title, body, level):
        self.sent.append((title, body, level))


def test_alerter_rate_limits_but_critical_bypasses():
    ch = FakeChannel(min_interval=3600)
    al = Alerter([ch])
    assert al.send("t1", "b1", "info") == 1
    assert al.send("t2", "b2", "info") == 0    # rate-limited
    assert al.send("t3", "b3", "critical") == 1
    assert [s[2] for s in ch.sent] == ["info", "critical"]


def test_alerter_never_raises_on_channel_failure():
    class Broken(FakeChannel):
        def send(self, *a):
            raise RuntimeError("smtp down")
    al = Alerter([Broken()])
    assert al.send("t", "b", "info") == 0


def test_webhook_payload_shapes(monkeypatch):
    captured = {}

    class FakeResp:
        def read(self):
            return b""

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=10):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return FakeResp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    WebhookChannel("https://discord.com/api/webhooks/1/x", 0) \
        .send("Hi", "Body", "info")
    assert captured["body"] == {"content": "**Hi**\nBody"}

    WebhookChannel("https://hooks.example.com/x", 0) \
        .send("Door", "Body", "critical")
    assert captured["body"] == {
        "title": "Door", "body": "Body", "level": "critical",
        "source": "VC Cabin"}


def test_build_alerter_is_noop_without_configuration():
    from alerter import build_alerter
    assert build_alerter().armed == []


# ------------------------------------------------------------------- backup

def test_backup_creates_verified_archive(tmp_path):
    db_path = os.path.join(str(tmp_path), "src.db")
    db = Database(db_path)
    db.add_entry("Ada", "MATCHED", 0.9, None)

    import backup as backupmod
    # point the module at this throwaway DB
    old_db, old_snap = backupmod.DB_PATH, backupmod.SNAP_DIR
    backupmod.DB_PATH, backupmod.SNAP_DIR = db_path, str(tmp_path / "none")
    try:
        out = backup(out_dir=str(tmp_path / "out"), keep=5)
        with zipfile.ZipFile(out) as zf:
            names = zf.namelist()
            assert "vc_entry.db" in names
            assert "manifest.json" in names
            manifest = json.loads(zf.read("manifest.json"))
            assert manifest["counts"]["entry_log"] == 1
        assert verify(db_path) is True
    finally:
        backupmod.DB_PATH, backupmod.SNAP_DIR = old_db, old_snap


def test_backup_prune_keeps_newest(tmp_path):
    for i in range(4):
        (tmp_path / f"vc_backup_2020010{i}_000000.zip").write_bytes(b"x")
    removed = prune(str(tmp_path), keep=2)
    assert removed == 2
    left = sorted(f.name for f in tmp_path.iterdir())
    assert left == ["vc_backup_20200102_000000.zip",
                    "vc_backup_20200103_000000.zip"]


# --------------------------------------------------------- schema migration

LEGACY_SCHEMA = """
CREATE TABLE persons (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL,
    role TEXT, department TEXT, external_id TEXT, embedding BLOB NOT NULL,
    photo TEXT, created_at TEXT, status TEXT DEFAULT 'active', version INTEGER
);
CREATE TABLE entry_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, person_id INTEGER, name TEXT,
    role TEXT, department TEXT, external_id TEXT, recognized_at TEXT,
    confidence REAL, result TEXT, snapshot TEXT
);
CREATE TABLE alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, entry_id INTEGER, acked INTEGER DEFAULT 0,
    created_at TEXT, alert_key TEXT, acked_at TEXT
);
INSERT INTO alerts (entry_id, acked) VALUES (1, 1);
INSERT INTO entry_log (name, result, confidence) VALUES ('Ada', 'MATCHED', 0.9);
"""


def test_legacy_database_gains_liveness_column(tmp_path):
    path = str(tmp_path / "legacy.db")
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    conn.close()

    db = Database(path)
    row = db._conn.execute(
        "SELECT liveness FROM entry_log WHERE name='Ada'").fetchone()
    assert row["liveness"] is None
    decision = db._conn.execute(
        "SELECT decision FROM alerts WHERE acked=1").fetchone()
    assert decision["decision"] == "acknowledged"   # backfill

    eid = db.add_entry("Bo", "UNKNOWN", 0.3, None, liveness="spoof")
    got = db._conn.execute(
        "SELECT liveness FROM entry_log WHERE id=?", (eid,)).fetchone()
    assert got["liveness"] == "spoof"


# ------------------------------------------------------------ hardening

def test_security_headers_on_every_response(client):
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "SAMEORIGIN"
    assert resp.headers["Referrer-Policy"] == "same-origin"


def test_api_responses_are_not_cached(client, admin):
    resp = admin.get("/api/status")
    assert resp.status_code == 200
    assert resp.headers["Cache-Control"] == "no-store"


def test_status_reports_security_layer(client, admin):
    s = admin.get("/api/status").get_json()
    for key in ("liveness", "alerts_armed", "camera_age", "degraded"):
        assert key in s, key
