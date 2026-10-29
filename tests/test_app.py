"""End-to-end tests for the production security layer: login gate, audit
trail, retention purge, CSRF defence. The heavy vision engine is stubbed —
these tests cover the web app, not model accuracy."""

import datetime
import io
import os
import sqlite3

import numpy as np
import pytest

import app as appmod
from conftest import audit_actions


@pytest.fixture(autouse=True)
def _clear_lockouts():
    appmod.LOGIN_ATTEMPTS.clear()
    yield
    appmod.LOGIN_ATTEMPTS.clear()


class StubPipeline:
    """Stands in for the real Pipeline so toggle tests need no camera."""

    def __init__(self):
        self.status = {"recognition": "on", "running": False}

    def set_enabled(self, on):
        self.status["recognition"] = "on" if on else "off"

    def enabled(self):
        return self.status["recognition"] == "on"


# ------------------------------------------------------------------- gate

def test_health_is_public(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_login_page_is_public(client):
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"VC Cabin" in resp.data


def test_pages_redirect_when_logged_out(client):
    for path in ("/", "/enroll"):
        resp = client.get(path)
        assert resp.status_code == 302
        assert "/login" in resp.headers["Location"]


def test_apis_return_401_when_logged_out(client):
    for path in ("/api/status", "/api/alerts", "/api/entries", "/api/audit"):
        resp = client.get(path)
        assert resp.status_code == 401, path
    resp = client.post("/api/recognition", json={"enabled": False})
    assert resp.status_code == 401


def test_camera_stream_requires_login(client):
    assert client.get("/api/camera-stream").status_code == 401


# ------------------------------------------------------------------- auth

def test_wrong_password_is_rejected_and_audited(client):
    resp = client.post("/login", data={"password": "nope"})
    assert resp.status_code == 200
    assert b"Incorrect password" in resp.data
    actions = [r["action"] for r in appmod.db.recent_audit()]
    assert "login.failed" in actions
    assert client.get("/").status_code == 302   # still not authenticated


def test_login_lockout_after_repeated_failures(client):
    for _ in range(appmod.MAX_ATTEMPTS):
        client.post("/login", data={"password": "nope"})
    resp = client.post("/login", data={"password": "unit-test-pw-123"})
    assert b"Too many failed attempts" in resp.data


def test_login_success_and_logout(admin):
    resp = admin.get("/")
    assert resp.status_code == 200
    assert b"VC Cabin" in resp.data
    # already-authed visits to /login bounce to the dashboard
    assert admin.get("/login").status_code == 302
    actions = audit_actions(admin)
    assert "login.success" in actions
    assert admin.get("/logout").status_code == 302
    assert admin.get("/").status_code == 302    # logged out again


# ------------------------------------------------------- csrf + auditing

def test_cross_site_mutations_are_blocked(admin):
    resp = admin.post("/api/recognition", json={"enabled": False},
                      headers={"Sec-Fetch-Site": "cross-site"})
    assert resp.status_code == 403


def test_recognition_toggle_writes_audit(admin, monkeypatch):
    monkeypatch.setattr(appmod, "pipeline", StubPipeline())
    resp = admin.post("/api/recognition", json={"enabled": False})
    assert resp.get_json() == {"ok": True, "recognition": "off"}
    assert "recognition.toggle" in audit_actions(admin)


def test_ack_alert_writes_audit(admin):
    entry = appmod.db.add_entry(name="Test Visitor", result="MATCHED",
                                confidence=0.9, snapshot=None)
    alert = appmod.db.add_alert(entry)
    assert admin.post(f"/api/alerts/{alert}/ack").get_json()["ok"] is True
    assert "alert.ack" in audit_actions(admin)
    pending = [a["alert_id"] for a in admin.get("/api/alerts").get_json()]
    assert alert not in pending


def test_reject_alert_writes_audit(admin):
    entry = appmod.db.add_entry(name="Stranger", result="UNKNOWN",
                                confidence=0.4, snapshot=None)
    alert = appmod.db.add_alert(entry)
    assert admin.post(f"/api/alerts/{alert}/reject").get_json()["ok"] is True
    resolved = admin.get("/api/resolved").get_json()
    assert any(r["alert_id"] == alert and r["decision"] == "rejected"
               for r in resolved)
    assert "alert.reject" in audit_actions(admin)


def test_deactivate_person_writes_audit(admin):
    pid = appmod.db.add_person("Stub Kid", "Student", "CSE", "S1",
                               np.zeros(128, dtype=np.float32), True)
    resp = admin.post(f"/api/persons/{pid}/deactivate")
    assert resp.get_json()["ok"] is True
    assert appmod.db.person(pid)["status"] == "inactive"
    assert "person.deactivate" in audit_actions(admin)


def test_enroll_requires_consent(admin, monkeypatch):
    monkeypatch.setattr(appmod, "_engine", lambda: (None, None))
    resp = admin.post("/enroll", data={
        "name": "Test Kid", "role": "Student", "department": "CSE",
        "external_id": "S-42",
        "photos": (io.BytesIO(b"dummy"), "face.jpg"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "consent" in resp.headers["Location"].lower()


def test_enroll_success_is_audited(admin, monkeypatch):
    monkeypatch.setattr(appmod, "_engine", lambda: (None, None))
    monkeypatch.setattr(
        appmod, "enroll",
        lambda *a, **k: {"id": 77, "snapshot": "person_77.jpg", "images": 1},
    )
    resp = admin.post("/enroll", data={
        "name": "Test Kid", "role": "Student", "department": "CSE",
        "external_id": "S-42", "consent": "on",
        "photos": (io.BytesIO(b"dummy"), "face.jpg"),
    }, content_type="multipart/form-data")
    assert resp.status_code == 302
    assert "enrolled" in resp.headers["Location"].lower()
    assert "person.enroll" in audit_actions(admin)


# -------------------------------------------------------------- retention

def test_retention_purge_removes_expired_rows(admin, tmp_path):
    old_id = appmod.db.add_entry(name="Old", result="UNKNOWN",
                                 confidence=0.4, snapshot=None)
    old_alert = appmod.db.add_alert(old_id)
    fresh_id = appmod.db.add_entry(name="New", result="MATCHED",
                                   confidence=0.9, snapshot=None)

    cutoff = (datetime.datetime.now() - datetime.timedelta(days=40)).isoformat(
        timespec="seconds")
    conn = sqlite3.connect(os.environ["VC_DB_PATH"])
    conn.execute("UPDATE entry_log SET recognized_at=? WHERE id=?",
                 (cutoff, old_id))
    conn.execute("UPDATE alerts SET created_at=? WHERE id=?", (cutoff, old_alert))
    conn.commit()
    conn.close()

    removed = appmod.db.purge_older_than(30, audit_days=90)
    assert removed["entries"] >= 1
    assert removed["alerts"] >= 1
    remaining = {e["id"] for e in appmod.db.entries(limit=200)}
    assert old_id not in remaining
    assert fresh_id in remaining


def test_retention_purge_removes_old_snapshots(admin):
    snaps = os.environ["VC_SNAPSHOT_DIR"]
    os.makedirs(snaps, exist_ok=True)
    old_path = os.path.join(snaps, "entry_20200101_000000_000000.jpg")
    with open(old_path, "wb") as f:
        f.write(b"\xff\xd8\xff")
    stale = time_seconds_ago(appmod.Config.RETENTION_DAYS * 86400 + 3600)
    os.utime(old_path, (stale, stale))

    appmod.run_retention()
    assert not os.path.exists(old_path)


def time_seconds_ago(seconds):
    import time
    return time.time() - seconds
