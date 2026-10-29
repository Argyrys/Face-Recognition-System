"""Test fixtures: point the app at a throwaway DB/snapshots/logs BEFORE the
app module is imported (it builds its Database at import time)."""

import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="vc_test_")
os.environ["VC_DB_PATH"] = os.path.join(_TMP, "test.db")
os.environ["VC_SNAPSHOT_DIR"] = os.path.join(_TMP, "snapshots")
os.environ["VC_LOG_FILE"] = os.path.join(_TMP, "test.log")
os.environ["VC_ADMIN_PASSWORD"] = "unit-test-pw-123"
os.environ["VC_SECRET_KEY"] = "unit-test-secret"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import app as appmod  # noqa: E402


@pytest.fixture()
def client():
    appmod.app.config["TESTING"] = True
    with appmod.app.test_client() as c:
        yield c


@pytest.fixture()
def admin(client):
    """A client already logged in as the admin operator."""
    resp = client.post("/login", data={"password": "unit-test-pw-123"})
    assert resp.status_code == 302, resp.data
    return client


def audit_actions(c):
    """Helper: all audit action names, newest first."""
    return [r["action"] for r in c.get("/api/audit").get_json()]
