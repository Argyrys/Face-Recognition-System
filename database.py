"""SQLite storage: enrolled persons, entry log, alerts, audit trail, settings."""

import os
import sqlite3
import threading
import datetime

import numpy as np

from config import Config

DB_PATH = Config.DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS persons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('Faculty', 'Student')),
    department  TEXT NOT NULL,
    external_id TEXT NOT NULL,
    embedding   BLOB NOT NULL,
    consent     INTEGER NOT NULL DEFAULT 0,
    status      TEXT NOT NULL DEFAULT 'active',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entry_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   INTEGER,
    name        TEXT NOT NULL,
    role        TEXT,
    department  TEXT,
    external_id TEXT,
    recognized_at TEXT NOT NULL,
    confidence  REAL,
    result      TEXT NOT NULL CHECK (result IN ('MATCHED', 'UNKNOWN')),
    snapshot    TEXT,
    FOREIGN KEY (person_id) REFERENCES persons (id)
);
CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id   INTEGER NOT NULL,
    acked      INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (entry_id) REFERENCES entry_log (id)
);
CREATE TABLE IF NOT EXISTS audit_log (
    id     INTEGER PRIMARY KEY AUTOINCREMENT,
    at     TEXT NOT NULL,
    actor  TEXT NOT NULL DEFAULT 'admin',
    action TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_time ON audit_log(at);
CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# Alerts older than this vanish from notifications without being acked.
# ISO "T" separator matches _now() so string comparison is correct.
_ALERT_TTL = "replace(datetime('now','localtime','-1 day'), ' ', 'T')"


class Database:
    def __init__(self, path: str = DB_PATH):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._version = 0

    def _migrate(self):
        """Add the alert decision column and backfill older rows."""
        cols = {r["name"] for r in self._conn.execute("PRAGMA table_info(alerts)")}
        if "decision" not in cols:
            self._conn.execute("ALTER TABLE alerts ADD COLUMN decision TEXT")
            # alerts already acked before decisions existed count as acknowledged
            self._conn.execute(
                "UPDATE alerts SET decision = 'acknowledged' WHERE acked = 1"
            )
            self._conn.commit()

    # -- persons -----------------------------------------------------------

    @property
    def version(self) -> int:
        """Bumped on person changes; pipeline uses it to reload the gallery."""
        return self._version

    def add_person(self, name, role, department, external_id, embedding,
                   consent: bool) -> int:
        blob = np.asarray(embedding, dtype=np.float32).tobytes()
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO persons (name, role, department, external_id,"
                " embedding, consent, created_at) VALUES (?,?,?,?,?,?,?)",
                (name.strip(), role, department.strip(), external_id.strip(),
                 blob, 1 if consent else 0, _now()),
            )
            self._conn.commit()
            self._version += 1
            return int(cur.lastrowid)

    def persons(self, active_only=True):
        sql = "SELECT id, name, role, department, external_id, consent, status, created_at FROM persons"
        if active_only:
            sql += " WHERE status = 'active'"
        sql += " ORDER BY id"
        return [dict(r) for r in self._conn.execute(sql).fetchall()]

    def person(self, person_id: int):
        row = self._conn.execute(
            "SELECT id, name, role, department, external_id, status"
            " FROM persons WHERE id = ?",
            (person_id,),
        ).fetchone()
        return dict(row) if row else None

    def gallery(self):
        """[(person_id, normalized embedding ndarray)] for active persons."""
        out = []
        rows = self._conn.execute(
            "SELECT id, embedding FROM persons WHERE status = 'active'"
        ).fetchall()
        for r in rows:
            emb = np.frombuffer(r["embedding"], dtype=np.float32)
            out.append((r["id"], emb))
        return out

    def deactivate_person(self, person_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE persons SET status = 'inactive' WHERE id = ?",
                (person_id,),
            )
            self._conn.commit()
            if cur.rowcount:
                self._version += 1
            return bool(cur.rowcount)

    # -- entry log ---------------------------------------------------------

    def add_entry(self, name, result, confidence, snapshot,
                  person_id=None, role=None, department=None, external_id=None):
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO entry_log (person_id, name, role, department,"
                " external_id, recognized_at, confidence, result, snapshot)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (person_id, name, role, department, external_id, _now(),
                 confidence, result, snapshot),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def entries(self, limit=50):
        rows = self._conn.execute(
            "SELECT * FROM entry_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- alerts ------------------------------------------------------------

    def add_alert(self, entry_id: int) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO alerts (entry_id, created_at) VALUES (?, ?)",
                (entry_id, _now()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def pending_alerts(self):
        rows = self._conn.execute(
            "SELECT a.id AS alert_id, a.acked, a.created_at AS alert_at,"
            " e.* FROM alerts a JOIN entry_log e ON e.id = a.entry_id"
            " WHERE a.decision IS NULL AND a.created_at >= " + _ALERT_TTL +
            " ORDER BY a.id DESC LIMIT 20"
        ).fetchall()
        return [dict(r) for r in rows]

    def resolved_alerts(self, limit=20):
        rows = self._conn.execute(
            "SELECT a.id AS alert_id, a.decision, a.created_at AS alert_at,"
            " e.* FROM alerts a JOIN entry_log e ON e.id = a.entry_id"
            " WHERE a.decision IS NOT NULL"
            " ORDER BY a.id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def resolve_alert(self, alert_id: int, decision: str) -> bool:
        """decision: 'acknowledged' or 'rejected'. Both clear the notification."""
        if decision not in ("acknowledged", "rejected"):
            raise ValueError(f"unknown decision: {decision}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE alerts SET acked = 1, decision = ? WHERE id = ?",
                (decision, alert_id),
            )
            self._conn.commit()
            return bool(cur.rowcount)

    def ack_alert(self, alert_id: int) -> bool:
        return self.resolve_alert(alert_id, "acknowledged")

    def reject_alert(self, alert_id: int) -> bool:
        return self.resolve_alert(alert_id, "rejected")

    def ack_all(self) -> int:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE alerts SET acked = 1, decision = 'acknowledged'"
                " WHERE decision IS NULL"
            )
            self._conn.commit()
            return cur.rowcount

    # -- stats -------------------------------------------------------------

    def stats(self):
        q = lambda sql: self._conn.execute(sql).fetchone()[0]
        return {
            "persons": q("SELECT COUNT(*) FROM persons WHERE status='active'"),
            "entries_today": q(
                "SELECT COUNT(*) FROM entry_log WHERE date(recognized_at) = date('now','localtime')"
            ),
            "entries_total": q("SELECT COUNT(*) FROM entry_log"),
            "recognized_today": q(
                "SELECT COUNT(*) FROM entry_log WHERE result='MATCHED'"
                " AND date(recognized_at) = date('now','localtime')"
            ),
            "students": q(
                "SELECT COUNT(*) FROM persons"
                " WHERE status='active' AND role='Student'"
            ),
            "staff": q(
                "SELECT COUNT(*) FROM persons"
                " WHERE status='active' AND role='Faculty'"
            ),
            "pending_alerts": q(
                "SELECT COUNT(*) FROM alerts WHERE decision IS NULL"
                " AND created_at >= " + _ALERT_TTL
            ),
            "rejected_today": q(
                "SELECT COUNT(*) FROM alerts WHERE decision = 'rejected'"
                " AND date(created_at) = date('now','localtime')"
            ),
        }

    # -- audit trail --------------------------------------------------------

    def audit(self, action: str, detail: str = "", actor: str = "admin") -> None:
        """Append-only record of operator actions (who did what, when)."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO audit_log (at, actor, action, detail) VALUES (?,?,?,?)",
                (_now(), actor, action, detail),
            )
            self._conn.commit()

    def recent_audit(self, limit=100):
        rows = self._conn.execute(
            "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- settings (session key / password hash persistence) -----------------

    def get_setting(self, key, default=None):
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key, value):
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self._conn.commit()

    # -- retention ----------------------------------------------------------

    def purge_older_than(self, days: int, audit_days: int = 90) -> dict:
        """Delete entries, their alerts, and old audit rows past the retention
        window. Returns the number of rows removed per table."""
        cutoff = (datetime.datetime.now() - datetime.timedelta(days=days)).isoformat(
            timespec="seconds"
        )
        audit_cutoff = (
            datetime.datetime.now() - datetime.timedelta(days=audit_days)
        ).isoformat(timespec="seconds")
        with self._lock:
            # alerts go first so no notification outlives its entry record
            cur = self._conn.execute(
                "DELETE FROM alerts WHERE id IN ("
                " SELECT a.id FROM alerts a JOIN entry_log e ON e.id = a.entry_id"
                " WHERE e.recognized_at < ?)",
                (cutoff,),
            )
            alerts = cur.rowcount
            cur = self._conn.execute(
                "DELETE FROM entry_log WHERE recognized_at < ?", (cutoff,)
            )
            entries = cur.rowcount
            cur = self._conn.execute(
                "DELETE FROM audit_log WHERE at < ?", (audit_cutoff,)
            )
            audit = cur.rowcount
            self._conn.commit()
        return {"entries": entries, "alerts": alerts, "audit": audit}
