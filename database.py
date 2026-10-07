"""SQLite storage: enrolled persons, entry log, dashboard alerts."""

import os
import sqlite3
import threading
import datetime

import numpy as np

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "database", "vc_entry.db"
)

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
"""


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str = DB_PATH):
        if path != ":memory:":
            os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._version = 0

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
            " WHERE a.acked = 0 ORDER BY a.id DESC LIMIT 20"
        ).fetchall()
        return [dict(r) for r in rows]

    def ack_alert(self, alert_id: int) -> bool:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE alerts SET acked = 1 WHERE id = ?", (alert_id,)
            )
            self._conn.commit()
            return bool(cur.rowcount)

    def ack_all(self) -> int:
        with self._lock:
            cur = self._conn.execute("UPDATE alerts SET acked = 1 WHERE acked = 0")
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
            "pending_alerts": q("SELECT COUNT(*) FROM alerts WHERE acked = 0"),
        }
