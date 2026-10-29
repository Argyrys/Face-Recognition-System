"""Environment-driven configuration + structured logging for VC Cabin.

Every knob is an environment variable with a safe local default, so the same
code runs on a laptop (dev server on localhost) and in Docker/production
(Waitress on 0.0.0.0) without edits.
"""

import logging
import logging.handlers
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _env(name, default):
    return os.environ.get(name, default)


class Config:
    # Security ------------------------------------------------------------
    SECRET_KEY = _env("VC_SECRET_KEY", "")          # empty -> generated, stored in DB
    ADMIN_PASSWORD = _env("VC_ADMIN_PASSWORD", "")  # empty -> generated on first run
    SESSION_HOURS = int(_env("VC_SESSION_HOURS", "12"))
    MAX_UPLOAD_MB = int(_env("VC_MAX_UPLOAD_MB", "8"))

    # Server --------------------------------------------------------------
    HOST = _env("VC_HOST", "127.0.0.1")             # set 0.0.0.0 to expose on LAN
    PORT = int(_env("VC_PORT", "5000"))
    WAITRESS_THREADS = int(_env("VC_THREADS", "4"))

    # Recognition ---------------------------------------------------------
    CAMERA = _env("VC_CAMERA", "0")                 # index, video file, or stream URL
    TOAST = _env("VC_TOAST", "1") == "1"

    # Data retention ------------------------------------------------------
    RETENTION_DAYS = int(_env("VC_RETENTION_DAYS", "30"))          # entries + snapshots
    AUDIT_RETENTION_DAYS = int(_env("VC_AUDIT_RETENTION_DAYS", "90"))
    RETENTION_INTERVAL_HOURS = int(_env("VC_RETENTION_INTERVAL_HOURS", "6"))

    # Storage (overridden in tests / containers) --------------------------
    DB_PATH = _env("VC_DB_PATH", os.path.join(BASE_DIR, "database", "vc_entry.db"))
    SNAPSHOT_DIR = _env(
        "VC_SNAPSHOT_DIR", os.path.join(BASE_DIR, "static", "snapshots")
    )

    # Logging -------------------------------------------------------------
    LOG_LEVEL = _env("VC_LOG_LEVEL", "INFO").upper()
    LOG_FILE = _env("VC_LOG_FILE", os.path.join(BASE_DIR, "logs", "vc_cabin.log"))


def setup_logging() -> logging.Logger:
    """Console + rotating-file logging. Called once at app import."""
    root = logging.getLogger("vc")
    if root.handlers:                      # already configured (e.g. tests)
        return root
    root.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    root.addHandler(console)
    try:
        os.makedirs(os.path.dirname(Config.LOG_FILE), exist_ok=True)
        fileh = logging.handlers.RotatingFileHandler(
            Config.LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
        fileh.setFormatter(fmt)
        root.addHandler(fileh)
    except OSError:                        # read-only fs: console logging is enough
        pass
    root.propagate = False
    return root
