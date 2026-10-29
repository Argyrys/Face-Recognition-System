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
    # Anti-spoofing: enforced blocks photo/replay matches on live sources,
    # monitor only labels them, off disables the check entirely.
    LIVENESS = _env("VC_LIVENESS", "enforced")      # enforced | monitor | off
    LIVENESS_MIN_MOTION = float(_env("VC_LIVENESS_MIN_MOTION", "0.004"))
    LIVENESS_HOLD_FRAMES = int(_env("VC_LIVENESS_HOLD_FRAMES", "20"))
    # N-frame confirmation: a match needs this many consecutive agreeing frames
    CONFIRM_FRAMES = int(_env("VC_CONFIRM_FRAMES", "3"))
    UNKNOWN_FRAMES = int(_env("VC_UNKNOWN_FRAMES", "3"))
    # Cosine similarity floor for a match (see threshold_sweep.py output;
    # 0.363 is the SFace vendor default and sits inside the measured
    # proven-safe interval on the bundled sample data).
    MIN_COSINE = float(_env("VC_MIN_COSINE", "0.363"))

    # Watchdog -------------------------------------------------------------
    STALL_SECONDS = int(_env("VC_STALL_SECONDS", "25"))   # no frames => outage

    # Out-of-band alerts (both optional; disabled when unset) --------------
    ALERT_MIN_INTERVAL = int(_env("VC_ALERT_MIN_INTERVAL", "60"))  # per channel
    SMTP_HOST = _env("VC_SMTP_HOST", "")
    SMTP_PORT = int(_env("VC_SMTP_PORT", "587"))
    SMTP_USER = _env("VC_SMTP_USER", "")
    SMTP_PASS = _env("VC_SMTP_PASS", "")
    SMTP_FROM = _env("VC_SMTP_FROM", "vc-cabin@localhost")
    SMTP_STARTTLS = _env("VC_SMTP_STARTTLS", "1") == "1"
    ALERT_EMAIL_TO = _env("VC_ALERT_EMAIL_TO", "")
    ALERT_WEBHOOK_URL = _env("VC_ALERT_WEBHOOK_URL", "")   # Discord/Slack/IFTTT/Zapier

    # TLS (combined PEM with certificate + private key) ---------------------
    SSL_PEM = _env("VC_SSL_PEM", "")

    # Backups ---------------------------------------------------------------
    BACKUP_DIR = _env("VC_BACKUP_DIR", os.path.join(BASE_DIR, "backups"))
    BACKUP_KEEP = int(_env("VC_BACKUP_KEEP", "30"))

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
