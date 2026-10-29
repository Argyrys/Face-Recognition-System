"""Timestamped backup of the SQLite database + snapshot JPEGs.

Usage:
    python backup.py                # backup now into ./backups/
    python backup.py --keep 14      # keep only the newest 14 archives
    python backup.py --verify       # integrity-check the live database

Safe while the app is running: SQLite's backup API does a consistent copy
(no file-level race with the live writer), then everything is zipped with a
manifest. Scheduled via Task Scheduler (Windows) or cron (Docker/Linux) —
see README.
"""

import argparse
import datetime
import json
import os
import sqlite3
import sys
import zipfile

from config import Config
from database import DB_PATH

SNAP_DIR = Config.SNAPSHOT_DIR
IMAGE_EXT = (".jpg", ".jpeg", ".png")


def verify(db_path: str = DB_PATH) -> bool:
    """PRAGMA integrity_check on the live database."""
    try:
        conn = sqlite3.connect(db_path)
        result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        conn.close()
    except sqlite3.Error as exc:
        print(f"integrity check FAILED: {exc}")
        return False
    print(f"integrity check: {result}")
    return result == "ok"


def backup(out_dir: str = None, keep: int = None) -> str:
    out_dir = out_dir or Config.BACKUP_DIR
    keep = keep or Config.BACKUP_KEEP
    os.makedirs(out_dir, exist_ok=True)

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_db = os.path.join(out_dir, f".copied_{stamp}.db")

    src = sqlite3.connect(DB_PATH)
    dst = sqlite3.connect(tmp_db)
    src.backup(dst)                        # consistent snapshot, race-free

    counts = {}
    for table in ("persons", "entry_log", "alerts", "audit_log"):
        try:
            counts[table] = dst.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except sqlite3.Error:
            counts[table] = None
    dst.close()
    src.close()

    snapshots = []
    if os.path.isdir(SNAP_DIR):
        snapshots = sorted(f for f in os.listdir(SNAP_DIR)
                           if f.lower().endswith(IMAGE_EXT))

    zip_path = os.path.join(out_dir, f"vc_backup_{stamp}.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(tmp_db, "vc_entry.db")
        for name in snapshots:
            zf.write(os.path.join(SNAP_DIR, name), os.path.join("snapshots", name))
        zf.writestr("manifest.json", json.dumps({
            "created": datetime.datetime.now().isoformat(timespec="seconds"),
            "source_db": DB_PATH,
            "counts": counts,
            "snapshots": len(snapshots),
            "restore": "unzip, stop the app, replace database/vc_entry.db "
                       "and static/snapshots/, restart",
        }, indent=2))
    os.remove(tmp_db)

    prune(out_dir, keep)
    size = os.path.getsize(zip_path)
    print(f"backup -> {zip_path} ({size // 1024} KB, "
          f"persons={counts['persons']}, entries={counts['entry_log']}, "
          f"snapshots={len(snapshots)})")
    return zip_path


def prune(out_dir: str, keep: int) -> int:
    """Delete all but the newest `keep` archives."""
    archives = sorted(f for f in os.listdir(out_dir)
                      if f.startswith("vc_backup_") and f.endswith(".zip"))
    removed = 0
    for old in archives[:-keep] if keep > 0 else []:
        os.remove(os.path.join(out_dir, old))
        removed += 1
    if removed:
        print(f"pruned {removed} old backup(s) (keep={keep})")
    return removed


def main():
    parser = argparse.ArgumentParser(description="Back up the VC Cabin database")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--keep", type=int, default=None,
                        help="archives to keep (default from VC_BACKUP_KEEP)")
    parser.add_argument("--verify", action="store_true",
                        help="integrity-check the database instead of backing up")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        sys.exit(f"database not found: {DB_PATH}")
    if args.verify:
        sys.exit(0 if verify() else 1)
    backup(args.out, args.keep)


if __name__ == "__main__":
    main()
