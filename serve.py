"""Production WSGI entrypoint: same Flask app, served by Waitress.

Usage:
    VC_ADMIN_PASSWORD=your-password python serve.py

Waitress is a pure-Python, production-grade server (no Werkzeug dev server),
safe for LAN/demo deployment and containerized runs.
"""

import logging

from waitress import serve

from app import app, start_backend
from config import Config

start_backend()      # recognition pipeline + retention thread

log = logging.getLogger("vc.serve")
log.info("VC Cabin (production) -> http://%s:%s", Config.HOST, Config.PORT)
serve(
    app,
    host=Config.HOST,
    port=Config.PORT,
    threads=Config.WAITRESS_THREADS,
    channel_timeout=120,                       # keep the MJPEG stream open
    max_request_body_size=Config.MAX_UPLOAD_MB * 1024 * 1024,
)
