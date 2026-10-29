# VC Cabin — production image (Waitress on port 8000).
#
#   docker build -t vc-cabin .
#   docker run -p 8000:8000 -e VC_ADMIN_PASSWORD=change-me vc-cabin
#
# Camera access inside a container needs the device passed through, e.g.
#   docker run --device=/dev/video0 ... -e VC_CAMERA=0
# or point VC_CAMERA at an IP Webcam / DroidCam stream URL.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    VC_HOST=0.0.0.0 \
    VC_PORT=8000 \
    VC_CAMERA=0

WORKDIR /app

# opencv-python-headless still needs libgl1 + glib at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# persist DB, snapshots and logs across container restarts
VOLUME ["/app/database", "/app/static/snapshots", "/app/logs"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status==200 else 1)"

CMD ["python", "serve.py"]
