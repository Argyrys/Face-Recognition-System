"""Out-of-band alert delivery: SMTP email + generic webhook.

Configured entirely through environment variables; no channels = no-op, so
the app behaves exactly as before when nothing is set.

  VC_SMTP_HOST / VC_SMTP_PORT / VC_SMTP_USER / VC_SMTP_PASS /
  VC_SMTP_FROM / VC_SMTP_STARTTLS / VC_ALERT_EMAIL_TO   -> email channel
  VC_ALERT_WEBHOOK_URL                                  -> webhook channel

The webhook posts {"title","body","level","source"} JSON — directly usable
by Discord/Slack-compatible endpoints (a "content" field is sent when the
URL looks like Discord), IFTTT/Zapier makers, or your own receiver.

Delivery is rate-limited per channel (VC_ALERT_MIN_INTERVAL seconds,
default 60) so a noisy stream cannot flood a mailbox; level="critical"
(e.g. suspected spoofing, camera outage) always goes through.
"""

import json
import logging
import smtplib
import threading
import time
import urllib.request
from email.message import EmailMessage

from config import Config

log = logging.getLogger("vc.alerting")

SOURCE = "VC Cabin"


class EmailChannel:
    name = "email"

    def __init__(self, to_addr, min_interval):
        self.to_addr = to_addr
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def ready(self, level: str) -> bool:
        if level == "critical":
            return True
        with self._lock:
            if time.time() - self._last < self.min_interval:
                return False
            self._last = time.time()
            return True

    def send(self, title, body, level):
        msg = EmailMessage()
        msg["From"] = Config.SMTP_FROM
        msg["To"] = self.to_addr
        msg["Subject"] = f"[{level.upper()}] {title}"
        msg.set_content(f"{body}\n\n-- {SOURCE}")
        with smtplib.SMTP(Config.SMTP_HOST, Config.SMTP_PORT,
                          timeout=10) as server:
            if Config.SMTP_STARTTLS:
                server.starttls()
            if Config.SMTP_USER:
                server.login(Config.SMTP_USER, Config.SMTP_PASS)
            server.send_message(msg)


class WebhookChannel:
    name = "webhook"

    def __init__(self, url, min_interval):
        self.url = url
        self.min_interval = min_interval
        self._last = 0.0
        self._lock = threading.Lock()

    def ready(self, level: str) -> bool:
        if level == "critical":
            return True
        with self._lock:
            if time.time() - self._last < self.min_interval:
                return False
            self._last = time.time()
            return True

    def send(self, title, body, level):
        if "discord.com" in self.url or "discordapp" in self.url:
            payload = {"content": f"**{title}**\n{body}"}
        else:
            payload = {"title": title, "body": body, "level": level,
                       "source": SOURCE}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()


class Alerter:
    """Fan-out with per-channel rate limiting; never raises into the caller."""

    def __init__(self, channels=()):
        self.channels = list(channels)

    @property
    def armed(self) -> list:
        return [c.name for c in self.channels]

    def send(self, title: str, body: str, level: str = "info") -> int:
        """Deliver to every ready channel. Returns delivery count."""
        sent = 0
        for ch in self.channels:
            if not ch.ready(level):
                continue
            try:
                ch.send(title, body, level)
                sent += 1
            except Exception as exc:      # bad creds / network: log, keep going
                log.warning("alert channel '%s' failed: %s", ch.name, exc)
        return sent


def build_alerter() -> Alerter:
    """Assemble channels from the environment (empty => silent no-op)."""
    channels = []
    if Config.ALERT_EMAIL_TO and Config.SMTP_HOST:
        channels.append(EmailChannel(Config.ALERT_EMAIL_TO,
                                     Config.ALERT_MIN_INTERVAL))
    if Config.ALERT_WEBHOOK_URL:
        channels.append(WebhookChannel(Config.ALERT_WEBHOOK_URL,
                                       Config.ALERT_MIN_INTERVAL))
    if channels:
        log.info("out-of-band alerts armed: %s", ", "
                 .join(c.name for c in channels))
    return Alerter(channels)
