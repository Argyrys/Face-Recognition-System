# Security — Threat Model & Residual Risks

This document states what the VC Cabin Face Recognition System defends
against, how, and — just as important — **what it does not defend against**.
No claim here goes beyond what the code actually does.

## Scope

- **Asset**: the visitor record (who entered the VC cabin, when, with photo),
  the enrolled gallery (faculty/staff face embeddings), and the operator
  dashboard.
- **Trust boundary**: everything on the network side of the Flask port is
  untrusted; the machine running the app is trusted (physical access to the
  host defeats any software).
- **Non-goal**: controlling doors/locks. The software detects, recognizes,
  logs, and notifies. A human makes every access decision.

## Threats and mitigations

| # | Threat | Mitigation | Residual risk |
|---|--------|------------|---------------|
| T1 | Wrong identity from a single blurred/lucky frame | IoU tracking + N-frame confirmation (`VC_CONFIRM_FRAMES`, default 3 consecutive agreeing frames) before any MATCHED entry | A persistent impostor photographed similarly to an enrolled person could accumulate agreeing frames; mitigated by the cosine threshold and liveness gate |
| T2 | **Photo attack** — printed photo of an enrolled person held to the camera | Passive PAD: landmark motion + blink detection. A static face has neither; on live sources (`VC_LIVENESS=enforced`) it is logged as a presentation attack with a critical alert, never MATCHED | A very high-resolution photo with subtle hand jitter might read as low motion "unknown"; it still cannot reach MATCHED in enforced mode without crossing the liveness gate |
| T3 | **Replay attack** — video of a blinking person on a phone screen | Same passive PAD sees genuine motion and blinks in a replay | **A video replay of a real, blinking person passes passive liveness.** This is a known limit of single-camera passive PAD. Defense in depth: challenge-response (below), guarded enrollment (the attacker must first obtain a usable video of the target), and the human guard watching the live feed |
| T4 | Credential theft of the dashboard password | Single admin password with lockout (5 fails → 60 s per IP), HttpOnly/SameSite session cookie, CSRF referrer check, audit trail of all logins | Single-factor by design (operator appliance); strengthen with a reverse proxy providing SSO/2FA if deployed on an untrusted network |
| T5 | Alert flooding / mailbox DoS | Per-channel rate limit (`VC_ALERT_MIN_INTERVAL`), one entry per track (cooldown `VC_COOLDOWN`) | A crowd of genuinely new visitors is throttled rather than lost — check the dashboard |
| T6 | Camera sabotage / outage (unmonitored entry) | Watchdog: no frames for `VC_STALL_SECONDS` → automatic restart with backoff, red outage banner, audit entries, critical alert | The restart re-opens the same device; a hardware failure needs an operator. During the stall window the entry is unmonitored — the banner and alert make this visible rather than silent |
| T7 | Database tampering / evidence loss | Append-only audit trail, retention windows, `backup.py` consistent backups with integrity verify | An attacker with host access can alter or delete the SQLite file; keep backups off-host (the backup zip is designed to be copied elsewhere) |
| T8 | Network eavesdropping / session hijack on LAN | Works correctly behind a TLS-terminating reverse proxy (Caddy/nginx — see README *HTTPS*); security headers on every response | **The built-in server (Waitress) speaks plain HTTP** — do not expose the port directly on an untrusted network; bind `VC_HOST=127.0.0.1` and proxy |
| T9 | False accept from an inflated match threshold | `threshold_sweep.py` measures FAR/FRR on your own images; recommended operating point targets FAR ≤ 1%; threshold is one env var (`VC_MIN_COSINE`) | Sample-data statistics are small (2 identities); re-run the sweep with real captures before trusting a number as an installation FAR |
| T10 | Sensitive data over-retention | Automatic purge of entries/alerts/snapshots (`VC_RETENTION_DAYS`) and audit (`VC_AUDIT_RETENTION_DAYS`), consent-gated enrollment, full erasure = delete DB + snapshots | Operator may shorten retention to 0-days only by configuration discipline; see `docs/PRIVACY_NOTICE.md` |

## Defense-in-depth recommendation (challenge-response)

The `liveness` module is passive. If the installation faces a higher-risk
threat (T3), add an **active** step at the guard's discretion: ask the
visitor to turn their head left/right or blink on command while the guard
watches the live feed. Because the operator makes the access decision, this
costs nothing in software and defeats recorded replays. (An automated
challenge-response mode is out of scope for this project version.)

## Secure deployment checklist

1. `VC_HOST=127.0.0.1` (default) + TLS reverse proxy; never expose plain
   HTTP on an untrusted network.
2. Set `VC_ADMIN_PASSWORD` from a secrets manager, not a shell history.
3. `VC_LIVENESS=enforced` (default) for any live camera source.
4. Run `python threshold_sweep.py --dir <real_probes>` on site and set
   `VC_MIN_COSINE` from the report.
5. Configure at least one out-of-band alert channel (webhook is easiest).
6. Schedule `python backup.py` daily; copy archives off-host.
7. Review the dashboard `Audit Trail` weekly.
