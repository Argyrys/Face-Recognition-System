# Privacy Notice — VC Cabin Face Recognition System

*Data practices of the face-recognition visitor system installed at the VC
cabin entry, described in plain language. Aligned with the principles of
India's Digital Personal Data Protection (DPDP) Act 2023 and general
good-practice requirements of other frameworks.*

## What personal data is collected

| Data | When | Where it lives |
|------|------|----------------|
| Face photograph (snapshot crop) | Each detected visitor event | `static/snapshots/*.jpg` on the host machine |
| Face embedding (mathematical vector derived from the face) | Only for consented enrolled persons | `database/vc_entry.db` |
| Name, role, department, ID (as provided) | Enrollment of consented persons | `database/vc_entry.db` |
| Visitor event record (time, MATCHED/UNKNOWN, confidence, liveness label, snapshot filename) | Every logged visitor | `database/vc_entry.db` |
| Operator actions (logins, acknowledgements, enrollments, toggles) | Continuously | `database/vc_entry.db` (audit trail) |

**No biometric data leaves the machine** unless the operator explicitly
configures an alert channel (email/webhook), which sends only the event
summary text (name/result/time), not embeddings. There is no cloud, no
telemetry, and no third-party processing.

## Purpose (why the data is taken)

To support the VC cabin's security workflow: identifying enrolled persons at
the entry, flagging unknown visitors for manual verification, and keeping a
short-lived record of visits. **The software never makes an access
decision** — a human guard does.

## Legal basis and consent

- Enrollment of faculty/staff is **opt-in with explicit consent** (the
  enrollment UI requires a consent checkbox; the CLI must only be used for
  subjects who have consented). Consent can be withdrawn by the data
  subject at any time.
- Visitors passing through the camera are processed for the legitimate
  security operation of the premises; their records are short-lived by
  configuration (`VC_RETENTION_DAYS`, default 30 days) and snapshots are
  pruned automatically.

## Retention and erasure

- Visitor entries, alerts, and snapshots: purged automatically after
  `VC_RETENTION_DAYS` (default 30 days).
- Operator audit trail: purged after `VC_AUDIT_RETENTION_DAYS`
  (default 90 days).
- Enrolled persons: kept until consent is withdrawn or the person is
  deactivated in the dashboard; deletion removes the embedding, photo, and
  profile.
- Full erasure of the installation = delete `database/vc_entry.db` and
  `static/snapshots/` (backups under `backups/` must also be deleted).

## Data subject rights (implemented)

| Right | How |
|-------|-----|
| **Notice** | this document |
| **Consent withdrawal / erasure** | dashboard: deactivate or delete the person; their embedding and photo are removed |
| **Access** | the operator can export any person's data from the dashboard/API; visitors may ask the operator for their entry records |
| **Grievance** | contact the department operating the cabin (fill in your institution's DPO contact here before deployment) |

## Security measures applied to the data

Single-password operator gate with lockout, session hardening, CSRF
defence, append-only audit trail, security headers, local-only storage,
automatic backups, and TLS support via reverse proxy. See
`docs/SECURITY.md` for the threat model and residual risks (including the
acknowledged limits of passive liveness detection).

## Honest statement of capability

This system is a **visitor-monitoring aid**, not an identity guarantee.
Recognition can err (see the threshold-sweep report for measured error
rates), and passive anti-spoofing cannot catch a high-quality video replay.
Every access decision is made by a human who can override the software.

---
*Institution name / Data Protection Officer contact: _fill in before
deployment_*
