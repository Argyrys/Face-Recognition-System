"""Visitor notifications: alert records (dashboard) + optional Windows toast."""

import os
import subprocess
import threading

_TOAST_SCRIPT = (
    "Add-Type -AssemblyName System.Windows.Forms;"
    "Add-Type -AssemblyName System.Drawing;"
    "$n = New-Object System.Windows.Forms.NotifyIcon;"
    "$n.Icon = [System.Drawing.SystemIcons]::Information;"
    "$n.Visible = $true;"
    "$n.ShowBalloonTip(5000, $env:VC_TOAST_TITLE, $env:VC_TOAST_MSG,"
    " [System.Windows.Forms.ToolTipIcon]::Info);"
    "Start-Sleep -Seconds 6; $n.Dispose()"
)


def show_toast(title: str, message: str, enabled: bool = True) -> None:
    """Fire-and-forget Windows balloon notification; failures are ignored."""
    if not enabled or os.name != "nt":
        return

    def _run():
        env = os.environ.copy()
        env["VC_TOAST_TITLE"] = str(title)[:60]
        env["VC_TOAST_MSG"] = str(message)[:120]
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST_SCRIPT],
                capture_output=True, timeout=15, env=env,
            )
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()
