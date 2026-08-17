"""Update notification -- PIIP audit 2026-08. Compares the local VERSION file against the same
file on GitHub's raw-content CDN (raw.githubusercontent.com, NOT the rate-limited api.github.com
REST API -- a plain file fetch, no auth, no meaningful rate limit for this low-frequency check)
and surfaces a small sidebar notice + the CHANGELOG.md content when a newer version is available.

Design rules, same discipline as the rest of this codebase:
  * NEVER blocks or breaks the app -- any failure (offline, GitHub down, timeout, repo not public
    yet) is swallowed and the check just quietly doesn't fire. Nobody's session should ever be
    interrupted by a version-check network call.
  * Meant to be cached by the caller (see app.py's @st.cache_data wrapper) so this hits the
    network at most once every several hours, not on every page load/rerun.
  * Never auto-updates anything -- this only INFORMS; the user decides when/whether to update.
"""
from __future__ import annotations

from pathlib import Path

import requests

_LOCAL_VERSION_FILE = Path(__file__).resolve().parent.parent / "VERSION"
_RAW_BASE = "https://raw.githubusercontent.com/{repo}/master"


def local_version() -> str:
    """The version baked into THIS copy of the app -- read fresh every call (cheap, tiny local
    file), never cached, since it must always reflect what's actually running right now."""
    try:
        return _LOCAL_VERSION_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        return "unknown"


def check_for_update(repo: str, timeout: int = 5) -> dict | None:
    """Fetches the remote VERSION and CHANGELOG.md from GitHub's raw-content CDN for `repo`
    ("owner/name"). Returns None on ANY failure -- offline, timeout, a 404 if the repo isn't
    public yet, etc. -- never raises. Caller is responsible for caching this."""
    try:
        v_resp = requests.get(f"{_RAW_BASE.format(repo=repo)}/VERSION", timeout=timeout)
        if v_resp.status_code != 200:
            return None
        remote_version = v_resp.text.strip()
    except Exception:
        return None

    local = local_version()
    # Date-formatted (YYYY.MM.DD) versions sort correctly as plain strings -- no version-parsing
    # library needed. Only ever flags an update when the remote is LATER than local, never on a
    # bare mismatch (e.g. a local dev checkout ahead of what's published shouldn't say "update
    # available" just because the strings differ).
    update_available = local != "unknown" and remote_version > local

    changelog = None
    if update_available:
        try:
            c_resp = requests.get(f"{_RAW_BASE.format(repo=repo)}/CHANGELOG.md", timeout=timeout)
            if c_resp.status_code == 200:
                changelog = c_resp.text
        except Exception:
            pass   # the update notice still works without changelog text, just no preview

    return {"local_version": local, "remote_version": remote_version,
            "update_available": update_available, "changelog": changelog}
