from __future__ import annotations

import os
from pathlib import Path


def load_local_environment(
    path: Path | None = None,
) -> dict[str, str]:
    """Load simple KEY=VALUE pairs without overriding process environment."""

    service_root = Path(__file__).resolve().parent.parent
    env_path = path or service_root / ".env.local"
    loaded: dict[str, str] = {}
    if not env_path.exists():
        return loaded

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded
