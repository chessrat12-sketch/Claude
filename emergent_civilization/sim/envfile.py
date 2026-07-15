"""Tiny ``.env`` loader — no dependency, so you don't retype ``set`` every session.

Reads ``KEY=VALUE`` pairs (``#`` comments and blank lines ignored) from a
``.env`` file and loads them into ``os.environ`` for anything not already set
there. A real environment variable — or a ``set``/``$env:`` you typed by hand —
always wins over the file, so the two can be mixed freely.
"""

from __future__ import annotations

import os


def load_env_file(*search_dirs: str) -> str | None:
    """Load the first ``.env`` found in ``search_dirs`` (checked in order).

    Returns the path that was loaded, or None if no ``.env`` was found.
    """
    for d in search_dirs:
        path = os.path.join(d, ".env")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key:
                    os.environ.setdefault(key, value)
        return path
    return None
