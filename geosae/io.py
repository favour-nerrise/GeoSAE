"""Input and output helpers for GeoSAE."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_dir(path: Path) -> None:
    """Creates a directory if it does not already exist.

    Args:
      path: Directory path to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def write_json(payload: dict[str, Any], path: Path) -> None:
    """Writes a JSON payload with stable formatting.

    Args:
      payload: JSON-serializable mapping.
      path: Output path.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    """Reads a JSON object from disk.

    Args:
      path: Path to a JSON file.

    Returns:
      Parsed JSON object.
    """
    return json.loads(path.read_text())

