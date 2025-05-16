# src/core/hash_utils.py

from __future__ import annotations
import csv
import hashlib
import io
from datetime import datetime
from pathlib import Path
from typing import List, Optional


def _get_sort_key(cols: List[str]) -> Optional[str]:
    """
    Prefer 'timestamp' or 'date' for temporal sorting; if none, return None
    so rows are sorted lexicographically on all columns.
    """
    for key in ("timestamp", "date", "run_timestamp"):
        if key in cols:
            return key
    return None


def canonical_bytes(path: Path) -> bytes:
    """
    Read CSV at `path`, strip whitespace, sort rows by temporal key if present
    (ISO-8601 parsed), else lexicographically on all fields; then sort columns
    alphabetically; serialize to UTF-8 CSV bytes with '\n' line endings.
    """
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Empty file → empty bytes
    if not rows:
        return b""

    cols = list(rows[0].keys())
    temporal = _get_sort_key(cols)
    if temporal:
        rows.sort(key=lambda r: datetime.fromisoformat(r[temporal]))
    else:
        rows.sort(key=lambda r: [r[c] for c in sorted(cols)])

    ordered_cols = sorted(cols)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(ordered_cols)
    for r in rows:
        writer.writerow([r[c].strip() for c in ordered_cols])

    return buf.getvalue().encode("utf-8")


def file_hash(path: Path) -> str:
    """
    Compute Blake2b-256 digest of one canonical CSV file.
    """
    data = canonical_bytes(path)
    return hashlib.blake2b(data, digest_size=32).hexdigest()


def folder_hash(root: Path, patterns: List[str]) -> str:
    """
    Compute a single Blake2b-256 digest over *all* canonical_bytes() outputs for
    every file under `root` matching any glob in `patterns` (e.g. ["*.csv"]).
    """
    hasher = hashlib.blake2b(digest_size=32)
    for pat in patterns:
        for p in sorted(root.glob(pat)):
            hasher.update(canonical_bytes(p))
    return hasher.hexdigest()
