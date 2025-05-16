# src/core/determinism.py

from __future__ import annotations
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import pandas as pd

from core.config_loader import load_config, DEFAULT_CONFIG_PATH
from utils.hash_utils import file_hash, folder_hash

def write_run_meta(
    run_id: str,
    output_dir: Union[str, Path],
    prompt_text: str,
    model_name: str,
) -> None:
    """
    Appends run_hash, config_hash, prompt_hash, model_digest to data/runs.csv
    and writes a JSON line into data/run_meta.csv with counts+latency.
    For determinism runs, prefers 'det_flagged_messages.csv' in output_dir;
    if missing, treats failed=0.
    """
    CFG = load_config()

    # Resolve paths
    runs_csv    = Path(CFG.paths.runs_csv)                    # data/runs.csv
    meta_json   = Path(CFG.paths.metadata_csv).with_name("run_meta.csv")
    config_path = DEFAULT_CONFIG_PATH
    out_dir     = Path(output_dir)

    # Timing start
    t0 = time.perf_counter()

    # 3) Compute fingerprints
    master_name  = Path(CFG.paths.master_file).name
    flagged_name = Path(CFG.paths.flagged_csv).name
    det_flagged  = f"det_{flagged_name}"
    patterns = [master_name]
    if (out_dir / det_flagged).exists():
        patterns.append(det_flagged)
    elif (out_dir / flagged_name).exists():
        patterns.append(flagged_name)

    # Compute run_hash over analytics CSVs in out_dir
    master_name = Path(CFG.paths.master_file).name            # e.g. "transactions.csv"
    flagged_name = Path(CFG.paths.flagged_csv).name           # e.g. "flagged_messages.csv"
    det_flagged = f"det_{flagged_name}"
    patterns = [master_name]
    # If the determinism‐specific flagged file exists, include it; else ignore.
    if (out_dir / det_flagged).exists():
        patterns.append(det_flagged)
    elif (out_dir / flagged_name).exists():
        patterns.append(flagged_name)
    # else no flagged file in this out_dir; failed will be zero.

    run_hash    = folder_hash(out_dir, patterns)
    config_hash = file_hash(config_path) if config_path.exists() else ""   # ← hash DEFAULT_CONFIG_PATH
    prompt_hash = hashlib.blake2b(prompt_text.encode(), digest_size=32).hexdigest()
    model_hash  = hashlib.blake2b(model_name.encode(), digest_size=32).hexdigest()
    latency_sec = round(time.perf_counter() - t0, 3)

    # Update data/runs.csv
    df = pd.read_csv(runs_csv)
    for col in ("run_hash", "config_hash", "prompt_hash", "model_digest"):
        if col not in df.columns:
            df[col] = ""
    df.loc[df.run_id == run_id, ["run_hash","config_hash","prompt_hash","model_digest"]] = [
        run_hash, config_hash, prompt_hash, model_hash
    ]
    df.to_csv(runs_csv, index=False)

    # Compute success/failed
    # success = non-empty transactions file
    success = 0
    tx_csv = out_dir / master_name
    if tx_csv.exists() and tx_csv.stat().st_size > 0:
        success = 1

    # failed = number of rows in the determinism‐specific flagged file if present
    failed = 0
    flagged_path = out_dir / det_flagged
    if flagged_path.exists():
        failed = len(pd.read_csv(flagged_path))
    else:
        # fallback to unprefixed, if someone put flagged.csv here
        f2 = out_dir / flagged_name
        if f2.exists():
            failed = len(pd.read_csv(f2))

    # Append line to run_meta.csv
    meta = {
        "run_id":      run_id,
        "timestamp":   datetime.now(timezone.utc).isoformat(),
        "run_hash":    run_hash,
        "success":     success,
        "failed":      failed,
        "latency_sec": latency_sec
    }
    with meta_json.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")