
#TransactionExporter
#
#Writes data/master.csv with a fixed 14-column header:
#transaction_id,date,internal_account_number,internal_entity,institution,
#external_entity,amount,available_balance,currency,description,
#transaction_type,source_email,message_id,run_id
#
#Now also auto-assigns a unique integer transaction_id:
#  Reads the existing max ID from the master CSV
#  New rows get IDs = max_existing + 1, +2, …
#Other responsibilities unchanged (snapshot, flags, duplicate guard, etc.)
#
#New in this version:
#  Fast-path append if new batch dates ≥ last existing date.
#  Otherwise, merge the existing CSV and new batch in O(N+M) time.
#
#
#from __future__ import annotations
#
#
#TransactionExporter
#
#Responsible for appending new transactions into your master CSV,
#assigning incrementing IDs, and snapshotting per‐run files.
#

import csv
from pathlib import Path
from typing import Dict, List, Set, Optional, Union, get_origin, get_args
from collections import defaultdict
import heapq
from datetime import datetime, date
from decimal import Decimal

from pydantic import BaseModel
from core.config_loader import load_config
from utils.validators import get_model, validate_record

class TransactionExporter:
    def __init__(self, master_file: str | Path | None = None, runs_csv: str | None = None) -> None:
        # Load paths via Pydantic config
        cfg   = load_config()
        paths = cfg.paths
        self.cfg = cfg
        self.schema = cfg.active_schema
        self.master_file = Path(master_file) if master_file else Path(cfg.paths.master_file)
        self.runs_csv    = Path(runs_csv)    if runs_csv    else Path(cfg.paths.runs_csv)
        self.run_dir   = Path(paths.invoice_dir)
        self.flag_csv: Path = Path(paths.flagged_csv)
        self.meta_csv: Path = Path(paths.metadata_csv)

        # Ensure directories exist
        for p in (self.master, self.runs_csv):
            p.parent.mkdir(parents=True, exist_ok=True)

        # Canonical columns from chosen schema
        self.schema             = cfg.active_schema
        self.RecordModel: type[BaseModel] = get_model(self.schema)
        yaml_fields_order = list(self.RecordModel.model_fields.keys())  # returns fields in YAML order

        # exporter-owned ID column at the end
        self._columns: List[str] = yaml_fields_order 

        # -------------------------------------------------------------- #
        #  Infer which fields hold strings vs. numbers/dates            #
        # -------------------------------------------------------------- #
        numeric_like = {int, float, Decimal, date, datetime}

        def _is_numeric(field_name: str) -> bool:
            ann = self.RecordModel.model_fields[field_name].annotation
            origin = get_origin(ann)
            if origin is Union:                     # Optional[...] or Union
                ann = next(a for a in get_args(ann) if a is not type(None))
            return ann in numeric_like

        self._string_cols: set[str] = {
            col for col in self._columns if not _is_numeric(col)
        }

        # cache Gmail IDs to prevent dupes
        self._processed_ids: Set[str] = self._load_processed_ids()

        self.id_field = (getattr(cfg, "exporter", None) and getattr(cfg.exporter, "id_field", None))

    # ================================================================== #
    # Master-file writing                                                #
    # ================================================================== #
    def _write_master(self, new_batch: List[BaseModel]) -> None:
        if self.master.exists() and self.master.stat().st_size > 0:
            with self.master.open(newline="") as f:
                last_row = None
                for last_row in csv.DictReader(f):
                    pass
            last_date = last_row["date"] if last_row else ""
            if new_batch and new_batch[0].date.strftime("%Y-%m-%d") >= last_date:
                self._append(self.master, new_batch, header_always=False)
            else:
                self._merge_and_write(self.master, new_batch)
        else:
            self._append(self.master, new_batch, header_always=True)
    
    # ------------------------------------------------------------------ #
    # Duplicate detection                                                 #
    # ------------------------------------------------------------------ #
    def _load_processed_ids(self) -> Set[str]:
        if not self.master.exists():
            return set()
        with self.master.open(newline="") as f:
            r = csv.DictReader(f)
            return {
                row["message_id"]
                for row in r
                if r.fieldnames and "message_id" in r.fieldnames and row.get("message_id")
            }

    def has_processed(self, msg_id: str) -> bool:
        return msg_id in self._processed_ids

    # ================================================================== #
    # Public API                                                         #
    # ================================================================== #
    def export(self, txns: List[Optional[BaseModel]], run_id: str) -> None:
        txns = [t for t in txns if t is not None]

        buckets = self._prep_rows(txns, run_id)
        for lst in buckets.values():
            self._impute_available_balance(lst)

        new_batch = [t for lst in buckets.values() for t in lst]
        new_batch.sort(key=lambda r: r.date)

        # ------------ optional auto-ID ---------------------------------
        if self.id_field:
            max_id = self._get_max_id()
            for i, rec in enumerate(new_batch, start=1):
                setattr(rec, self.id_field, max_id + i)

        self._write_master(new_batch)
        self._snapshot_run(new_batch, run_id)
        self._processed_ids.update(
            getattr(t, "message_id") for t in new_batch if getattr(t, "message_id")
        )

    # ================================================================== #
    # ID helpers                                                         #
    # ================================================================== #
    def _get_max_id(self) -> int:
        if not self.id_field or not self.master.exists():
            return 0
        max_id = 0
        with self.master.open(newline="") as f:
            reader = csv.DictReader(f)
            if self.id_field in (reader.fieldnames or []):
                for row in reader:
                    try:
                        max_id = max(max_id, int(row[self.id_field]))
                    except ValueError:
                        pass
        return max_id

    # ------------------------------------------------------------------ #
    # Merge helper (linear O(N+M))                                        #
    # ------------------------------------------------------------------ #
    def _merge_and_write(self, path: Path, new_batch: List[BaseModel]) -> None:
        with path.open(newline="") as f:
            old_rows = list(csv.DictReader(f))
        new_rows = [self._model_to_dict(r) for r in new_batch]

        def _iso_key(r: Dict[str, str]) -> datetime:
                try:
                    return datetime.fromisoformat(r["date"])
                except Exception:
                    # fallback to epoch for malformed dates so they appear first
                    return datetime.fromtimestamp(0)

        merged = heapq.merge(old_rows, new_rows, key=_iso_key)
        
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._columns)
            for rec in merged:
                w.writerow([rec[c] for c in self._columns])

    # ------------------------------------------------------------------ #
    # Cleaning / flagging                                                 #
    # ------------------------------------------------------------------ #
    def _prep_rows(self, txns: List[BaseModel], run_id: str):
        buckets: Dict[str, List[BaseModel]] = defaultdict(list)
        for t in txns:
            missing: List[str] = []
            for col in self._string_cols:
                if getattr(t, col, None) in (None, ""):
                    setattr(t, col, "")
                    missing.append(col)
            if t.available_balance is None:
                missing.append("available_balance")
            if missing:
                self._log_flag(
                    run_id,
                    t.message_id or "",
                    t.source_email or "",
                    "",
                    f"missing: {', '.join(missing)}",
                )
            buckets[t.internal_account_number].append(t)
        return buckets

    # ------------------------------------------------------------------ #
    # Balance inference                                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _impute_available_balance(txns: List[BaseModel]) -> None:
        txns.sort(key=lambda r: r.date)
        prev = None
        for r in txns:
            if r.available_balance is None and prev is not None:
                r.available_balance = prev + r.amount
            if r.available_balance is not None:
                prev = r.available_balance
        nxt = None
        for r in reversed(txns):
            if r.available_balance is None and nxt is not None:
                r.available_balance = nxt - r.amount
            if r.available_balance is not None:
                nxt = r.available_balance

    # ------------------------------------------------------------------ #
    # CSV writing                                                         #
    # ------------------------------------------------------------------ #
    def _append(self, path: Path, rows: List[BaseModel], *, header_always: bool):
        mode = "w" if header_always else "a"
        with path.open(mode, newline="") as f:
            w = csv.writer(f)
            if header_always:
                w.writerow(self._columns)
            for t in rows:
                rec = t.to_dict()
                # ensure all keys exist
                for key in self._columns:
                    rec.setdefault(key, "")
                # fallback for date and run_id
                rec["date"] = rec.get("date") or t.date.strftime("%Y-%m-%d")
                rec["run_id"] = rec.get("run_id") or t.run_id
                # blank out null strings
                for col in self._string_cols:
                    if rec.get(col) is None:
                        rec[col] = ""
                w.writerow([rec[col] for col in self._columns])


    def _snapshot_run(self, rows: List[BaseModel], run_id: str) -> None:
        snap = self.run_dir / run_id / f"records_{run_id}.csv"
        snap.parent.mkdir(parents=True, exist_ok=True)
        self._append(snap, rows, header_always=True)

    # ================================================================== #
    # Cleaning / flagging / balance logic  (unchanged)                   #
    # ================================================================== #
    # … (keeping the same _prep_rows, _impute_available_balance, etc.) …

    # ================================================================== #
    # CSV writing                                                        #
    # ================================================================== #
    def _append(self, path: Path, rows: List[BaseModel], *, header_always: bool):
        mode = "w" if header_always else "a"
        with path.open(mode, newline="") as f:
            w = csv.writer(f)
            if header_always:
                w.writerow(self._columns)
            for r in rows:
                w.writerow([self._model_to_dict(r)[c] for c in self._columns])

    def _model_to_dict(self, m: BaseModel) -> Dict[str, str]:
        rec = m.model_dump()
        for c in self._columns:
            rec.setdefault(c, "")
        rec["date"] = rec.get("date") or m.date.strftime("%Y-%m-%d")
        if not rec["date"]:
            rec["date"] = m.date.strftime("%Y-%m-%d")
        for col in self._string_cols:
            if rec.get(col) is None:
                rec[col] = ""
        return rec

    # ------------------------------------------------------------------ #
    # Flag logging                                                        #
    # ------------------------------------------------------------------ #
    def _log_flag(
        self,
        run_id: str,
        message_id: str,
        sender: str,
        subject: str,
        reason: str,
    ) -> None:
        hdr = not self.flag_csv.exists()
        with self.flag_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if hdr:
                w.writerow(["run_id", "message_id", "sender", "subject", "reason"])
            w.writerow([run_id, message_id, sender, subject, reason])

    # ------------------------------------------------------------------ #
    # Run summary                                                         #
    # ------------------------------------------------------------------ #
    def log_run(
        self,
        run_id: str,
        started_ts: str,
        last_txn_ts: str,
        n_fetched: int,
        n_processed: int,
    ) -> None:
        hdr_needed = not self.runs_csv.exists()
        with self.runs_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if hdr_needed:
                w.writerow(
                    [
                        "run_id",
                        "started_utc",
                        "last_transaction_utc",
                        "messages_fetched",
                        "rows_written",
                    ]
                )
            w.writerow([run_id, started_ts, last_txn_ts, n_fetched, n_processed])

    # ------------------------------------------------------------------ #
    # Metadata                                                             #
    # ------------------------------------------------------------------ #
    def _ensure_metadata(self) -> None:
        if not self.meta_csv.exists():
            # just create the file with a header placeholder
            self.meta_csv.write_text("last_transaction_utc\n", encoding="utf-8")