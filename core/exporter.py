# ===== core/exporter.py =====
"""
TransactionExporter
-------------------
Writes data/transactions.csv with a fixed 14-column header:
transaction_id,date,internal_account_number,internal_entity,institution,
external_entity,amount,available_balance,currency,description,
transaction_type,source_email,email_id,run_id

Now also auto-assigns a unique integer transaction_id:
  • Reads the existing max ID from the master CSV
  • New rows get IDs = max_existing + 1, +2, …
Other responsibilities unchanged (snapshot, flags, duplicate guard, etc.)

New in this version:
  • Fast-path append if new batch dates ≥ last existing date.
  • Otherwise, merge the existing CSV and new batch in O(N+M) time.
"""

from __future__ import annotations

"""
TransactionExporter
-------------------
Responsible for appending new transactions into your master CSV,
assigning incrementing IDs, and snapshotting per‐run files.
"""

import csv
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict
import heapq
from datetime import datetime

from core.models import Transaction
from config.config_loader import load_config

class TransactionExporter:
    def __init__(self) -> None:
        # Load paths via Pydantic config
        cfg   = load_config()
        paths = cfg.paths

        self.master    = Path(paths.transactions_csv)
        self.run_dir   = Path(paths.invoice_dir)
        self.runs_csv  = Path(paths.runs_csv)
        self.flag_csv: Path = Path(paths.flagged_csv)
        self.meta_csv: Path = Path(paths.metadata_csv)

        # Ensure directories exist
        for p in (self.master, self.runs_csv):
            p.parent.mkdir(parents=True, exist_ok=True)

        # Canonical 14-column header
        self._columns: List[str] = [
            "transaction_id",
            "date",
            "internal_account_number",
            "internal_entity",
            "institution",
            "external_entity",
            "amount",
            "available_balance",
            "currency",
            "description",
            "transaction_type",
            "source_email",
            "email_id",
            "run_id",
        ]

        # for string‐type fields when normalizing
        self._string_cols: set[str] = {
            c for c in self._columns
            if c not in {"amount", "available_balance"}
        }

        # cache Gmail IDs to prevent dupes
        self._processed_ids: Set[str] = self._load_processed_ids()

    def export_batch(self, txns: List[Dict]) -> int:
        """
        Append this batch of transaction dicts into transactions.csv,
        auto-assigning transaction_id = max_existing_id + incremental index.
        Returns number of rows written.
        """
        # Find current max ID
        max_id = 0
        if self.master.exists() and self.master.stat().st_size > 0:
            with self.master.open(newline="") as f:
                reader = csv.DictReader(f)
                if "transaction_id" in reader.fieldnames:
                    for row in reader:
                        try:
                            tid = int(row["transaction_id"])
                            if tid > max_id:
                                max_id = tid
                        except ValueError:
                            continue

        # Prepare rows with new IDs
        out_rows = []
        for idx, rec in enumerate(txns, start=1):
            rec["transaction_id"] = max_id + idx
            out_rows.append(rec)

        # Append to master CSV
        with self.master.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            # Write header if file was empty
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(out_rows)

        # Return count of appended rows
        return len(out_rows)
    
    # ------------------------------------------------------------------ #
    # Duplicate detection                                                 #
    # ------------------------------------------------------------------ #
    def _load_processed_ids(self) -> Set[str]:
        if not self.master.exists():
            return set()
        with self.master.open(newline="") as f:
            r = csv.DictReader(f)
            return {
                row["email_id"]
                for row in r
                if r.fieldnames and "email_id" in r.fieldnames and row.get("email_id")
            }

    def has_processed(self, msg_id: str) -> bool:
        return msg_id in self._processed_ids

    # ------------------------------------------------------------------ #
    # Public export                                                       #
    # ------------------------------------------------------------------ #
    def export(self, txns: List[Transaction], run_id: str) -> None:
        # 0) Clean + flag + bucket
        buckets = self._prep_rows(txns, run_id)

        # 1) Infer balances
        for lst in buckets.values():
            self._impute_available_balance(lst)

        # 2) Flatten & sort the new batch
        new_batch = [t for lst in buckets.values() for t in lst]
        new_batch.sort(key=lambda r: r.date)

        # ── assign new transaction_id ───────────────────────────────────────
        max_id = 0
        if self.master.exists() and self.master.stat().st_size > 0:
            with self.master.open(newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and "transaction_id" in reader.fieldnames:
                    for row in reader:
                        try:
                            tid = int(row["transaction_id"])
                            if tid > max_id:
                                max_id = tid
                        except Exception:
                            continue
        for idx, t in enumerate(new_batch, start=1):
            t.transaction_id = max_id + idx

        # 3) Write to master CSV (either fast‐append or merge)
        if self.master.exists() and self.master.stat().st_size > 0:
            # read only the last date from existing file
            with self.master.open(newline="") as f:
                reader = csv.DictReader(f)
                last_row = None
                for last_row in reader:
                    pass
            last_date_str = last_row["date"] if last_row else ""
            # compare ISO strings
            if new_batch and new_batch[0].date.strftime("%Y-%m-%d") >= last_date_str:
                # fast‐append (safe because entire new_batch is ≥ last existing date)
                self._append(self.master, new_batch, header_always=False)
            else:
                # merge old + new into one sorted file
                self._merge_and_write(self.master, new_batch)
        else:
            # master missing or empty → just write header + new_batch
            self._append(self.master, new_batch, header_always=True)

        # 4) Snapshot only the new batch
        snap = self.run_dir / run_id / f"transactions_{run_id}.csv"
        snap.parent.mkdir(parents=True, exist_ok=True)
        self._append(snap, new_batch, header_always=True)

        # 5) Update processed‐IDs cache
        self._processed_ids.update(t.email_id for t in new_batch if t.email_id)

    # ------------------------------------------------------------------ #
    # Merge helper (linear O(N+M))                                        #
    # ------------------------------------------------------------------ #
    def _merge_and_write(self, path: Path, new_batch: List[Transaction]) -> None:
        # 1) load existing CSV rows as dicts
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            old_rows = list(reader)

        # 2) convert new_batch to list of dicts
        new_dicts = []
        for t in new_batch:
            rec = t.to_dict()
            for key in self._columns:
                rec.setdefault(key, "")
            rec["date"] = rec.get("date") or t.timestamp.strftime("%Y-%m-%d")
            new_dicts.append(rec)

        # 3) stream‐merge on ISO date
        def key_fn(r: Dict[str,str]) -> datetime:
            return datetime.fromisoformat(r["date"])

        merged = heapq.merge(old_rows, new_dicts, key=key_fn)

        # 4) rewrite entire master file
        with path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(self._columns)
            for rec in merged:
                # ensure order of columns
                w.writerow([rec[col] for col in self._columns])

    # ------------------------------------------------------------------ #
    # Cleaning / flagging                                                 #
    # ------------------------------------------------------------------ #
    def _prep_rows(self, txns: List[Transaction], run_id: str):
        buckets: Dict[str, List[Transaction]] = defaultdict(list)
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
                    t.email_id or "",
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
    def _impute_available_balance(txns: List[Transaction]) -> None:
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
    def _append(self, path: Path, rows: List[Transaction], *, header_always: bool):
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

    # ------------------------------------------------------------------ #
    # Flag logging                                                        #
    # ------------------------------------------------------------------ #
    def _log_flag(
        self,
        run_id: str,
        email_id: str,
        sender: str,
        subject: str,
        reason: str,
    ) -> None:
        hdr = not self.flag_csv.exists()
        with self.flag_csv.open("a", newline="") as f:
            w = csv.writer(f)
            if hdr:
                w.writerow(["run_id", "email_id", "sender", "subject", "reason"])
            w.writerow([run_id, email_id, sender, subject, reason])

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
                        "transactions_written",
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