"""
Unified runner: incremental Gmail fetch + rich trackers + DB loaders.

Usage:
    poetry run python -m pipeline.run_pipeline                  # normal
    poetry run python -m pipeline.run_pipeline --force         # force all
    poetry run python -m pipeline.run_pipeline --start-date 2024-01-01
"""

# ===== pipeline/run_pipeline.py =====
import argparse
import csv
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
import importlib

from config.config_loader import load_config
from utils.email_query_builder import build_gmail_query
from utils.pdf_utils import extract_pdf_texts, extract_text_from_bytes
from utils.invoice_saver import save_attachments
from core.classifier import TransactionClassifier
from core.exporter import TransactionExporter
from core.run_tracker import RunTracker
from core.flagged_tracker import FlaggedTracker
from core.db_loader import load_postgres, load_snowflake
from parser.llm_parser import LLMParser

logging.basicConfig(
    level  = logging.DEBUG,
    format = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
)

CFG           = load_config("config/config.yaml")
DEFAULT_MAX   = CFG["scraper"]["max_results"]
BANKS_MAPPING = CFG.get("banks", {})


def run(start_date: str=None, max_results: int=None, force: bool=False) -> None:
    # ── Initialize connector from config ────────────────────────────────────
    connector_name = CFG["connector"]
    mod            = importlib.import_module(f"connectors.{connector_name}")
    ConnectorClass = getattr(mod, f"{connector_name.capitalize()}Connector")
    connector      = ConnectorClass(CFG["connectors"][connector_name])
    connector.authenticate()

    exporter    = TransactionExporter()
    classifier  = TransactionClassifier()
    parser      = LLMParser()
    run_tracker = RunTracker()
    flagger     = FlaggedTracker()

    # ── determine fetch boundary ────────────────────────────────────────────
    last_scrape = (
        exporter.meta_csv.read_text().strip()
        if exporter.meta_csv.exists() else None
    )
    if force:
        fetch_after = None
        logging.warning("Force mode: bypassing last_scrape/start_date")
    else:
        fetch_after = start_date or last_scrape

    run_id, run_ts = run_tracker.start_run(fetch_after)

    # ── build query & format after ──────────────────────────────────────────
    query = build_gmail_query()
    formatted_after = (
        fetch_after.split("T")[0].replace("-", "/")
        if fetch_after else None
    )

    max_n = max_results if max_results is not None else DEFAULT_MAX
    messages = connector.fetch_messages(
        since       = formatted_after,
        max_results = max_n,
        query       = query,
    )

    logging.info(
        "Fetched %d messages (q=%s, after=%s, max=%s)",
        len(messages), query, formatted_after, max_n
    )

    processed_ids = exporter._load_processed_ids()

    # ── process each message ────────────────────────────────────────────────
    duplicates, parse_fail, processed = 0, 0, 0
    txns, sample = [], []

    for m in messages:
        msg_id = m["id"]
        if msg_id in processed_ids:
            duplicates += 1
            continue

        # 1) Try PDF‐first if enabled
        pdf_text = None
        if CFG["scraper"].get("parse_pdfs", False):
            for part in m.get("payload", {}).get("parts", []):
                if part.get("filename", "").lower().endswith(".pdf"):
                    raw = connector.fetch_pdf_attachment(
                        m, part["body"]["attachmentId"]
                    )
                    if raw:
                        # you can use either helper
                        pdf_text = extract_text_from_bytes(raw)
                        # or: pdf_text = extract_pdf_texts(connector.service, m)
                        break

        # 2) Parse (PDF or body)
        try:
            if pdf_text:
                txn = parser.parse_text(m, pdf_text)
                # fallback if critical fields missing
                if not (
                    txn.account_number and txn.institution
                    and txn.amount is not None
                    and txn.currency and txn.description
                ):
                    raise ValueError("PDF parse incomplete")
            else:
                raise ValueError("no PDF")
        except Exception:
            logging.info(f"PDF parse failed for {msg_id}, falling back to email body")
            try:
                txn = parser.parse(m)
            except Exception as e:
                # flag and skip
                hdrs = {h["name"]:h["value"] for h in m["payload"]["headers"]}
                flagger.flag(
                    run_id, msg_id,
                    hdrs.get("From",""),
                    hdrs.get("Subject",""),
                    f"parse failure: {e}"
                )
                parse_fail += 1
                continue

        # 3) classify & attach metadata
        txn.transaction_type = classifier.classify(txn)
        txn.run_id          = run_id

        # 4) dynamic institution override from config→banks mapping
        from_hdr = {h["name"]:h["value"] for h in m["payload"]["headers"]}
        sender = from_hdr.get("From","").lower()
        for bank_name, domain in BANKS_MAPPING.items():
            if domain in sender:
                txn.institution = bank_name
                # pick up internal account from accounts.csv
                with open(CFG["paths"]["accounts_csv"], newline="") as f:
                    for row in csv.DictReader(f):
                        if row.get("institution","").strip().upper() == bank_name.upper():
                            txn.internal_account_number = row["internal_account_number"]
                            txn.internal_entity         = row["internal_entity"]
                            break
                break

        # 5) save invoice attachments
        try:
            save_attachments(connector.service, m, txn)
        except Exception as exc:
            logging.exception(f"Attachment-save error for message {msg_id}")
            hdrs2 = {h["name"]:h["value"] for h in m["payload"]["headers"]}
            flagger.flag(
                run_id, msg_id,
                hdrs2.get("From",""),
                hdrs2.get("Subject",""),
                f"attachment error: {exc}"
            )

        txns.append(txn)
        processed_ids.add(msg_id)
        processed += 1
        if len(sample) < 3:
            sample.append(txn.to_dict())

    # ── export & log ────────────────────────────────────────────────────────
    exporter.export(txns, run_id)
    last_time = (
        txns[-1].timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
        if txns else fetch_after or ""
    )
    exporter.log_run(run_id, run_ts, last_time, len(messages), processed)

    # ── load into DB sinks ──────────────────────────────────────────────────
    load_postgres()
    load_snowflake()

    # ── summary ─────────────────────────────────────────────────────────────
    print("=== DEBUG SAMPLE (first 3 parsed rows) ===")
    for row in sample:
        print(json.dumps(row, indent=2))
    print("=========================================")
    print(f"Duplicates skipped : {duplicates}")
    print(f"Parse failures     : {parse_fail}")
    print(f"Transactions parsed: {processed}")
    print(f"[Run {run_id}] fetched={len(messages)}, processed={processed}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", help="YYYY-MM-DD for backfill")
    ap.add_argument("--max-results", type=int, help="Override fetch size")
    ap.add_argument("--force",     action="store_true",
                    help="Ignore last_scrape; fetch all")
    args = ap.parse_args()
    run(args.start_date, args.max_results, args.force)