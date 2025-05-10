# pipeline/run_pipeline.py
"""
Single-process incremental pipeline runner
(behaviour identical to 9-May baseline, cleaner config flow).
"""

from __future__ import annotations
import argparse, csv, json, logging, importlib
from datetime import datetime
from pathlib import Path

from config.config_loader import load_config
from utils.email_query_builder import build_gmail_query
from utils.invoice_saver      import save_attachments
from core.classifier          import TransactionClassifier
from core.exporter            import TransactionExporter
from core.run_tracker         import RunTracker
from core.flagged_tracker     import FlaggedTracker
from core.db_loader           import load_postgres, load_snowflake
from parser.llm_parser        import LLMParser

logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
)
LOG = logging.getLogger(__name__)

CFG           = load_config()
DEFAULT_CONN  = CFG.default_connector
conn_cfg     = CFG.connectors[DEFAULT_CONN]
DEFAULT_QUERY = conn_cfg.query
DEFAULT_MAX   = conn_cfg.max_results
BANKS = {a.institution.lower(): a.internal_entity for a in CFG.accounts}

# ─────────────────────────────────────────────────────────────────────────────
def run(start_date: str | None = None,
        max_results: int | None = None,
        force: bool = False) -> None:

    # connector plugin -------------------------------------------------------
    mod            = importlib.import_module(f"connectors.{DEFAULT_CONN}")
    ConnectorClass = getattr(mod, f"{DEFAULT_CONN.capitalize()}Connector")
    connector      = ConnectorClass(CFG.connectors[DEFAULT_CONN])

    exporter    = TransactionExporter()
    classifier  = TransactionClassifier()
    parser      = LLMParser()
    run_tracker = RunTracker()
    flagger     = FlaggedTracker()

    # scraping window --------------------------------------------------------
    last_scrape = Path(CFG.paths.metadata_csv).read_text().strip() if Path(CFG.paths.metadata_csv).exists() else None
    fetch_after = None if force else (start_date or last_scrape)

    run_id, run_ts = run_tracker.start_run(fetch_after)

    # gmail search query -----------------------------------------------------
    formatted_after = fetch_after.split("T")[0].replace("-", "/") if fetch_after else None
    query   = DEFAULT_QUERY or build_gmail_query()
    max_n   = max_results if max_results is not None else DEFAULT_MAX

    msgs = connector.fetch_messages(
        since       = formatted_after,
        max_results = max_n,
        query       = query,
    )
    LOG.info("Fetched %d Gmail messages", len(msgs))

    # process ---------------------------------------------------------------
    processed_ids = set()
    txns, sample  = [], []
    dup = fail = 0

    for m in msgs:
        mid = m["id"]
        if mid in processed_ids:
            dup += 1
            continue

        # **always build headers here, so 'hdr' is available below**
        hdr = {h["name"]: h["value"] for h in m["payload"].get("headers", [])}

        # attempt parse
        try:
            
            txn = parser.parse(m, run_id= run_id)
            txn.run_id = run_id
        except Exception as e:
            flagger.flag(
                run_id,
                mid,
                hdr.get("From", ""),
                hdr.get("Subject", ""),
                f"parse: {e}"
            )
            fail += 1
            continue

        # classify + bank override
        txn.transaction_type = classifier.classify(txn)
        for bank in BANKS:
            if bank in hdr.get("From", "").lower():
                txn.institution = bank.title()
                txn.internal_entity = BANKS[bank]
                break

        # attachments
        try:
            save_attachments(connector.service, m, txn)
        except Exception as e:
            flagger.flag(
                run_id,
                mid,
                hdr.get("From", ""),
                hdr.get("Subject", ""),
                f"attachment: {e}"
            )

        txns.append(txn)
        processed_ids.add(mid)
        if len(sample) < 3:
            sample.append(txn.dict())

    # export & record run
    exporter.export(txns, run_id)
    last_ts = txns[-1].date.strftime("%Y-%m-%dT%H:%M:%SZ") \
                if txns else fetch_after or ""
    exporter.log_run(run_id, run_ts, last_ts, len(msgs), len(txns))

    # optional downstream loads
    load_postgres()
    load_snowflake()

    LOG.info("[Run %s] total=%d parsed=%d dup=%d fail=%d",
             run_id, len(msgs), len(txns), dup, fail)


# CLI -----------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date")
    ap.add_argument("--max-results", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    run(a.start_date, a.max_results, a.force)