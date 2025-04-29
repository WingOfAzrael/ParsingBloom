import argparse
import csv
from gmail.client import GmailClient
from parser.llm_parser import LLMParser
from core.classifier import TransactionClassifier
from core.exporter import TransactionExporter
from core.run_tracker import RunTracker
from core.flagged_tracker import FlaggedTracker
from utils.invoice_saver import InvoiceSaver
from utils.email_query_builder import build_gmail_query

def run(start_date=None):
    gmail       = GmailClient()
    exporter    = TransactionExporter()
    classifier  = TransactionClassifier()
    parser      = LLMParser()
    saver       = InvoiceSaver()
    run_tracker = RunTracker()
    flagger     = FlaggedTracker()

    # determine fetch boundary
    last_scrape = exporter.get_last_scrape()
    fetch_after = start_date or last_scrape

    # log run start
    run_id, run_ts = run_tracker.start_run(start_date)

    # Build Gmail query
    query = build_gmail_query()

    # ─── Format “after:” as Gmail-friendly date only ──────────────────────
    formatted_after = None
    if fetch_after:
        date_only = fetch_after.split("T", 1)[0]        # e.g. "2025-04-01"
        formatted_after = date_only.replace("-", "/")   # e.g. "2025/04/01"

    # fetch & process
    msgs = gmail.fetch_messages(
        query=query,
        after=formatted_after,
        max_results=500
    )
    print(f"Fetched {len(msgs)} messages (q={query!r}, after={formatted_after!r})")
    fetched = len(msgs)
    processed = 0
    txns = []

    for msg in msgs:
        email_id = msg['id']
        if exporter.has_processed(email_id):
            continue

        parsed = parser.parse(msg)
        if not parsed:
            sender  = next((h['value'] for h in msg['payload']['headers']
                            if h['name']=="From"), "")
            subject = next((h['value'] for h in msg['payload']['headers']
                            if h['name']=="Subject"), "")
            flagger.flag(run_id, email_id, sender, subject,
                         "no transactions parsed")
            continue

        for t in parsed:
            t.transaction_type = classifier.classify(t)
            t.run_id = run_id
            saver.save_attachments(gmail.service, msg, t)
            txns.append(t)
            processed += 1

    exporter.export(txns)

    last_time = (
        txns[-1].timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')
        if txns else last_scrape
    )
    run_tracker.end_run(fetched, processed, last_time)

    print(f"[Run {run_id}] fetched={fetched}, processed={processed}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--start-date', help="YYYY-MM-DD for one-time backfill")
    args = p.parse_args()
    run(args.start_date)
