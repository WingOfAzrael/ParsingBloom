# pipeline/tasks.py

import logging
from typing import List, Dict, Any, Optional

from connectors.registry import load_connector
from parser.registry import load_parsers
from core.exporter import TransactionExporter
from core.flagged_tracker import FlaggedTracker
from core.models import Transaction
from config.config_loader import load_config

CFG = load_config()
DEFAULT = CFG.default_connector
_CONNECTOR = load_connector(DEFAULT, CFG.connectors[DEFAULT])
_LOG = logging.getLogger(__name__)

# build maps once
_account_cfg_map = {
    acct.internal_account_number: acct
    for acct in CFG.accounts
}

def fetch_messages(
    since: Optional[str],
    max_results: Optional[int] = None,
    query: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Pull messages from Gmail (or other connector), honoring overrides
    for since, max_results and query.
    """
    default_name = CFG.default_connector               # e.g. "gmail" or "outlook"
    conn_cfg     = CFG.connectors[default_name] 

    max_results = max_results if max_results is not None else conn_cfg.max_results
    query = query if query is not None else conn_cfg.query

    _LOG.info("Starting fetch_messages", extra={
        "since": since,
        "max_results": max_results,
        "query": query,
    })
    msgs = _CONNECTOR.fetch_messages(
        since=since,
        max_results=max_results,
        query=query
    )
    _LOG.info("Listed message IDs", extra={"count": len(msgs)})
    return msgs


def parse_message(
    raw_email: Dict[str, Any],
    run_id: str
) -> Optional[Transaction]:
    """
    Try each registered parser in turn. On success, enrich with internal-account
    info, stamp run_id, and return a Transaction. If all fail, flag the message.
    """
    parsers = load_parsers()

    # Optionally extract PDF text
    pdf_text = None
    if CFG.scraper.parse_pdfs:
        attachment_id = (
            raw_email.get("payload", {})
                     .get("parts", [{}])[0]
                     .get("body", {})
                     .get("attachmentId")
        )
        pdf_text = _CONNECTOR.fetch_pdf_attachment(raw_email, attachment_id)

    for parser in parsers:
        try:
            ## 1) parse into a Transaction model (may have only date + last-4)
            #txn = parser.parse(raw_email, pdf_text)
#
            ## 2) enrich from accounts.csv: full account, name, institution, currency
            #acct_num = txn.internal_account_number
            #acct_cfg = _account_cfg_map.get(acct_num)
            #if acct_cfg:
            #    txn.internal_entity = acct_cfg.internal_entity
            #    txn.institution      = acct_cfg.institution
            #    txn.currency         = acct_cfg.currency
#
            ## 3) stamp run_id and return
            #txn.run_id = run_id

            txn = parser.parse(raw_email, pdf_text, run_id)
            return txn

        except Exception as e:
            _LOG.warning("Parser error", extra={
                "parser": parser.__class__.__name__,
                "error": str(e),
                "email_id": raw_email.get("id"),
            })

    # All parsers failed → flag it
    headers = {h["name"]: h["value"] for h in raw_email
               .get("payload", {})
               .get("headers", [])}
    email_id = raw_email.get("id", "")
    sender   = headers.get("From", "")
    subject  = headers.get("Subject", "")
    reason   = "All parsers failed"

    _LOG.info("Flagged unparsed email", extra={"email_id": email_id})
    FlaggedTracker().flag(run_id, email_id, sender, subject, reason)
    return None



def export_transactions(parsed: List[Optional[Transaction]]) -> int:
    """
    Write out all non-None transactions and return how many were written.
    """
    txns = [t for t in parsed if t is not None]
    exporter = TransactionExporter()
    count = exporter.export(txns)
    _LOG.info("Exported transactions", extra={"count": count})
    return count