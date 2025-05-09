import logging
import uuid
import base64
from datetime import datetime

import yaml

from core.exporter import TransactionExporter
from core.db_loader import load_postgres, load_snowflake
from config.config_loader import load_config
from gmail.client import GmailClient
from parser.llm_parser import LLMParser
from utils.invoice_saver import save_attachments
from utils.email_query_builder import build_gmail_query

#logging.basicConfig(level=logging.INFO)

logging.basicConfig(
    level    = logging.DEBUG,
    format   = "%(asctime)s %(levelname)-8s %(name)s %(message)s",
)

CFG = load_config(open("config/config.yaml"))


def run_once():
    run_id = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()

    #gmail_client = GmailClient(CFG)
    gmail_client = GmailClient("config/config.yaml")
    saver = gmail_client.service

    exporter = TransactionExporter()
    parser = LLMParser()
    
    query = build_gmail_query()
    messages = gmail_client.fetch_messages(query=query, max_results=1)
    #logging.info("Fetched %s messages", len(messages))
    #logging.info("Messages:")
    #logging.info(messages)
    #logging.info("=== DEBUG SAMPLE ===")
    txns = []
    duplicates, parse_fail, written = 0, 0, 0
    sample_json = [] 
    for m in messages:
        if exporter.has_processed(m["id"]):
            duplicates += 1
            continue
        import base64, textwrap, pprint
        logging.info(f"single message:{m}\n\n\n\n\n")
        #exit()
        raw_body = base64.urlsafe_b64decode(
        m["payload"].get("body", {}).get("data", b"")
        ).decode("utf-8", errors="ignore")
        #logging.info("----- RAW BODY -----")
        #logging.info(textwrap.shorten(raw_body, 400))
        #logging.info("--------------------\n\n\n\n\n")
        #exit()
        single = parser.parse(m)
        parsed = [single] if single else []
        #logging.info("----- Parsed test` -----")
        #logging.info(parsed)
        #logging.info("--------------------\n\n\n\n\n")
        if parsed:
            txns.extend(parsed)
            # save each txn's attachments (usually one per e-mail)
            sample_json.append(parsed[0].to_dict())
            for t in parsed:
                save_attachments(saver, m, t)
            logging.info("----- Parsed sample` -----")
            logging.info(parsed[0].to_dict())
            logging.info("--------------------\n\n\n\n\n")


        else:
            #exporter.log_flag(run_id, m["id"], m["snippet"], "parse_fail")
            hdr = {h["name"]: h["value"] for h in m["payload"]["headers"]}
            exporter.log_flag(run_id, m["id"], hdr.get("From", ""), hdr.get("Subject", ""), "parse_fail")
            parse_fail += 1

            logging.info("----- Parsed fail` -----")
            logging.info(parse_fail)
            logging.info("--------------------\n\n\n\n\n")
        #exit()
    # This is to test what exactly is being parsed
    print("=== DEBUG SAMPLE (first 3 parsed rows) ===")
    for j in sample_json:
        print(json.dumps(j, indent=2))
    print("=========================================")
    print(f"Duplicates skipped : {duplicates}")
    print(f"Parse failures     : {parse_fail}")
    print(f"Transactions parsed: {len(txns)}")
    # ───────────────────────────────────────────────────────────────────────
    exporter.export(txns, run_id)            # writes CSVs
    written = len(txns)                      # track count
    ...
    print(f"Exporter attempted rows: {written}")
    #exporter.export(txns, run_id)

    last_time = (
    txns[-1].timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    if txns else ""
)
    exporter.log_run(run_id, ts, last_time, len(messages), len(txns))

    load_postgres()
    load_snowflake()

    logging.info(f"Exporter attempted rows: {written}")
    logging.info("Run complete: %d txns", len(txns))


if __name__ == "__main__":
    run_once()