# utils/email_query_builder.py

import csv
from pathlib import Path
from typing import Optional
from config.config_loader import load_config


CFG = load_config()

def build_gmail_query(
    since: Optional[str] = None,
    config_path: Optional[str]  = "config/config.yaml"
) -> str:
    """
    Build a Gmail search query string composed of:
      1) label:<label_name>           (if use_label)
      2) (from:domain1 OR from:domain2…)  (if use_domain_filter)
      3) <global scraper.query>       (if set)
      4) after:<since>                (if since is provided)
    """
    

    DEFAULT_CONN  = CFG.default_connector
    conn_cfg     = CFG.connectors[DEFAULT_CONN]
    DEFAULT_QUERY = conn_cfg.query
    DEFAULT_MAX   = conn_cfg.max_results
    ef  = CFG.email_filter

    parts: list[str] = []

    # 1) Label filter
    if ef.use_label and ef.label_name:
        parts.append(f"label:{ef.label_name}")

    # 2) Domain filter
    if ef.use_domain_filter and ef.sender_csv:
        try:
            with open(ef.sender_csv, newline="") as f:
                domains = [row[0].strip() for row in csv.reader(f) if row]
            if domains:
                domain_q = " OR ".join(f"from:{d}" for d in domains)
                parts.append(f"({domain_q})")
        except FileNotFoundError:
            print(f"[email_query_builder] sender_csv not found: {ef.sender_csv}")

    # 3) Global scraper-level override
    if DEFAULT_QUERY:
        parts.append(DEFAULT_QUERY)

    # 4) Date window
    if since:
        # Gmail expects YYYY/MM/DD
        parts.append(f"after:{since}")

    final_query = " ".join(parts)
    print(f"[email_query_builder] Query = {final_query}")
    return final_query
