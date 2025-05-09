import csv
import yaml
from config.config_loader import load_config

def build_gmail_query(config_path="config/config.yaml"):
    cfg = load_config(open(config_path))
    filter_cfg = cfg.get("email_filter", {})

    use_label = filter_cfg.get("use_label", False)
    label = filter_cfg.get("label_name", "banking")
    use_domains = filter_cfg.get("use_domain_filter", False)
    sender_csv = filter_cfg.get("sender_csv", "config/bank_senders.csv")

    parts = []

    if use_label:
        parts.append(f"label:{label}")

    if use_domains:
        try:
            with open(sender_csv) as f:
                domains = [r['sender_domain'] for r in csv.DictReader(f)]
                domain_query = " OR ".join([f"from:{d}" for d in domains])
                if domain_query:
                    parts.append(f"({domain_query})")
        except FileNotFoundError:
            print(f"[email_query_builder] sender_csv not found: {sender_csv}")

    # Combine parts into a Gmail-compatible search query
    final_query = " ".join(parts)
    #final_query = "label:Banking"
    print(f"[email_query_builder] Query = {final_query}")
    return final_query