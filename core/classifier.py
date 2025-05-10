import csv
from pathlib import Path
import yaml
from config.config_loader import load_config


class TransactionClassifier:
    def __init__(self):
        cfg = load_config()  
        acct_csv: Path = cfg.paths.accounts_csv
        if not acct_csv or not acct_csv.exists():
            raise FileNotFoundError(f"Accounts CSV not found at {acct_csv!r}")


        self.last4_to_full, self.account_to_institution = {}, {}
        self.personal, self.business = set(), set()

        with acct_csv.open() as f:
            for r in csv.DictReader(f):
                full = r["internal_account_number"]
                self.last4_to_full[full[-4:]] = full
                self.account_to_institution[full] = r.get("institution", "")
                (self.personal if r["owner_type"] == "personal" else self.business).add(full)

    # ------------------------------------------------------------------ #
    def classify(self, txn):
        digits = "".join(ch for ch in txn.internal_account_number if ch.isdigit())
        if len(digits) == 4 and digits in self.last4_to_full:
            txn.account_number = self.last4_to_full[digits]

        txn.institution = self.account_to_institution.get(txn.internal_account_number, "")
        if txn.internal_account_number in self.personal:
            return "personal"
        if txn.internal_account_number in self.business:
            return "business"
        return "unclassified"