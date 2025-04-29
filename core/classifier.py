import csv
import yaml

class TransactionClassifier:
    def __init__(self, config_path="config/config.yaml"):
        cfg = yaml.safe_load(open(config_path))
        acct_csv = cfg['paths']['accounts_csv']

        self.personal = set()
        self.business = set()
        self.last4_to_full = {}
        self.account_to_institution = {}

        with open(acct_csv) as f:
            for r in csv.DictReader(f):
                full = r['internal_account_number']
                last4 = full[-4:]
                self.last4_to_full[last4] = full
                self.account_to_institution[full] = r.get('institution', '')
                if r['owner_type'] == "personal":
                    self.personal.add(full)
                else:
                    self.business.add(full)

    def classify(self, txn):
        # Resolve masked or last4 to full
        digits = "".join(ch for ch in txn.account_number if ch.isdigit())
        if len(digits) == 4 and digits in self.last4_to_full:
            txn.account_number = self.last4_to_full[digits]

        # Attach institution
        txn.institution = self.account_to_institution.get(txn.account_number, "")

        # Classify
        if txn.account_number in self.personal:
            return "personal"
        if txn.account_number in self.business:
            return "business"
        return "unclassified"