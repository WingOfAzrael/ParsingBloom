import os
import csv

class TransactionExporter:
    def __init__(self,
                 transactions_csv: str = "data/transactions.csv",
                 metadata_csv: str = "data/metadata.csv"):
        self.tx_csv = transactions_csv
        self.meta_csv = metadata_csv
        # ensure files exist
        os.makedirs(os.path.dirname(self.tx_csv), exist_ok=True)
        if not os.path.exists(self.tx_csv):
            with open(self.tx_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "transaction_id","date","internal_account_number",
                    "internal_entity", "institution", "external_entity","amount", "available_balance","currency",
                    "description","transaction_type","source_email","email_id","run_id"
                ])
        if not os.path.exists(self.meta_csv):
            with open(self.meta_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["last_scrape_time"])

    def has_processed(self, email_id: str) -> bool:
        with open(self.tx_csv) as f:
            for r in csv.DictReader(f):
                if r['email_id'] == email_id:
                    return True
        return False

    def get_last_scrape(self) -> str:
        with open(self.meta_csv) as f:
            reader = csv.DictReader(f)
            for r in reader:
                return r['last_scrape_time']
        return None

    def export(self, transactions):
        # determine starting transaction_id
        max_id = 0
        with open(self.tx_csv) as f:
            for r in csv.DictReader(f):
                max_id = max(max_id, int(r['transaction_id']))

        # append sorted transactions
        transactions.sort(key=lambda x: x.timestamp)
        with open(self.tx_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            for i, txn in enumerate(transactions, start=1):
                txn.transaction_id = max_id + i
                writer.writerow(list(txn.to_dict().values()))