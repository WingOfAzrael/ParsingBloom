import os
import csv

class FlaggedTracker:
    def __init__(self, flagged_csv='data/flagged_messages.csv'):
        self.flagged_csv = flagged_csv
        os.makedirs(os.path.dirname(self.flagged_csv), exist_ok=True)
        if not os.path.exists(self.flagged_csv):
            with open(self.flagged_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['run_id', 'email_id', 'sender', 'subject', 'reason'])
                writer.writeheader()

    def flag(self, run_id, email_id, sender, subject, reason):
        with open(self.flagged_csv, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['run_id', 'email_id', 'sender', 'subject', 'reason'])
            writer.writerow({
                'run_id': run_id,
                'email_id': email_id,
                'sender': sender,
                'subject': subject,
                'reason': reason
            })