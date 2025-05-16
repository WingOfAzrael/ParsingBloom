
import os
import csv
import uuid
from datetime import datetime

class RunTracker:
    def __init__(
        self,
        runs_csv: str = "data/runs.csv",
        metadata_csv: str = "data/metadata.csv"
    ):
        self.runs_csv = runs_csv
        self.metadata_csv = metadata_csv
        os.makedirs(os.path.dirname(self.runs_csv), exist_ok=True)

        # Initialize runs log if missing
        if not os.path.exists(self.runs_csv):
            with open(self.runs_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'run_id','run_timestamp','start_date',
                    'fetched_messages','processed_transactions',
                    'run_hash','config_hash','prompt_hash','model_digest'
                ])

        # Initialize metadata (unused except for legacy last_scrape)
        if not os.path.exists(self.metadata_csv):
            with open(self.metadata_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['last_scrape_time'])

    def start_run(self, start_date=None):
        self.run_id        = uuid.uuid4().hex
        self.start_date    = start_date or ""
        self.run_timestamp = datetime.utcnow().isoformat()
        return self.run_id, self.run_timestamp

    def end_run(self, fetched: int, processed: int, last_scrape_time: str):
        # Append basic counts; fingerprint columns populated later
        with open(self.runs_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.run_id,
                self.run_timestamp,
                self.start_date,
                fetched,
                processed,
                "","", "",""  # placeholders for run_hash etc.
            ])
        # Update last_scrapeTime
        with open(self.metadata_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['last_scrape_time'])
            writer.writerow([last_scrape_time])