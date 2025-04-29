import schedule
import time
from pipeline.run_pipeline import run

# every hour on the hour
schedule.every().hour.at(":00").do(run)

if __name__ == "__main__":
    print("Scheduler started—Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(1)