"""
APScheduler wrapper. Launch with:

    poetry run python -m pipeline.scheduler

Reads `config/scheduler.interval`.
"""

import logging
from datetime import datetime, timedelta

from apscheduler.schedulers.blocking import BlockingScheduler
import yaml
#from pipeline.run_once import run_once
from pipeline.run_pipeline import run_pipeline
from core.config_loader import load_config

logging.basicConfig(level=logging.INFO)
CFG = load_config(open("config/config.yaml"))

sched = BlockingScheduler()


def add_job():
    interval = CFG["scheduler"]["interval"]
    if interval == "hourly":
        sched.add_job(run_pipeline, "interval", hours=24)
    elif interval == "daily":
        sched.add_job(run_pipeline, "cron", hour=24)  # default 03:00
    elif interval == "cron":
        expr = CFG["scheduler"]["cron_expr"]
        sched.add_job(run_pipeline, "cron", **{
            k: v for k, v in zip(
                ["minute", "hour", "day", "month", "day_of_week"], expr.split()
            )
        })
    else:
        raise ValueError("Bad scheduler.interval")

add_job()

if __name__ == "__main__":
    logging.info("Scheduler started.")
    sched.start()
