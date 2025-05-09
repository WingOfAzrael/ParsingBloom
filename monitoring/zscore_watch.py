
#Z-score anomaly checker: flag days where the transaction count deviates by >3σ from 30-day rolling mean. Intended to run daily after pipeline.


from pathlib import Path
import pandas as pd
from scipy.stats import zscore
import yaml
from config.config_loader import load_config

CFG = load_config(open("config/config.yaml"))
CSV = Path(CFG["paths"]["transactions_csv"])

if not CSV.exists():
    print("No data yet.")
    exit(0)

df = pd.read_csv(CSV, parse_dates=["date"])
daily = df.groupby(df["date"].dt.date).size().rename("tx_count").to_frame()
daily["z"] = zscore(daily["tx_count"])
outliers = daily[daily["z"].abs() > 3]
if not outliers.empty:
    print("Anomalies detected:")
    print(outliers)
else:
    print("No anomalies.")