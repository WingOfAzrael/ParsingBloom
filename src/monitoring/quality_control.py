#!/usr/bin/env python3

import argparse
import pandas as pd
from statistics import mean
from analysis.clopper_pearson import clopper_pearson

def analyse(meta_csv: str, runs: int | None) -> None:
    """
    Reads the last `runs` entries from `meta_csv` (or all entries if runs is None),
    computes a p-chart (mean ±3σ) and the exact Clopper–Pearson 95% CI for the latest run,
    then prints an OK/ALERT status.
    """
    # Load the full diagnostics log
    df = pd.read_json(meta_csv, lines=True)
    if runs is not None:
        df = df.tail(runs)

    if df.empty:
        print(f"No data in {meta_csv}.")
        return

    # Compute success rates
    df["total"] = df["success"] + df["failed"]
    p = df["success"] / df["total"]

    mu    = p.mean()
    sigma = ((p * (1 - p) / df["total"]).pow(0.5)).mean()
    ucl, lcl = mu + 3 * sigma, mu - 3 * sigma

    last      = p.iloc[-1]
    last_succ = int(df["success"].iloc[-1])
    last_tot  = int(df["total"].iloc[-1])
    ci_low, ci_high = clopper_pearson(last_succ, last_tot, alpha=0.05)

    status = "OK"
    if last > ucl or last < lcl or not (ci_low <= mu <= ci_high):
        status = "ALERT"

    print(
        f"Window={len(p)} runs  Last success-rate={last:.3f}  "
        f"Mean={mu:.3f}  3σ=({lcl:.3f},{ucl:.3f})  "
        f"95%CI=({ci_low:.3f},{ci_high:.3f})  ⇒ {status}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quality-control: compute control limits over run_meta.csv"
    )
    parser.add_argument(
        "--meta-csv",
        default="data/run_meta.csv",
        help="Path to the run_meta JSON-lines file"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=None,
        help=(
            "Number of most-recent runs to include in analysis "
            "(default: all entries in meta-csv)"
        )
    )
    args = parser.parse_args()
    analyse(args.meta_csv, args.runs)