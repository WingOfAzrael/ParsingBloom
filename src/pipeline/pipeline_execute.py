#!/usr/bin/env python3

#This script executes the pipeline by default or if --runs option in command line =1. It computes multiple runs for structural and identity determinism tests
#on the parsing pipeline if --runs option is some integer N greater than 1 (computes the same batch of emails N times). Compares each run to the baseline (first run).

import argparse 
import logging
import os
import time
import csv
import json
from statistics import mean, stdev
from pathlib import Path
import importlib
from tqdm import trange


from core.config_loader import load_config
from core.run_tracker         import RunTracker
from core.flagged_tracker     import FlaggedTracker
from core.exporter import TransactionExporter
from core.db_loader           import load_postgres, load_snowflake
from parser.llm_parser import LLMParser
from analysis.clopper_pearson import clopper_pearson
from utils.email_query_builder import build_gmail_query
from utils.invoice_saver import save_attachments


def main(start_date: str | None = None,
        max_results: int | None = None,
        force: bool = False,
        **kwargs) -> None:
    

    # Load configuuration
    CFG = load_config()
    det_cfg = CFG.determinism
    struct_thr = det_cfg.struct_threshold
    id_thr = det_cfg.id_threshold
    alpha = det_cfg.alpha
    latency_cv_thr = det_cfg.latency_cv_threshold

    # Prepare output directory
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    # Fetch all messages 
    default = CFG.default_connector
    conn_cfg = CFG.connectors[default]
    mod = importlib.import_module(f"connectors.{default}")
    ConnectorClass = getattr(mod, f"{default.capitalize()}Connector")
    connector = ConnectorClass(conn_cfg)
    query = conn_cfg.query or build_gmail_query()
    max_n = conn_cfg.max_results
    msgs = connector.fetch_messages(
        since=None,
        max_results=max_n,
        query=query
    )
    print(f"Fetched {len(msgs)} messages for determinism testing")

    pdf_texts = [None] * len(msgs)
    #parser = LLMParser()
    device_arg = os.environ["PARSINGFORGE_DEVICE"]       # "cuda" or "cpu"
    # only keep quant_bits if we're on GPU
    qb_arg     = CFG.parser.quant_bits if args.gpus else None
    parser     = LLMParser(device=device_arg, quant_bits=qb_arg)
    run_tracker = RunTracker()
    flagger     = FlaggedTracker()
    # scraping window --------------------------------------------------------
    last_scrape = Path(CFG.paths.metadata_csv).read_text().strip() if Path(CFG.paths.metadata_csv).exists() else None
    fetch_after = None if force else (start_date or last_scrape)
    # ── Single-run export (mirror deploy_local) ─────────────────────────
    if args.runs <= 1:
        run_id, run_ts = run_tracker.start_run(fetch_after)
        print("Running single-batch export (run=1)…")
        # parse everything in one go, speed things up
        txns = parser.parse_batch(msgs, pdf_texts, run_id=run_id)

        processed_ids = set()
        if CFG.scraper.save_attachments:
            for msg, txn in zip(msgs, txns):
                mid = msg["id"]

                # duplicate-in-same-batch check
                if mid in processed_ids:
                    dup += 1
                    continue
                
                try:
                    save_attachments(connector.service, msg, txn)
                except Exception as e:
                    # flag and keep going
                    hdr = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
                    flagger.flag(
                        run_id,
                        mid,
                        hdr.get("From", ""),
                        hdr.get("Subject", ""),
                        f"attachment: {e}"
                    )
                    fail += 1

                processed_ids.add(mid)
        exporter = TransactionExporter()
        exporter.export(txns, run_id)
        last_ts = txns[-1].date.strftime("%Y-%m-%dT%H:%M:%SZ") \
                if txns else fetch_after or ""
        exporter.log_run(run_id, run_ts, last_ts, len(msgs), len(txns))
        dup = fail = 0
        # optional downstream loads
        load_postgres()
        load_snowflake()

        logging.info("[Run %s] total=%d parsed=%d dup=%d fail=%d",
             run_id, len(msgs), len(txns), dup, fail)
        print(f"Written {len(txns)} transactions -> {CFG.paths.master_file}")
        return


    # Baseline run
    print("Running baseline parse (run 1)")
    baseline_outputs = []
    for txn in parser.parse_batch(msgs, pdf_texts, run_id="determinism_1"):
        d = txn.to_dict()
        baseline_outputs.append(json.dumps(d, sort_keys=True))
    N = len(baseline_outputs)

    # Collections for summary
    results = []
    latencies = []
    p_structs = []
    p_ids = []
    ci_bounds = []

    # Replicate runs
    for run_idx in trange(1, args.runs + 1, desc="Determinism runs"):
        print(f"Starting run {run_idx}")
        start = time.time()
        txns = parser.parse_batch(msgs, pdf_texts, run_id=f"determinism_{run_idx}")
        duration = time.time() - start
        latency_ms = duration * 1000
        latencies.append(latency_ms)

        # Prepare run directory
        run_dir = out_base / f"transaction_det_test_{run_idx}"
        run_dir.mkdir(exist_ok=True)
        csv_path = run_dir / 'results.csv'

        # Compare to baseline
        struct_success = 0
        id_success = 0
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['message_id', 'struct_match', 'id_match', 'latency_ms'])
            for msg, txn, base_json in zip(msgs, txns, baseline_outputs):
                d = txn.to_dict()
                j = json.dumps(d, sort_keys=True)
                schema_keys = set(json.loads(base_json).keys())
                schema2_keys = set(d.keys())
                struct_ok = (schema_keys == schema2_keys)
                id_ok = (j == base_json)
                if struct_ok:
                    struct_success += 1
                if id_ok:
                    id_success += 1
                writer.writerow([msg['id'], struct_ok, id_ok, latency_ms])

        # Compute metrics
        p_struct = struct_success / N
        p_id = id_success / N
        pmin, pmax = clopper_pearson(id_success, N, alpha)

        results.append({
            'run': run_idx,
            'p_struct': p_struct,
            'p_id': p_id,
            'pmin': pmin,
            'pmax': pmax,
            'latency_ms': latency_ms
        })
        p_structs.append(p_struct)
        p_ids.append(p_id)
        ci_bounds.append((pmin, pmax))

    # Write summary
    summary_path = out_base / 'summary.csv'
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'run', 'p_struct', 'struct_pass',
            'p_id', 'pmin', 'pmax', 'id_pass',
            'latency_ms', 'latency_cv'
        ])
        cv = stdev(latencies) / mean(latencies) if len(latencies) > 1 else 0.0
        for res, (pmin, pmax) in zip(results, ci_bounds):
            struct_pass = (res['p_struct'] >= struct_thr)
            id_pass = (pmin >= id_thr)
            writer.writerow([
                res['run'], res['p_struct'], struct_pass,
                res['p_id'], pmin, pmax, id_pass,
                res['latency_ms'], cv
            ])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Determinism test runner (1 run = batch export; N>1 = full suite)"
    )
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of replicates (default: 1 → single batch export)")
    parser.add_argument("--out-dir", default="",
                        help="Output directory override (else: data/ or data/determinism_tests/)")
    parser.add_argument("--force", action="store_true",
                        help="Bypass incremental scraper cache")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="Confidence‐interval alpha (default 0.05)")
    parser.add_argument("--gpus", action="store_true",
                        help="Enable GPU for parser")
    args, extra = parser.parse_known_args()

    # Merge any EXTRA_ARGS you forward into args
    args.extra_args = extra
    os.environ["PARSINGFORGE_DEVICE"] = "cuda" if args.gpus else "cpu"
    
    main(args)