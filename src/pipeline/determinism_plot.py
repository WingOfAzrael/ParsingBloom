#!/usr/bin/env python3
"""
Extended Determinism Plotting Module

Usage:
    python pipeline/determinism_plot.py --dir data/determinism_tests

Generates multiple views of self‐consistency performance.
"""
import argparse
from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt
from statistics import mean, stdev

def read_summary(base):
    summary = base / 'summary.csv'
    runs, p_structs, p_ids, pmins, pmaxs, latencies = [], [], [], [], [], []
    with open(summary) as f:
        for row in csv.DictReader(f):
            runs.append(int(row['run']))
            p_structs.append(float(row['p_struct']))
            p_ids.append(float(row['p_id']))
            pmins.append(float(row['pmin']))
            pmaxs.append(float(row['pmax']))
            latencies.append(float(row['latency_ms']))
    return runs, p_structs, p_ids, pmins, pmaxs, latencies

def read_per_run(base, runs):
    # returns per-run latency lists, struct_match matrix, id_match matrix
    lat_per_run = []
    struct_mat = []
    id_mat = []
    for run in runs:
        latencies = []
        struct_row = []
        id_row = []
        path = base / f'transaction_det_test_{run}' / 'results.csv'
        with open(path) as f:
            for row in csv.DictReader(f):
                latencies.append(float(row['latency_ms']))
                struct_row.append(1 if row['struct_match']=="True" else 0)
                id_row.append(1 if row['id_match']=="True" else 0)
        lat_per_run.append(latencies)
        struct_mat.append(struct_row)
        id_mat.append(id_row)
    return lat_per_run, np.array(struct_mat), np.array(id_mat)

def plot_latency_histogram(all_latencies, out):
    plt.figure()
    plt.hist(all_latencies, bins=20)
    plt.title('Latency Distribution (ms)')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Count')
    plt.savefig(out / 'latency_histogram.png')
    plt.close()

def plot_latency_box(lat_per_run, runs, out):
    plt.figure()
    plt.boxplot(lat_per_run, labels=runs, showfliers=True)
    plt.title('Latency Boxplot per Run')
    plt.xlabel('Run')
    plt.ylabel('Latency (ms)')
    plt.savefig(out / 'latency_boxplot.png')
    plt.close()

def plot_latency_violin(lat_per_run, runs, out):
    plt.figure()
    plt.violinplot(lat_per_run, positions=runs, showextrema=True)
    plt.title('Latency Violin Plot per Run')
    plt.xlabel('Run')
    plt.ylabel('Latency (ms)')
    plt.savefig(out / 'latency_violinplot.png')
    plt.close()

def plot_latency_ecdf(all_latencies, out):
    x = np.sort(all_latencies)
    y = np.arange(1, len(x)+1) / len(x)
    plt.figure()
    plt.plot(x, y, marker='.', linestyle='none')
    plt.title('Latency ECDF')
    plt.xlabel('Latency (ms)')
    plt.ylabel('ECDF')
    plt.savefig(out / 'latency_ecdf.png')
    plt.close()

def plot_determinism_trends(runs, p_structs, p_ids, pmins, pmaxs, out):
    err_low = [pid - lb for pid, lb in zip(p_ids, pmins)]
    err_high = [ub - pid for ub, pid in zip(pmaxs, p_ids)]
    plt.figure()
    plt.errorbar(runs, p_ids, yerr=[err_low, err_high], fmt='-o', label='p_id (95% CI)')
    plt.plot(runs, p_structs, '-s', label='p_struct')
    plt.title('Determinism Across Runs')
    plt.xlabel('Run')
    plt.ylabel('Determinism')
    plt.legend()
    plt.savefig(out / 'determinism_trends.png')
    plt.close()

def plot_bar_chart(runs, p_structs, p_ids, pmins, pmaxs, out):
    x = np.arange(len(runs))
    width = 0.35
    fig, ax = plt.subplots()
    ax.bar(x - width/2, p_structs, width, label='p_struct')
    err = [ [pid - lb for pid, lb in zip(p_ids, pmins)],
            [ub - pid for ub, pid in zip(pmaxs, p_ids)] ]
    ax.bar(x + width/2, p_ids, width, yerr=err, capsize=5, label='p_id')
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_xlabel('Run')
    ax.set_ylabel('Determinism')
    ax.set_title('Structural vs Identity Determinism')
    ax.legend()
    fig.savefig(out / 'determinism_bar.png')
    plt.close(fig)

def plot_struct_heatmap(struct_mat, out):
    plt.figure()
    plt.imshow(struct_mat, aspect='auto', interpolation='nearest')
    plt.title('Schema Match Heatmap\n(1=match, 0=mismatch)')
    plt.xlabel('Email Index')
    plt.ylabel('Run Index')
    plt.colorbar(label='struct_match')
    plt.savefig(out / 'struct_heatmap.png')
    plt.close()

def plot_control_chart(runs, p_ids, out):
    mu = mean(p_ids)
    sigma = stdev(p_ids) if len(p_ids)>1 else 0
    upper = mu + 3*sigma
    lower = mu - 3*sigma
    plt.figure()
    plt.plot(runs, p_ids, marker='o', linestyle='-')
    plt.axhline(mu, label='Mean')
    plt.axhline(upper, linestyle='--', label='+3σ')
    plt.axhline(lower, linestyle='--', label='-3σ')
    plt.title('Control Chart for p_id')
    plt.xlabel('Run')
    plt.ylabel('p_id')
    plt.legend()
    plt.savefig(out / 'control_chart.png')
    plt.close()

def plot_cusum(runs, p_ids, out):
    mu = mean(p_ids)
    deviations = [pid - mu for pid in p_ids]
    cusum = np.cumsum(deviations)
    plt.figure()
    plt.plot(runs, cusum, marker='o')
    plt.title('CUSUM Chart for p_id Deviations')
    plt.xlabel('Run')
    plt.ylabel('Cumulative Sum (p_id - mean)')
    plt.savefig(out / 'cusum_chart.png')
    plt.close()

def plot_scatter_id(id_mat, out):
    plt.figure()
    n_runs, n_emails = id_mat.shape
    for i in range(n_runs):
        for j in range(n_emails):
            if id_mat[i,j]:
                plt.scatter(i+1, j, marker='o')
            else:
                plt.scatter(i+1, j, marker='x')
    plt.title('Per-Email Identity Match (o=match, x=fail)')
    plt.xlabel('Run')
    plt.ylabel('Email Index')
    plt.savefig(out / 'scatter_id_match.png')
    plt.close()

def plot_flag_rate(runs, p_ids, out):
    flag_rate = [1 - p for p in p_ids]
    plt.figure()
    plt.plot(runs, flag_rate, marker='o')
    plt.title('Flag Rate (1 - p_id) Across Runs')
    plt.xlabel('Run')
    plt.ylabel('Flag Rate')
    plt.savefig(out / 'flag_rate_trend.png')
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', required=True, help='Directory of test outputs')
    args = p.parse_args()
    base = Path(args.dir)

    # load data
    runs, p_structs, p_ids, pmins, pmaxs, latencies = read_summary(base)
    lat_per_run, struct_mat, id_mat = read_per_run(base, runs)
    all_lats = [lt for sub in lat_per_run for lt in sub]

    # generate plots
    plot_latency_histogram(all_lats, base)
    plot_latency_box(lat_per_run, runs, base)
    plot_latency_violin(lat_per_run, runs, base)
    plot_latency_ecdf(all_lats, base)

    plot_determinism_trends(runs, p_structs, p_ids, pmins, pmaxs, base)
    plot_bar_chart(runs, p_structs, p_ids, pmins, pmaxs, base)
    plot_struct_heatmap(struct_mat, base)
    plot_control_chart(runs, p_ids, base)
    plot_cusum(runs, p_ids, base)
    plot_scatter_id(id_mat, base)
    plot_flag_rate(runs, p_ids, base)

    print(f"All extended plots saved to {base}")

if __name__ == '__main__':
    main()
