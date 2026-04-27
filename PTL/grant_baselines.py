"""
Compute dataset statistics + persistence / climatology baseline metrics
on the preprocessed SuperDARN cnvmap pairs.  No model training needed.

Usage:
    python grant_baselines.py [--data_dir ~/rst/preprocessed/g120_f500]
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

CH_NAMES = ["Velocity (m/s)", "Vel.SD (m/s)", "Kvect (°)", "Occupancy", "Density (log n)"]


def load_chunks(data_dir):
    dataA_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("dataA_") and f.endswith(".npy")
    )
    dataB_files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.startswith("dataB_") and f.endswith(".npy")
    )
    return dataA_files, dataB_files


def compute_baselines(dataA_files, dataB_files, device="cpu"):
    n_channels = 5
    total_n = 0

    # running sums for per-channel metrics
    pers_sse   = np.zeros(n_channels, dtype=np.float64)  # persistence (x→y)
    clim_sse   = np.zeros(n_channels, dtype=np.float64)  # climatology (0→y)
    y_sum      = np.zeros(n_channels, dtype=np.float64)  # for mean
    y_sq_sum   = np.zeros(n_channels, dtype=np.float64)  # for std

    # event / quiet split (channel 3 = occupancy)
    event_pers_sse, quiet_pers_sse = 0.0, 0.0
    event_n,        quiet_n        = 0,   0

    t0 = time.time()
    for a_path, b_path in zip(dataA_files, dataB_files):
        A = np.load(a_path, mmap_mode='r').astype(np.float32)  # (N, 5, G, G)
        B = np.load(b_path, mmap_mode='r').astype(np.float32)
        N = A.shape[0]
        total_n += N

        err = (A - B).astype(np.float64)   # persistence error
        for c in range(n_channels):
            pers_sse[c]  += (err[:, c] ** 2).sum()
            clim_sse[c]  += (B[:, c].astype(np.float64) ** 2).sum()
            y_sum[c]     += B[:, c].astype(np.float64).sum()
            y_sq_sum[c]  += (B[:, c].astype(np.float64) ** 2).sum()

        # event vs quiet on raw (un-normalised) occupancy channel
        occ = B[:, 3].astype(np.float64)   # (N, G, G)
        mask_event = occ > 0.5
        mask_quiet = ~mask_event
        sq_pers = err.astype(np.float64) ** 2  # (N, 5, G, G) — recompute here avg over channels
        sq_ch0  = (err[:, 0] ** 2)             # velocity channel only for event split
        event_pers_sse += float((sq_ch0 * mask_event).sum())
        quiet_pers_sse += float((sq_ch0 * mask_quiet).sum())
        event_n        += int(mask_event.sum())
        quiet_n        += int(mask_quiet.sum())

    elapsed = time.time() - t0

    n_pixels = total_n * 120 * 120  # per channel

    pers_mse  = pers_sse / n_pixels
    clim_mse  = clim_sse / n_pixels
    pers_rmse = np.sqrt(pers_mse)
    clim_rmse = np.sqrt(clim_mse)
    y_mean    = y_sum    / n_pixels
    y_std     = np.sqrt(np.maximum(y_sq_sum / n_pixels - y_mean ** 2, 0.0))

    overall_pers_mse = pers_sse.sum() / (n_pixels * n_channels)
    overall_clim_mse = clim_mse.sum() / n_channels

    event_vel_rmse = np.sqrt(event_pers_sse / max(event_n, 1))
    quiet_vel_rmse = np.sqrt(quiet_pers_sse / max(quiet_n, 1))

    return {
        "n_pairs":           total_n,
        "elapsed_s":         elapsed,
        "pers_mse":          pers_mse,
        "pers_rmse":         pers_rmse,
        "clim_mse":          clim_mse,
        "clim_rmse":         clim_rmse,
        "overall_pers_mse":  overall_pers_mse,
        "overall_clim_mse":  overall_clim_mse,
        "y_mean":            y_mean,
        "y_std":             y_std,
        "event_vel_rmse":    event_vel_rmse,
        "quiet_vel_rmse":    quiet_vel_rmse,
        "event_n":           event_n,
        "quiet_n":           quiet_n,
        "n_chunks":          len(dataA_files),
    }


def dataset_temporal_coverage(cnvmap_dir):
    files = sorted(f for f in os.listdir(cnvmap_dir) if f.endswith(".cnvmap"))
    if not files:
        return None, None, len(files)
    # filename format: YYYYMMDDHH.cnvmap
    try:
        first = files[0][:10]
        last  = files[-1][:10]
        start = f"{first[:4]}-{first[4:6]}-{first[6:8]} {first[8:10]}:00 UT"
        end   = f"{last[:4]}-{last[4:6]}-{last[6:8]} {last[8:10]}:00 UT"
    except Exception:
        start, end = files[0], files[-1]
    return start, end, len(files)


def print_report(stats, cnvmap_dir=None):
    sep = "=" * 62

    print(f"\n{sep}")
    print(" SuperDARN ML Forecast — Baseline Evaluation Report")
    print(sep)

    if cnvmap_dir and os.path.isdir(cnvmap_dir):
        start, end, n_files = dataset_temporal_coverage(cnvmap_dir)
        print(f"\n  Dataset")
        print(f"    Raw cnvmap files  : {n_files:>8,}")
        print(f"    Temporal span     : {start}  →  {end}")

    print(f"\n    Preprocessed pairs: {stats['n_pairs']:>8,}  ({stats['n_chunks']} chunks)")
    print(f"    Grid resolution   : 120 × 120 (geographic lat/lon bins)")
    print(f"    Channels          : Velocity · Vel.SD · Kvect · Occupancy · Density")
    print(f"    Eval time         : {stats['elapsed_s']:.1f}s")

    print(f"\n  Channel statistics (target distribution)")
    print(f"    {'Channel':<22s}  {'Mean':>8s}  {'Std':>8s}")
    print(f"    {'-'*44}")
    for c, name in enumerate(CH_NAMES):
        print(f"    {name:<22s}  {stats['y_mean'][c]:>8.3f}  {stats['y_std'][c]:>8.3f}")

    print(f"\n  Persistence baseline  (forecast next = current)")
    print(f"    (lower MSE = less temporal change = harder to improve upon)")
    print(f"    {'Channel':<22s}  {'MSE':>10s}  {'RMSE':>10s}")
    print(f"    {'-'*48}")
    for c, name in enumerate(CH_NAMES):
        print(f"    {name:<22s}  {stats['pers_mse'][c]:>10.5f}  {stats['pers_rmse'][c]:>10.5f}")
    print(f"    {'Overall (all ch.)':<22s}  {stats['overall_pers_mse']:>10.5f}")

    print(f"\n  Climatology baseline  (forecast next = 0, all channels)")
    print(f"    {'Channel':<22s}  {'MSE':>10s}  {'RMSE':>10s}")
    print(f"    {'-'*48}")
    for c, name in enumerate(CH_NAMES):
        print(f"    {name:<22s}  {stats['clim_mse'][c]:>10.5f}  {stats['clim_rmse'][c]:>10.5f}")
    print(f"    {'Overall (all ch.)':<22s}  {stats['overall_clim_mse']:>10.5f}")

    # occupancy breakdown
    total_cells = stats["event_n"] + stats["quiet_n"]
    occ_pct = 100.0 * stats["event_n"] / max(total_cells, 1)
    print(f"\n  Radar coverage (velocity channel)")
    print(f"    Active cells  : {stats['event_n']:>12,}  ({occ_pct:.1f} % of all grid cells)")
    print(f"    Quiet  cells  : {stats['quiet_n']:>12,}  ({100-occ_pct:.1f} %)")
    print(f"    Persistence RMSE — active : {stats['event_vel_rmse']:.5f} m/s")
    print(f"    Persistence RMSE — quiet  : {stats['quiet_vel_rmse']:.5f} m/s")

    print(f"\n  Interpretation for grant proposal")
    vel_idx = 0
    pers_rmse_vel = stats["pers_rmse"][vel_idx]
    clim_rmse_vel = stats["clim_rmse"][vel_idx]
    print(f"    The persistence forecast achieves RMSE = {pers_rmse_vel:.4f} on the")
    print(f"    normalised velocity channel.  A trained ML model that beats this")
    print(f"    score demonstrates genuine predictive skill beyond 'repeat the last")
    print(f"    observation'.  The climatology RMSE ({clim_rmse_vel:.4f}) bounds the")
    print(f"    hardest-possible baseline (zero signal knowledge).")
    print(f"\n    Skill score formula (used during model training):")
    print(f"      skill_persistence = 1 - model_MSE / persistence_MSE")
    print(f"      A score > 0 means the model beats the persistence baseline.")
    print(sep + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",    default=os.path.expanduser("~/rst/preprocessed/g120_f500"))
    p.add_argument("--cnvmap_dir",  default=os.path.expanduser("~/rst/extracted_data"))
    args = p.parse_args()

    print(f"Loading chunk list from: {args.data_dir}")
    dataA_files, dataB_files = load_chunks(args.data_dir)
    if not dataA_files:
        print("ERROR: no dataA_*.npy files found. Run preprocessing first.")
        sys.exit(1)
    print(f"  Found {len(dataA_files)} chunks — computing baselines...")

    stats = compute_baselines(dataA_files, dataB_files)
    print_report(stats, cnvmap_dir=args.cnvmap_dir)


if __name__ == "__main__":
    main()
