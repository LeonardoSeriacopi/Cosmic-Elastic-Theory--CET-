#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET — Orbital Period Bootstrap Validation
Confirms stability of detected modulation period across bootstrap resamples.

Input: CSV with 'phase' and 'intensity' columns.
Output:
  - CSV of all bootstrap detections
  - Summary table with mean, std, CI
  - Histogram plot
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

def detect_period(time, intensity, fmin=1e-3, fmax=10.0):
    """Runs Lomb–Scargle and returns best period and power."""
    ls = LombScargle(time, intensity - np.nanmean(intensity))
    freq, power = ls.autopower(minimum_frequency=fmin, maximum_frequency=fmax)
    best_period = 1.0 / freq[np.argmax(power)]
    return best_period, np.max(power)

def main():
    ap = argparse.ArgumentParser(description="CET bootstrap validation of orbital period")
    ap.add_argument("--infile", required=True, help="Input CSV with phase,intensity columns")
    ap.add_argument("--pb_ref", type=float, default=None, help="Reference orbital period (days)")
    ap.add_argument("--n_boot", type=int, default=200, help="Number of bootstrap iterations")
    ap.add_argument("--frac", type=float, default=0.8, help="Fraction of data per bootstrap sample")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    if "phase" not in df.columns or "intensity" not in df.columns:
        raise ValueError("Input must contain columns 'phase' and 'intensity'.")

    phase = df["phase"].to_numpy()
    intensity = df["intensity"].to_numpy()

    rng = np.random.default_rng(42)
    boot_periods = []
    boot_powers = []

    for i in range(args.n_boot):
        idx = rng.choice(len(phase), size=int(args.frac * len(phase)), replace=True)
        t = phase[idx]
        y = intensity[idx]
        p, pow_ = detect_period(t, y, fmin=1e-3, fmax=10)
        boot_periods.append(p)
        boot_powers.append(pow_)

    boot_df = pd.DataFrame({"bootstrap_period_days": boot_periods, "power": boot_powers})
    boot_df.to_csv(f"{args.outdir}/bootstrap_periods.csv", index=False)

    # summary statistics
    mean_p = np.mean(boot_periods)
    std_p = np.std(boot_periods, ddof=1)
    ci_low, ci_high = np.percentile(boot_periods, [2.5, 97.5])
    summary = pd.DataFrame({
        "mean_period_days": [mean_p],
        "std_days": [std_p],
        "ci_low_days": [ci_low],
        "ci_high_days": [ci_high],
        "n_boot": [args.n_boot],
        "pb_ref_days": [args.pb_ref]
    })
    summary.to_csv(f"{args.outdir}/bootstrap_summary.csv", index=False)

    # plot histogram
    plt.figure(figsize=(8,5))
    plt.hist(boot_periods, bins=20, alpha=0.7, color="C0", label="Bootstrap detections")
    plt.axvline(mean_p, color="red", lw=2, label=f"Mean = {mean_p:.3f} d")
    if args.pb_ref:
        plt.axvline(args.pb_ref, color="purple", ls="--", label=f"Reference P_b = {args.pb_ref:.3f} d")
    plt.xlabel("Detected period (days)")
    plt.ylabel("Count")
    plt.title("CET — Bootstrap Stability of Orbital Modulation Period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/bootstrap_hist.png", dpi=150)
    plt.close()

    print(f"✅ Bootstrap validation complete.")
    print(f"→ Mean detected period: {mean_p:.3f} ± {std_p:.3f} d (95% CI: {ci_low:.3f}–{ci_high:.3f})")

if __name__ == "__main__":
    main()