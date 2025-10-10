#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET — Full Orbital Calibration Pipeline
---------------------------------------
Detects orbital modulation period from pulsar emission (phase/intensity),
runs bootstrap validation, and summarizes calibration consistency
with the reference orbital period.

Inputs:
  --in <CSV>           CSV containing columns ['phase','intensity']
  --pb_ref <float>     Reference orbital period in days
  --outdir <str>       Output directory

Outputs (inside <outdir>):
  - lombscargle_periodogram.png
  - bootstrap_period_hist.png
  - cet_calibration_summary.png
  - orbital_period_candidate.csv
  - bootstrap_periods.csv
  - bootstrap_summary.csv
  - cet_calibration_summary.csv
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle

# ---------------------------------------------------------------------
def lomb_scargle_period_search(df, y_col="intensity"):
    phase = df["phase"].values
    y = df[y_col].values

    # convert phase [0,1) to pseudo-time in days
    t = phase * np.max(phase)
    freq, power = LombScargle(t, y).autopower(minimum_frequency=1/2000,
                                              maximum_frequency=1/0.1,
                                              samples_per_peak=50)
    periods = 1 / freq
    best_period = periods[np.argmax(power)]

    plt.figure(figsize=(8,4))
    plt.plot(periods, power, color="k")
    plt.xscale("log")
    plt.xlabel("Period (days)")
    plt.ylabel("Lomb–Scargle Power")
    plt.title("CET — Lomb–Scargle Periodogram (Emission Modulation)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "lombscargle_periodogram.png"), dpi=150)

    pd.DataFrame({"best_period_days":[best_period]}).to_csv(
        os.path.join(args.outdir, "orbital_period_candidate.csv"), index=False)

    return best_period, periods, power

# ---------------------------------------------------------------------
def bootstrap_period(df, n_boot=300, frac=0.7, y_col="intensity"):
    rng = np.random.default_rng(42)
    boot_periods = []
    for _ in range(n_boot):
        sample = df.sample(frac=frac, replace=True, random_state=rng)
        t = sample["phase"].values * np.max(sample["phase"])
        y = sample[y_col].values
        freq, power = LombScargle(t, y).autopower(minimum_frequency=1/2000,
                                                  maximum_frequency=1/0.1,
                                                  samples_per_peak=30)
        periods = 1 / freq
        boot_periods.append(periods[np.argmax(power)])
    boot_periods = np.array(boot_periods)

    mean = np.mean(boot_periods)
    std = np.std(boot_periods)
    ci = np.percentile(boot_periods, [2.5, 97.5])

    summary = pd.DataFrame({
        "mean_period_days":[mean],
        "std_period_days":[std],
        "ci_low_days":[ci[0]],
        "ci_high_days":[ci[1]],
        "n_boot":[n_boot]
    })
    summary.to_csv(os.path.join(args.outdir, "bootstrap_summary.csv"), index=False)
    pd.DataFrame({"bootstrap_period_days":boot_periods}).to_csv(
        os.path.join(args.outdir, "bootstrap_periods.csv"), index=False)

    plt.figure(figsize=(8,4))
    plt.hist(boot_periods, bins=30, color="skyblue", edgecolor="k")
    plt.axvline(mean, color="r", lw=2, label=f"Mean = {mean:.3f} d")
    plt.xlabel("Detected period (days)")
    plt.ylabel("Count")
    plt.title("CET — Bootstrap Stability of Orbital Modulation Period")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "bootstrap_period_hist.png"), dpi=150)

    return mean, std, ci

# ---------------------------------------------------------------------
def summarize_calibration(best_period, boot_mean, pb_ref, outdir):
    df = pd.DataFrame({
        "Detected":[best_period],
        "Reference":[pb_ref],
        "Bootstrap_mean":[boot_mean]
    })
    df.to_csv(os.path.join(outdir, "cet_calibration_summary.csv"), index=False)

    plt.figure(figsize=(6,4))
    bars = plt.bar(["Detected","Reference","Bootstrap mean"],
                   [best_period, pb_ref, boot_mean],
                   color=["#4C72B0","#55A868","#C44E52"])
    plt.ylabel("Orbital Period (days)")
    plt.title("CET — Orbital Period Calibration Summary")
    for bar, val in zip(bars, [best_period, pb_ref, boot_mean]):
        plt.text(bar.get_x()+bar.get_width()/2, val+1, f"{val:.3f} d",
                 ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "cet_calibration_summary.png"), dpi=150)

# ---------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="CET Full Orbital Calibration Pipeline")
    ap.add_argument("--in", dest="infile", required=True, help="CSV with phase and intensity columns")
    ap.add_argument("--pb_ref", type=float, required=True, help="Reference orbital period (days)")
    ap.add_argument("--outdir", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.infile)

    best_period, _, _ = lomb_scargle_period_search(df)
    boot_mean, boot_std, ci = bootstrap_period(df)
    summarize_calibration(best_period, boot_mean, args.pb_ref, args.outdir)

    print(f"\n✅ CET Calibration complete.")
    print(f"Detected P_orb = {best_period:.3f} d")
    print(f"Bootstrap mean = {boot_mean:.3f} ± {boot_std:.3f} d")
    print(f"95% CI = [{ci[0]:.3f}, {ci[1]:.3f}] d")
    print(f"Reference P_b = {args.pb_ref:.3f} d\n")