#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET — Orbital Calibration via Emission-State Modulation (Lomb–Scargle)

Goal:
    Recover the candidate orbital period P̂_orb from emission-state modulation (intensity or on/off).

Input:
    - CSV file with time (MJD) and intensity (or on/off state)
      e.g. phase_intensity.csv from previous step.

Output:
    - Lomb–Scargle periodogram (PNG)
    - Period candidates (CSV)
    - Bootstrap uncertainty estimate

Usage example:
    python cet_orbital_calibration.py --in results_CET/phase_intensity.csv --pb_ref 5.686
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.timeseries import LombScargle
from tqdm import trange
import argparse

# =========================================================
# Arguments
# =========================================================
ap = argparse.ArgumentParser(description="CET — Orbital period calibration via Lomb–Scargle")
ap.add_argument("--in", dest="infile", required=True, help="Input CSV (with 'intensity' and time column)")
ap.add_argument("--time_col", default="MJD", help="Column with time (MJD or seconds)")
ap.add_argument("--y_col", default="intensity", help="Column with emission metric (intensity or on_off)")
ap.add_argument("--pb_ref", type=float, default=None, help="Reference orbital period in days (for comparison)")
ap.add_argument("--outdir", default="results_CET", help="Output directory")
ap.add_argument("--n_bootstrap", type=int, default=1000, help="Number of bootstrap resamples")
args = ap.parse_args()

# =========================================================
# Load data
# =========================================================
df = pd.read_csv(args.infile)
if args.time_col not in df.columns:
    raise ValueError(f"Column '{args.time_col}' not found in input CSV.")
if args.y_col not in df.columns:
    raise ValueError(f"Column '{args.y_col}' not found in input CSV.")

t = np.array(df[args.time_col], dtype=float)
y = np.array(df[args.y_col], dtype=float)
y = y - np.nanmean(y)

# Normalize time (in days)
if np.nanmax(t) > 1e6:
    # assume seconds
    t = t / 86400.0

# Remove NaNs
mask = np.isfinite(t) & np.isfinite(y)
t, y = t[mask], y[mask]

print(f"Loaded {len(t)} valid points.")

# =========================================================
# Lomb–Scargle
# =========================================================
min_period = 0.5     # days
max_period = 100.0   # days
freq = np.linspace(1/max_period, 1/min_period, 50000)

ls = LombScargle(t, y)
power = ls.power(freq)

best_freq = freq[np.argmax(power)]
best_period = 1.0 / best_freq

# =========================================================
# Bootstrap uncertainty
# =========================================================
rng = np.random.default_rng(42)
boot_periods = []
for _ in trange(args.n_bootstrap, desc="Bootstrap"):
    sample_idx = rng.choice(len(t), size=len(t), replace=True)
    ls_boot = LombScargle(t[sample_idx], y[sample_idx])
    power_boot = ls_boot.power(freq)
    boot_periods.append(1.0 / freq[np.argmax(power_boot)])
boot_periods = np.array(boot_periods)
p_med = np.median(boot_periods)
p_std = np.std(boot_periods)

# =========================================================
# Save candidates
# =========================================================
out_df = pd.DataFrame({
    "best_period_days": [best_period],
    "bootstrap_median": [p_med],
    "bootstrap_std": [p_std],
})
if args.pb_ref:
    out_df["pb_ref_days"] = args.pb_ref
    out_df["rel_error_%"] = 100.0 * abs(best_period - args.pb_ref) / args.pb_ref

out_df.to_csv(f"{args.outdir}/orbital_period_candidate.csv", index=False)

# =========================================================
# Plot
# =========================================================
plt.figure(figsize=(9,5))
plt.plot(1/freq, power, color="black", lw=1)
plt.xlabel("Period (days)")
plt.ylabel("Lomb–Scargle Power")
plt.title("CET — Lomb–Scargle Periodogram (Emission Modulation)")

plt.axvline(best_period, color="red", ls="--", label=f"Detected P̂_orb = {best_period:.3f} d")
if args.pb_ref:
    plt.axvline(args.pb_ref, color="blue", ls=":", label=f"ATNF P_b = {args.pb_ref:.3f} d")

plt.legend()
plt.xscale("log")
plt.tight_layout()
plt.savefig(f"{args.outdir}/orbital_periodogram.png", dpi=150)
plt.close()

# =========================================================
# Summary
# =========================================================
print("\n✅ Lomb–Scargle analysis complete:")
print(f"  Best candidate P̂_orb = {best_period:.3f} days")
print(f"  Bootstrap median ± std = {p_med:.3f} ± {p_std:.3f} days")
if args.pb_ref:
    print(f"  Reference ATNF P_b = {args.pb_ref:.3f} days")
    print(f"  Relative error = {100.0 * abs(best_period - args.pb_ref)/args.pb_ref:.2f}%")
print(f"\nResults saved in: {args.outdir}/")