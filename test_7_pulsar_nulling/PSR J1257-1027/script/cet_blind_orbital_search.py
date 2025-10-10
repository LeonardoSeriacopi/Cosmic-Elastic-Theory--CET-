#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET — Blind orbital period search via intensity modulation (for isolated pulsars)

Input:
    - One or more PSRFITS files with SUBINT table (e.g., J1257).
    - Extracts mean intensity per subint and builds a combined CSV.
    - Performs Lomb–Scargle periodogram and bootstrap test.

Outputs:
    - phase_intensity_blind.csv   (time, intensity)
    - lombscargle_periodogram.png
    - bootstrap_stability.png
    - blind_period_summary.csv

Usage:
    python cet_blind_orbital_search.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.timeseries import LombScargle
import glob, os

# --- Step 1: extract mean intensity per subint ---
def extract_intensity_from_fits(fname):
    print(f"Processing {fname} ...")
    with fits.open(fname) as hdul:
        subint = None
        for h in hdul:
            if "EXTNAME" in h.header and "SUBINT" in h.header["EXTNAME"]:
                subint = h
                break
        if subint is None:
            print(f"  ⚠ No SUBINT extension found in {fname}. Skipping.")
            return None

        data = subint.data
        nsub = len(data)
        tsub = subint.header.get("TSUBINT", 1.0)
        mjd0 = subint.header.get("STT_IMJD", 0) + subint.header.get("STT_SMJD", 0)/86400.0
        t = np.arange(nsub)*tsub/86400.0 + mjd0

        dat = data["DATA"].astype(float)
        # Flatten polarization, frequency, and bin dimensions
        mean_intensity = dat.mean(axis=(1,2,3))

        return pd.DataFrame({"MJD": t, "intensity": mean_intensity, "source": os.path.basename(fname)})

# --- Step 2: gather all FITS files ---
fits_files = sorted([f for f in glob.glob("*.fits") if not f.endswith(".sf.fits")])

dfs = [extract_intensity_from_fits(f) for f in fits_files]
dfs = [d for d in dfs if d is not None]
if len(dfs) == 0:
    raise RuntimeError("No valid FITS files with SUBINT data found.")
df_all = pd.concat(dfs, ignore_index=True)
df_all.to_csv("phase_intensity_blind.csv", index=False)
print("\n✓ Saved combined intensity data → phase_intensity_blind.csv")

# --- Step 3: Lomb–Scargle periodogram ---
time = df_all["MJD"].to_numpy()
flux = df_all["intensity"].to_numpy()
time = time - np.min(time)

min_period = 0.01  # days
max_period = 10.0  # days
freqs = np.linspace(1/max_period, 1/min_period, 10000)
ls = LombScargle(time, flux - np.mean(flux))
power = ls.power(freqs)

best_freq = freqs[np.argmax(power)]
best_period = 1 / best_freq
print(f"\nDetected dominant period: {best_period:.4f} days")

# --- Step 4: Bootstrap significance ---
rng = np.random.default_rng(42)
n_boot = 1000
peak_vals = []
for _ in range(n_boot):
    y_perm = rng.permutation(flux)
    p = LombScargle(time, y_perm - np.mean(y_perm)).power(freqs)
    peak_vals.append(np.nanmax(p))
peak_vals = np.array(peak_vals)
p_value = np.mean(peak_vals >= np.nanmax(power))
print(f"Bootstrap false alarm probability ≈ {p_value:.4e}")

# --- Step 5: Save results and plots ---
summary = pd.DataFrame({
    "best_period_days": [best_period],
    "bootstrap_p_value": [p_value],
    "n_bootstrap": [n_boot]
})
summary.to_csv("blind_period_summary.csv", index=False)
print("✓ Saved → blind_period_summary.csv")

# Periodogram plot
plt.figure(figsize=(8,4))
plt.plot(1/freqs, power, "k-", lw=1.5)
plt.axvline(best_period, color="red", ls="--", label=f"Best P = {best_period:.3f} d")
plt.xlabel("Trial period [days]")
plt.ylabel("Lomb–Scargle power")
plt.title("CET blind periodogram (intensity modulation)")
plt.legend()
plt.tight_layout()
plt.savefig("lombscargle_periodogram.png", dpi=150)
print("✓ Saved → lombscargle_periodogram.png")

# Bootstrap stability plot
plt.figure(figsize=(6,4))
plt.hist(peak_vals, bins=40, color="gray", alpha=0.7, label="Bootstrap maxima")
plt.axvline(np.nanmax(power), color="red", lw=2, label="Observed peak")
plt.xlabel("Lomb–Scargle peak power")
plt.ylabel("Count")
plt.legend()
plt.title("Bootstrap significance of detected period")
plt.tight_layout()
plt.savefig("bootstrap_stability.png", dpi=150)
print("✓ Saved → bootstrap_stability.png")

print("\n=== CET blind orbital search complete ===")
print(f"Best period: {best_period:.4f} days, p-value ≈ {p_value:.3e}")