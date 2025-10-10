#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET — Blind Orbital Validation Suite (Final v3)
Performs harmonic, variance, coherence, and instrumental sanity tests
on blind periodicity candidates derived from pulsar emission modulation.
Now includes robust FITS handling and harmonic folding + summary table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import spearmanr
from astropy.timeseries import LombScargle
from astropy.io import fits

# =========================
# Configuration
# =========================
np.seterr(all="ignore")
input_csv = "phase_intensity_blind.csv"
summary_csv = "blind_period_summary.csv"
outdir = "results_CET_blind_validation"
os.makedirs(outdir, exist_ok=True)

# =========================
# Load data
# =========================
df = pd.read_csv(input_csv)
summary = pd.read_csv(summary_csv)
P_best = float(summary["best_period_days"].iloc[0])
freq_best = 1.0 / P_best

print(f"Detected period: {P_best:.5f} days ({P_best*24*60:.1f} minutes)")

# =========================
# 1. Harmonic Consistency Test
# =========================
harmonics = np.arange(1, 6)
freqs = np.linspace(1e-3, 50 / P_best, 2000)
power = LombScargle(df.index, df["intensity"]).power(freqs)

harmonic_power = []
for h in harmonics:
    f_h = h * freq_best
    idx = np.argmin(np.abs(freqs - f_h))
    harmonic_power.append((h, f_h, power[idx]))

harm_df = pd.DataFrame(harmonic_power, columns=["harmonic", "frequency", "power"])
harm_df.to_csv(os.path.join(outdir, "harmonic_analysis.csv"), index=False)

plt.figure(figsize=(8, 4))
plt.plot(1/freqs, power, color="black")
for h, _, _ in harmonic_power:
    plt.axvline(P_best/h, color="red", linestyle="--", alpha=0.5)
plt.xlabel("Trial period [days]")
plt.ylabel("Lomb–Scargle power")
plt.title("CET — Harmonic Analysis of Modulation")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "harmonic_analysis.png"), dpi=150)
plt.close()

# =========================
# 2. Phase-Locked Variance Test
# =========================
phases = (df.index / P_best) % 1
bins = np.linspace(0, 1, 64)
bin_centers = (bins[:-1] + bins[1:]) / 2
var_by_phase = [np.var(df["intensity"][(phases >= bins[i]) & (phases < bins[i+1])])
                for i in range(len(bins)-1)]

var_df = pd.DataFrame({"phase": bin_centers, "variance": var_by_phase})
var_df.to_csv(os.path.join(outdir, "variance_phase_curve.csv"), index=False)

plt.figure(figsize=(8,4))
plt.plot(bin_centers, var_by_phase, lw=2, label="Variance (per phase bin)")
peaks, _ = find_peaks(var_by_phase, prominence=np.nanstd(var_by_phase))
plt.scatter(bin_centers[peaks], np.array(var_by_phase)[peaks], color="red", zorder=3)
plt.xlabel("Orbital Phase (folded)")
plt.ylabel("Variance of Intensity")
plt.title("CET — Phase-Locked Variance Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "variance_phase_curve.png"), dpi=150)
plt.close()

# =========================
# 3. Cross-Observation Coherence
# =========================
fits_files = [f for f in os.listdir() if f.endswith(".fits") and "beam01" in f]

def extract_intensity(f):
    try:
        with fits.open(f, memmap=False) as hdul:
            if "SUBINT" not in hdul:
                return None
            sub = hdul["SUBINT"].data
            if "TSUBINT" not in sub.names or "DATA" not in sub.names:
                return None

            # tempo cumulativo (segundos)
            t = np.cumsum(sub["TSUBINT"])

            # intensidade média: colapsa sobre polarizações e frequências
            data = sub["DATA"]
            if data.ndim == 3:
                # normaliza blocos com diferentes tamanhos
                try:
                    min_pol = min(data.shape[1], np.min([d.shape[0] for d in data]))
                    min_freq = min(data.shape[2], np.min([d.shape[-1] for d in data]))
                    y = np.array([np.nanmean(d[:min_pol, :min_freq]) for d in data])
                except Exception:
                    y = np.nanmean(data, axis=(1,2))
            elif data.ndim == 2:
                try:
                    min_freq = min(data.shape[1], np.min([len(d) for d in data]))
                    y = np.array([np.nanmean(d[:min_freq]) for d in data])
                except Exception:
                    y = np.nanmean(data, axis=1)
            else:
                y = np.asarray(data, dtype=float).ravel()

            return pd.DataFrame({"time": np.ravel(t), "intensity": np.ravel(y)})
    except Exception as e:
        print(f"⚠ Error reading {f}: {e}")
        return None

dfs = []
for f in fits_files:
    df_tmp = extract_intensity(f)
    if df_tmp is not None and not df_tmp.empty:
        dfs.append(df_tmp)

if not dfs or len(dfs) < 2:
    print("⚠ Not enough valid FITS files for coherence test (need at least 2). Skipping section.")
    corr = np.nan
else:
    t1, y1 = dfs[0]["time"], dfs[0]["intensity"]
    t2, y2 = dfs[1]["time"], dfs[1]["intensity"]

    f_grid = np.linspace(1e-3, 50/P_best, 2000)
    p1 = LombScargle(t1, y1).power(f_grid)
    p2 = LombScargle(t2, y2).power(f_grid)
    corr, _ = spearmanr(p1, p2)
    coh_df = pd.DataFrame({"freq": f_grid, "p1": p1, "p2": p2})
    coh_df.to_csv(os.path.join(outdir, "coherence_comparison.csv"), index=False)

    plt.figure(figsize=(8,4))
    plt.plot(1/f_grid, p1, label="Obs 1")
    plt.plot(1/f_grid, p2, label="Obs 2", linestyle="--")
    plt.xlabel("Trial Period [days]")
    plt.ylabel("Lomb–Scargle Power")
    plt.title(f"CET — Cross-Observation Coherence (ρ={corr:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "coherence_comparison.png"), dpi=150)
    plt.close()

# =========================
# 4. Instrumental Timescale Check
# =========================
report_lines = []
for f in fits_files:
    try:
        with fits.open(f, memmap=False) as hdul:
            if "SUBINT" in hdul:
                hdr = hdul["SUBINT"].header
                tsub = hdr.get("TSUBINT", np.nan)
                nsub = hdr.get("NSUBINT", np.nan)
                total = tsub * nsub / 60
                report_lines.append(f"{f}: TSUBINT={tsub:.3f}s × NSUBINT={nsub} -> total {total:.1f} min")
    except Exception as e:
        report_lines.append(f"{f}: error reading header ({e})")

with open(os.path.join(outdir, "instrumental_timescale_check.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
    f.write(f"\n\nDetected cycle: {P_best*24*60:.1f} min")

# =========================
# 5. Harmonic Folding
# =========================
phases = (df.index / P_best) % 1
folded_profiles = []
for h in harmonics:
    phase_h = (df.index / (P_best/h)) % 1
    folded_profiles.append(
        np.array([np.mean(df["intensity"][(phase_h >= bins[i]) & (phase_h < bins[i+1])]) for i in range(len(bins)-1)])
    )

folded_profiles = np.array(folded_profiles)
plt.figure(figsize=(8,4))
for i, h in enumerate(harmonics):
    plt.plot(bin_centers, folded_profiles[i], label=f"{h}× harmonic")
plt.xlabel("Phase")
plt.ylabel("Mean intensity")
plt.title("CET — Harmonic Folding Reinforcement")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "harmonic_folding.png"), dpi=150)
plt.close()

# =========================
# 6. Statistical Summary
# =========================
total_time = len(df)  # index steps (proxy)
cycles_in_obs = total_time / (P_best * 24 * 60)  # rough estimate

summary_dict = {
    "best_period_days": P_best,
    "best_period_minutes": P_best * 24 * 60,
    "num_harmonics": len(harmonics),
    "num_variance_peaks": len(peaks),
    "coherence_rho": corr,
    "cycles_covered": cycles_in_obs,
    "variance_mean": np.nanmean(var_by_phase),
    "variance_std": np.nanstd(var_by_phase),
}
pd.DataFrame([summary_dict]).to_csv(os.path.join(outdir, "validation_summary.csv"), index=False)

print(f"\n✅ All validation outputs saved to '{outdir}/'")