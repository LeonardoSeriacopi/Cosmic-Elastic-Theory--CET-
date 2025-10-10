#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET — Empirical Significance Test for Variance Peaks (with σ column)
Evaluates how likely the observed variance peaks are under random phase–intensity shuffling.
Adds sigma-equivalent significance for each phase bin.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import norm
from tqdm import trange  # progress bar

# =========================================================
# Parameters
# =========================================================
phase_file = "results_CET/phase_intensity.csv"
variance_file = "results_CET/variance_by_phase.csv"
out_dir = "results_CET"
n_bootstrap = 2000
half_width = 0.05
random_seed = 42

# =========================================================
# Load data
# =========================================================
phase_df = pd.read_csv(phase_file)
var_df = pd.read_csv(variance_file)

phases = phase_df["phase"].to_numpy()
intensity = phase_df["intensity"].to_numpy()
centers = var_df["phase_center"].to_numpy()
var_obs = var_df["variance"].to_numpy()

rng = np.random.default_rng(random_seed)
boot_max = []

# =========================================================
# Bootstrap permutations
# =========================================================
print(f"Running {n_bootstrap} bootstrap permutations...")
for _ in trange(n_bootstrap):
    y_perm = rng.permutation(intensity)
    var_perm = []
    for c in centers:
        d = np.abs(((phases - c + 0.5) % 1.0) - 0.5)
        mask = d <= half_width
        if np.sum(mask) >= 10:
            var_perm.append(np.nanvar(y_perm[mask], ddof=1))
        else:
            var_perm.append(np.nan)
    boot_max.append(np.nanmax(var_perm))

boot_max = np.array(boot_max)
p_global = np.mean(boot_max >= np.nanmax(var_obs))

# =========================================================
# Local significance
# =========================================================
boot_mat = np.zeros((n_bootstrap, len(centers))) * np.nan
for b in range(n_bootstrap):
    y_perm = rng.permutation(intensity)
    for i, c in enumerate(centers):
        d = np.abs(((phases - c + 0.5) % 1.0) - 0.5)
        mask = d <= half_width
        if np.sum(mask) >= 10:
            boot_mat[b, i] = np.nanvar(y_perm[mask], ddof=1)

boot_mean = np.nanmean(boot_mat, axis=0)
boot_std  = np.nanstd(boot_mat, axis=0)
p_local = np.array([np.mean(boot_mat[:,i] >= var_obs[i]) for i in range(len(centers))])

# =========================================================
# Convert p-values → sigma equivalents
# =========================================================
sigma_local = norm.ppf(1 - p_local/2)
sigma_local[np.isnan(p_local)] = np.nan

# =========================================================
# Identify significant peaks
# =========================================================
peaks, props = find_peaks(np.nan_to_num(var_obs, nan=-np.inf), prominence=0.5)
sig_peaks = [(centers[i], var_obs[i], p_local[i], sigma_local[i]) 
             for i in peaks if p_local[i] < 0.05]

# Save results
pd.DataFrame({
    "phase_center": centers,
    "variance": var_obs,
    "p_local": p_local,
    "sigma_equiv": sigma_local
}).to_csv(f"{out_dir}/variance_significance_sigma.csv", index=False)

pd.DataFrame(sig_peaks, columns=["phase_peak","variance","p_value","sigma_equiv"]).to_csv(
    f"{out_dir}/variance_significant_peaks_sigma.csv", index=False
)

# =========================================================
# Plot with 95% bootstrap confidence band
# =========================================================
plt.figure(figsize=(9,4.5))
plt.plot(centers, var_obs, lw=2, label="Observed variance")
plt.fill_between(centers, boot_mean-2*boot_std, boot_mean+2*boot_std,
                 alpha=0.2, color="gray", label="95% bootstrap band")

for (ph, v, p, s) in sig_peaks:
    plt.plot(ph, v, "o", color="red")
    plt.text(ph, v, f"p={p:.3f}, {s:.1f}σ", ha="left", va="bottom", fontsize=8)

plt.xlabel("Orbital Phase")
plt.ylabel("Variance of Intensity")
plt.title("CET — Bootstrap Significance of Variance Peaks")
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/variance_significance_sigma.png", dpi=150)
plt.close()

# =========================================================
# Summary
# =========================================================
print(f"\n✅ Done! Global p ≈ {p_global:.3e}")
print(f"Significant peaks saved to {out_dir}/variance_significant_peaks_sigma.csv")
print("σ-equivalent column added successfully.")