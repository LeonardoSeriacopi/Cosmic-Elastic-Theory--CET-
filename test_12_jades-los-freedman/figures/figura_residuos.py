#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a 2×1 figure for the JADES test:

Top panel:
  ⟨Δμ(z)⟩ with error bars and a CET-like saturating exponential fit.

Bottom panel:
  Residuals Δμ_obs − Δμ_model.

Input:
  jades_dissipacao_curve_binned.csv
  (expected columns:
   z_bin_center, mean_delta_mu, std_delta_mu, N_bin)
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Basic configuration
# -----------------------------

CSV_FILE = "jades_dissipacao_curve_binned.csv"
OUT_PNG  = "jades_cet_dissipation_curve.png"

# Fixed relaxation scale (as in the previous fit)
TAU_FIXED = 3.0

# Minimum requirements for bins
MIN_N_BIN  = 5
MIN_STD_MU = 0.0  # positivity is enforced explicitly


# -----------------------------
# CET-like model (saturating exponential)
# -----------------------------

def cet_model(z, mu_inf, A, tau=TAU_FIXED):
    """
    CET-like saturating exponential model.
    """
    z0 = z.min()
    return mu_inf + A * np.exp(-(z - z0) / tau)


def chi2(mu_obs, mu_err, mu_model):
    """
    Chi-square statistic.
    """
    return np.sum(((mu_obs - mu_model) / mu_err) ** 2)


# -----------------------------
# Load binned data
# -----------------------------

data = np.genfromtxt(CSV_FILE, delimiter=",", names=True)

z_all     = data["z_bin_center"]
mu_all    = data["mean_delta_mu"]
sigma_all = data["std_delta_mu"]
N_all     = data["N_bin"]

# Use only statistically reliable bins
mask = (N_all >= MIN_N_BIN) & (sigma_all > 0.0)

z   = z_all[mask]
mu  = mu_all[mask]
err = sigma_all[mask]

print(f"Total number of bins : {len(z_all)}")
print(f"Bins used in the fit : {len(z)}")
print("Redshift range used :", z.min(), z.max())


# -----------------------------
# CET-like fit (τ fixed)
# Scan over μ_inf and A
# -----------------------------

# Rough initial guesses
mu_inf_guess = mu.min() - 0.001
A_guess      = mu.max() - mu_inf_guess

mu_inf_vals = np.linspace(mu_inf_guess - 0.002, mu_inf_guess + 0.002, 81)
A_vals      = np.linspace(A_guess * 0.5, A_guess * 1.5, 81)

best_chi2   = np.inf
best_mu_inf = None
best_A      = None

for mu_inf_try in mu_inf_vals:
    for A_try in A_vals:
        mu_model_try = cet_model(z, mu_inf_try, A_try, TAU_FIXED)
        if not np.all(np.isfinite(mu_model_try)):
            continue
        c2 = chi2(mu, err, mu_model_try)
        if c2 < best_chi2:
            best_chi2   = c2
            best_mu_inf = mu_inf_try
            best_A      = A_try

dof = len(z) - 2  # two free parameters: μ_inf and A
chi2_red = best_chi2 / dof if dof > 0 else np.nan

print("\n=== CET-like fit parameters (τ fixed) ===")
print(f"mu_inf = {best_mu_inf:.8f}")
print(f"A      = {best_A:.8f}")
print(f"tau    = {TAU_FIXED:.3f} (fixed)")
print(f"chi2   = {best_chi2:.3f}")
print(f"dof    = {dof}")
print(f"chi2red= {chi2_red:.3f}")


# -----------------------------
# Final model and residuals
# -----------------------------

z_plot = np.linspace(z.min(), z.max(), 400)
mu_model_plot = cet_model(z_plot, best_mu_inf, best_A, TAU_FIXED)
mu_model_bins = cet_model(z, best_mu_inf, best_A, TAU_FIXED)

residuals = mu - mu_model_bins


# -----------------------------
# 2×1 figure
# -----------------------------

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 6),
    gridspec_kw={"height_ratios": [3, 1]}
)

# Top panel: ⟨Δμ⟩ and CET-like fit
ax1.errorbar(
    z, mu, yerr=err,
    fmt="o", markersize=4, capsize=3,
    label=r"JADES (LOS/Freedman binned data)"
)
ax1.plot(
    z_plot, mu_model_plot,
    "-", label=r"CET-like model (saturating exponential)"
)

ax1.set_ylabel(r"$\langle \Delta\mu \rangle$")
ax1.legend(loc="best", fontsize=9)
ax1.grid(True, alpha=0.3)

# Bottom panel: residuals
ax2.axhline(0.0, linestyle="--", linewidth=1)
ax2.errorbar(
    z, residuals, yerr=err,
    fmt="o", markersize=4, capsize=3
)

ax2.set_xlabel(r"$z$")
ax2.set_ylabel(r"$\Delta\mu - \Delta\mu_{\rm model}$")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"\nFigure saved as: {OUT_PNG}")
plt.close(fig)