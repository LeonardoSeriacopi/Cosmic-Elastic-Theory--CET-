#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CET robustness test for SDSS + DESI catalog

This script assumes that the catalog "sdss_desi_cet_catalog.csv"
has already been produced by the main CET pipeline, with columns:

    ra, dec, z, survey, z_diss, z_kin, f_geo,
    D_LCDM_Mpc, D_CET_Mpc, DeltaD_Mpc

We test the robustness of the CET macroscopic distance relation
against realistic observational uncertainties:

  - spectroscopic redshift noise on z_obs
  - environmental scatter on the dissipative component z_diss

F_DISS is kept fixed (it is a property of the medium, not a free
per-object parameter). The test recomputes CET distances using a
linear D_CET ∝ z_kin mapping consistent with the catalog and
measures how much the binned ΔD(z) curve moves.

Outputs:
  - CSV table with reference and MC-perturbed ⟨ΔD⟩(z)
  - Plot showing reference curve and Monte Carlo band
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 0) Configuration
# -------------------------------------------------------------

INPUT_CATALOG = "sdss_desi_cet_catalog.csv"
OUT_DIR = "sdss_desi_cet_robustness"

os.makedirs(OUT_DIR, exist_ok=True)

# Redshift range to analyse (same as in the main analysis)
Z_MIN = 0.03
Z_MAX = 0.70

# Number of redshift bins for ⟨ΔD⟩(z)
N_BINS = 20

# Monte Carlo settings
N_REAL = 500            # number of MC realizations
SEED = 12345            # random seed for reproducibility

# Observational-like uncertainties
SIGMA_Z_SPEC = 5e-4     # typical spectroscopic redshift error (~150 km/s)
SIGMA_ENV_FRAC = 0.03   # 3% fractional scatter on z_diss (environmental)

# Minimum number of objects per bin to keep it
MIN_PER_BIN = 100

np.random.seed(SEED)


# -------------------------------------------------------------
# 1) Load CET catalog and basic selection
# -------------------------------------------------------------
def load_catalog(path):
    print(f"Loading CET catalog: {path}")
    df = pd.read_csv(path)

    required = [
        "z", "z_diss", "z_kin",
        "D_LCDM_Mpc", "D_CET_Mpc", "DeltaD_Mpc"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in catalog: {missing}")

    # Filter in redshift range and drop NaNs in ΔD
    mask = (df["z"] >= Z_MIN) & (df["z"] <= Z_MAX)
    df = df.loc[mask].copy()
    df = df[np.isfinite(df["DeltaD_Mpc"])]

    print(f"Objects in z-range [{Z_MIN:.2f}, {Z_MAX:.2f}]: {df.shape}")
    return df


# -------------------------------------------------------------
# 2) Build reference binned ΔD(z) curve
# -------------------------------------------------------------
def compute_binned_deltaD(z, deltaD, bins):
    """
    Compute mean ΔD in redshift bins.
    Returns:
        bin_centers, DeltaD_mean, DeltaD_std, N_per_bin
    """
    bin_edges = bins
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    DeltaD_mean = np.full_like(bin_centers, np.nan, dtype=float)
    DeltaD_std = np.full_like(bin_centers, np.nan, dtype=float)
    N_per_bin = np.zeros_like(bin_centers, dtype=int)

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        m = (z >= lo) & (z < hi)
        N = m.sum()
        N_per_bin[i] = int(N)
        if N < MIN_PER_BIN:
            continue
        vals = deltaD[m]
        DeltaD_mean[i] = np.nanmean(vals)
        DeltaD_std[i] = np.nanstd(vals, ddof=1)

    return bin_centers, DeltaD_mean, DeltaD_std, N_per_bin


# -------------------------------------------------------------
# 3) Monte Carlo robustness test
# -------------------------------------------------------------
def run_robustness(df, bins):
    """
    Perform Monte Carlo by perturbing:
      - z_obs (spectroscopic noise)
      - z_diss (environmental fractional noise)
    while keeping F_DISS fixed at its catalog value for each object.

    Distances:
      D_LCDM is kept fixed (cosmology is not changed).
      D_CET is recomputed using a linear mapping D_CET = A * z_kin,
      where A is calibrated from the catalog itself.
    """

    # Extract arrays
    z_obs = df["z"].values.astype(float)
    z_diss_ref = df["z_diss"].values.astype(float)
    z_kin_ref = df["z_kin"].values.astype(float)
    D_LCDM = df["D_LCDM_Mpc"].values.astype(float)
    D_CET_ref = df["D_CET_Mpc"].values.astype(float)
    DeltaD_ref = df["DeltaD_Mpc"].values.astype(float)

    N = z_obs.size
    print(f"\nTotal objects used in robustness test: N = {N}")

    # Calibrate the linear CET distance mapping: D_CET ≈ A * z_kin
    valid = (z_kin_ref > 0) & np.isfinite(D_CET_ref) & np.isfinite(z_kin_ref)
    if valid.sum() < 1000:
        raise RuntimeError("Too few valid points to calibrate CET mapping.")

    A_cet = np.median(D_CET_ref[valid] / z_kin_ref[valid])
    print(f"Calibrated CET distance factor: A_cet = {A_cet:.3f} Mpc per unit z_kin")

    # Build reference binned curve
    bin_edges = bins
    z_bin, DeltaD_ref_mean, _, N_per_bin = compute_binned_deltaD(
        z_obs, DeltaD_ref, bin_edges
    )

    # Prepare arrays for MC binned curves
    nbins = len(z_bin)
    DeltaD_mc_bins = np.full((N_REAL, nbins), np.nan)

    print(
        f"\nRunning Monte Carlo with N_REAL = {N_REAL}, "
        f"σ_z = {SIGMA_Z_SPEC:.1e}, σ_env_frac = {SIGMA_ENV_FRAC:.2f}"
    )

    for r in range(N_REAL):
        # 1) Spectroscopic noise on z_obs
        dz_spec = np.random.normal(loc=0.0, scale=SIGMA_Z_SPEC, size=N)
        z_obs_mc = z_obs + dz_spec

        # 2) Environmental fractional noise on z_diss
        d_env = np.random.normal(loc=0.0, scale=SIGMA_ENV_FRAC, size=N)
        z_diss_mc = z_diss_ref * (1.0 + d_env)

        # 3) Recompute kinematic component
        z_kin_mc = z_obs_mc - z_diss_mc

        # Physical consistency cuts
        invalid = (z_obs_mc <= 0) | (z_kin_mc <= 0)
        z_kin_mc[invalid] = np.nan

        # 4) Recompute CET distance (linear mapping)
        D_CET_mc = A_cet * z_kin_mc

        # 5) New ΔD (LCDM is kept fixed)
        DeltaD_mc = D_LCDM - D_CET_mc

        # 6) Bin in redshift (we use the catalog z_obs as bin coordinate)
        _, DeltaD_mc_mean, _, _ = compute_binned_deltaD(
            z_obs, DeltaD_mc, bin_edges
        )

        DeltaD_mc_bins[r, :] = DeltaD_mc_mean

        if (r + 1) % max(1, N_REAL // 10) == 0:
            print(f"  MC realization {r + 1}/{N_REAL} completed")

    # MC statistics per bin
    DeltaD_mc_mean_all = np.nanmean(DeltaD_mc_bins, axis=0)
    DeltaD_mc_std_all = np.nanstd(DeltaD_mc_bins, axis=0)

    # Global RMD: how far the MC mean moves away from the reference curve
    valid_bins = np.isfinite(DeltaD_ref_mean) & (DeltaD_ref_mean > 0)
    RMD_global = np.nanmean(
        np.abs(DeltaD_mc_mean_all[valid_bins] - DeltaD_ref_mean[valid_bins])
        / DeltaD_ref_mean[valid_bins]
    )

    print("\n=== CET robustness result (realistic noise) ===")
    print("Global RMD(DeltaD) between reference and MC mean:")
    print(f"  RMD_global = {RMD_global:.4f} ({RMD_global*100:.2f}%)")

    # Save table
    out_table = pd.DataFrame(
        {
            "z_bin_center": z_bin,
            "DeltaD_ref": DeltaD_ref_mean,
            "DeltaD_mc_mean": DeltaD_mc_mean_all,
            "DeltaD_mc_std": DeltaD_mc_std_all,
            "N_in_bin": N_per_bin,
        }
    )
    out_csv = os.path.join(OUT_DIR, "DeltaD_robustness_MC.csv")
    out_table.to_csv(out_csv, index=False)
    print(f"\nRobustness table saved to: {out_csv}")

    return (
        z_bin,
        DeltaD_ref_mean,
        DeltaD_mc_mean_all,
        DeltaD_mc_std_all,
        RMD_global,
    )


# -------------------------------------------------------------
# 4) Plot robustness band
# -------------------------------------------------------------
def plot_robustness(z_bin, DeltaD_ref, DeltaD_mc_mean, DeltaD_mc_std):
    plt.figure(figsize=(8, 5.5))
    plt.title("CET robustness test: realistic observational noise")
    # Reference curve from catalog
    plt.plot(
        z_bin,
        DeltaD_ref,
        "o-",
        ms=5,
        label="Reference ⟨ΔD⟩(z) (catalog)",
    )

    # MC band: mean ± 1σ
    y1 = DeltaD_mc_mean - DeltaD_mc_std
    y2 = DeltaD_mc_mean + DeltaD_mc_std
    plt.fill_between(
        z_bin,
        y1,
        y2,
        alpha=0.25,
        label="MC mean ± 1σ (noise on z, z_diss)",
    )

    plt.xlabel("Redshift z")
    plt.ylabel(r"$\Delta D = D_{\Lambda{\rm CDM}} - D_{\rm CET}$ [Mpc]")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, "CET_robustness_band.png")
    plt.savefig(out_png, dpi=200)
    print(f"Robustness plot saved to: {out_png}")
    plt.close()


# -------------------------------------------------------------
# 5) Main
# -------------------------------------------------------------
def main():
    df = load_catalog(INPUT_CATALOG)

    # Define bin edges for redshift
    bins = np.linspace(Z_MIN, Z_MAX, N_BINS + 1)

    (
        z_bin,
        DeltaD_ref,
        DeltaD_mc_mean,
        DeltaD_mc_std,
        RMD_global,
    ) = run_robustness(df, bins)

    plot_robustness(z_bin, DeltaD_ref, DeltaD_mc_mean, DeltaD_mc_std)

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"All robustness outputs written to folder: {OUT_DIR}")
    print(f"Global RMD = {RMD_global*100:.2f}%")

if __name__ == "__main__":
    main()