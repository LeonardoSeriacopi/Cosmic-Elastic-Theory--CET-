import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================

INPUT_CSV = "sdss_desi_cet_catalog.csv"
OUT_DIR   = "sdss_desi_cet_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

# CET macro redshift range
Z_MIN = 0.03
Z_MAX = 0.70

# Binning parameters for <ΔD>(z)
N_BINS_Z = 20
MIN_PER_BIN = 100


# =========================================================
# BINNING FUNCTION
# =========================================================

def bin_by_z(z, DeltaD, z_min, z_max, n_bins=20, min_per_bin=50):
    """
    Compute binned mean ΔD(z) over redshift.

    Returns
    -------
    z_ctr : centers of z bins
    d_mean : mean ΔD per bin
    d_err : standard error of the mean
    Nbin : number of objects per bin
    """

    z = np.asarray(z)
    DeltaD = np.asarray(DeltaD)

    mask = (z >= z_min) & (z <= z_max) & np.isfinite(DeltaD)
    z = z[mask]
    DeltaD = DeltaD[mask]

    if z.size == 0:
        raise RuntimeError("No data available in the requested z-range.")

    bins = np.linspace(z_min, z_max, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    z_ctr, d_mean, d_err, Nbin = [], [], [], []

    for i in range(len(bins) - 1):
        m = (z >= bins[i]) & (z < bins[i + 1])
        if np.sum(m) < min_per_bin:
            continue

        sub = DeltaD[m]

        z_ctr.append(centers[i])
        d_mean.append(sub.mean())

        if sub.size > 1:
            d_err.append(sub.std(ddof=1) / np.sqrt(sub.size))
        else:
            d_err.append(0.0)

        Nbin.append(sub.size)

    return (np.array(z_ctr),
            np.array(d_mean),
            np.array(d_err),
            np.array(Nbin))


# =========================================================
# MAIN EXECUTION
# =========================================================

def main():

    # -----------------------------------------------------
    # Load CET catalog
    # -----------------------------------------------------
    print(f"Loading CET catalog: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    required_cols = [
        "ra", "dec", "z", "survey",
        "z_diss", "z_kin", "f_geo",
        "D_LCDM_Mpc", "D_CET_Mpc", "DeltaD_Mpc"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Redshift cut for macro analysis
    mask = (df["z"] >= Z_MIN) & (df["z"] <= Z_MAX)
    df = df.loc[mask].copy()
    print(f"Objects in z-range [{Z_MIN}, {Z_MAX}]: {df.shape}")

    df = df[np.isfinite(df["DeltaD_Mpc"])]
    print(f"After removing NaN ΔD entries: {df.shape}")

    # Extract arrays
    z = df["z"].values
    z_diss = df["z_diss"].values
    z_kin = df["z_kin"].values
    DeltaD = df["DeltaD_Mpc"].values

    # -----------------------------------------------------
    # Global Statistics
    # -----------------------------------------------------
    print("\n=== GLOBAL CET STATISTICS ===")
    print(f"  N objects          = {df.shape[0]}")
    print(f"  <z>                = {z.mean():.4f}")
    print(f"  median(z)          = {np.median(z):.4f}")
    print(f"  <z_diss>           = {z_diss.mean():.4f}")
    print(f"  <z_kin>            = {z_kin.mean():.4f}")
    print(f"  <ΔD> [Mpc]         = {DeltaD.mean():.2f}")
    print(f"  median(ΔD)         = {np.median(DeltaD):.2f}")
    print(f"  min(ΔD)            = {DeltaD.min():.2f}")
    print(f"  max(ΔD)            = {DeltaD.max():.2f}")

    if z.size > 1:
        corr = np.corrcoef(z, DeltaD)[0, 1]
        print(f"  corr(z, ΔD)        = {corr:.4f}")

    # -----------------------------------------------------
    # Survey-wise statistics
    # -----------------------------------------------------
    print("\n=== SURVEY-WISE STATISTICS ===")

    for survey_name, sub in df.groupby("survey"):
        zz = sub["z"].values
        DD = sub["DeltaD_Mpc"].values

        print(f"\nSurvey: {survey_name}")
        print(f"  N                   = {sub.shape[0]}")
        print(f"  <z>                 = {zz.mean():.4f}")
        print(f"  median(z)           = {np.median(zz):.4f}")
        print(f"  <ΔD> [Mpc]          = {DD.mean():.2f}")
        print(f"  median(ΔD)          = {np.median(DD):.2f}")
        print(f"  min(ΔD)             = {DD.min():.2f}")
        print(f"  max(ΔD)             = {DD.max():.2f}")

        if zz.size > 1:
            cc = np.corrcoef(zz, DD)[0, 1]
            print(f"  corr(z, ΔD)         = {cc:.4f}")

    # -----------------------------------------------------
    # Binning ΔD(z)
    # -----------------------------------------------------
    print("\nComputing binned ⟨ΔD⟩(z)...")

    z_ctr, d_mean, d_err, Nbin = bin_by_z(
        df["z"].values,
        df["DeltaD_Mpc"].values,
        Z_MIN,
        Z_MAX,
        n_bins=N_BINS_Z,
        min_per_bin=MIN_PER_BIN
    )

    binned_df = pd.DataFrame({
        "z_bin_center": z_ctr,
        "DeltaD_mean": d_mean,
        "DeltaD_error": d_err,
        "N_in_bin": Nbin
    })

    out_binned = os.path.join(OUT_DIR, "DeltaD_binned.csv")
    binned_df.to_csv(out_binned, index=False)
    print(f"Binned ΔD table saved to: {out_binned}")

    # -----------------------------------------------------
    # PLOTS
    # -----------------------------------------------------

    # (1) Histogram of ΔD
    plt.figure(figsize=(7, 5))
    plt.hist(DeltaD, bins=60, alpha=0.85)
    plt.xlabel(r'$\Delta D = D_{\Lambda CDM} - D_{\rm CET}$ [Mpc]')
    plt.ylabel("Count")
    plt.title("Histogram of ΔD (SDSS + DESI CET)")
    plt.tight_layout()
    hist_path = os.path.join(OUT_DIR, "hist_DeltaD.png")
    plt.savefig(hist_path, dpi=200)
    plt.close()
    print(f"Saved: {hist_path}")

    # (2) ΔD vs z
    plt.figure(figsize=(7.5, 5.5))
    plt.scatter(z, DeltaD, s=1, alpha=0.3, label="Individual galaxies")

    if len(z_ctr) > 0:
        plt.errorbar(z_ctr, d_mean, yerr=d_err, fmt='o', ms=5,
                     label=r'Binned $\langle \Delta D \rangle(z)$')

    plt.xlabel("Redshift z")
    plt.ylabel(r'$\Delta D$ [Mpc]')
    plt.title(r'$\Delta D$ vs. $z$ (SDSS + DESI CET)')
    plt.legend()
    plt.tight_layout()
    scatter_path = os.path.join(OUT_DIR, "DeltaD_vs_z.png")
    plt.savefig(scatter_path, dpi=200)
    plt.close()
    print(f"Saved: {scatter_path}")

    # (3) D_LCDM vs D_CET
    D_LCDM = df["D_LCDM_Mpc"].values
    D_CET  = df["D_CET_Mpc"].values

    plt.figure(figsize=(7, 5.5))
    plt.scatter(D_LCDM, D_CET, s=1, alpha=0.3)
    dmin = min(D_LCDM.min(), D_CET.min())
    dmax = max(D_LCDM.max(), D_CET.max())
    plt.plot([dmin, dmax], [dmin, dmax], '--')

    plt.xlabel(r'$D_{\Lambda CDM}$ [Mpc]')
    plt.ylabel(r'$D_{\rm CET}$ [Mpc]')
    plt.title("CET vs ΛCDM Distances")
    plt.tight_layout()
    comp_path = os.path.join(OUT_DIR, "DCET_vs_DLCDM.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"Saved: {comp_path}")

    print("\n=== ANALYSIS COMPLETE ===")
    print(f"All outputs written to folder: {OUT_DIR}")


if __name__ == "__main__":
    main()