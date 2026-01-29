# compare_relics_vs_random.py
# ============================================================
# CET — RELICS (14) vs RANDOM KiDS (10k) COMPARISON
# Observable: r_peak (Mpc) only — no "transbordo" mixing
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional stats (if scipy exists)
try:
    from scipy.stats import ks_2samp
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# ============================================================
# CONFIG
# ============================================================
RELICS_FILE = "consolidated_relic_sample.csv"

# Choose ONE random file:
# RANDOM_FILE = "cet_environment_classification.csv"
RANDOM_FILE = "anchor_out_merged_with_proxies.csv"

OUTDIR = "comparative_results"
os.makedirs(OUTDIR, exist_ok=True)

# Plot range (keep wide but sane)
XMIN, XMAX = 0.0, 5.2
NBINS = 40

# KDE grid for smooth curves
KDE_GRID = np.linspace(XMIN, XMAX, 600)

# For numeric stability
EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================
def gaussian_kde_1d(x, grid, bw=None):
    """
    Lightweight KDE (Gaussian) to avoid seaborn dependency.
    bw: if None, use Scott-like rule.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return np.full_like(grid, np.nan, dtype=float)

    # bandwidth
    if bw is None:
        std = np.std(x, ddof=1) + EPS
        bw = 1.06 * std * (n ** (-1/5))

    bw = max(bw, 1e-3)
    diffs = (grid[:, None] - x[None, :]) / bw
    dens = np.mean(np.exp(-0.5 * diffs**2), axis=1) / (bw * np.sqrt(2*np.pi))
    return dens


def kde_mode(x, grid):
    d = gaussian_kde_1d(x, grid)
    if np.all(~np.isfinite(d)):
        return np.nan
    return grid[np.nanargmax(d)]


def ecdf(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    xs = np.sort(x)
    ys = np.arange(1, len(xs)+1) / len(xs) if len(xs) else np.array([])
    return xs, ys


def cohen_d(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    v1 = np.var(a, ddof=1); v2 = np.var(b, ddof=1)
    sp = np.sqrt(((len(a)-1)*v1 + (len(b)-1)*v2) / (len(a)+len(b)-2) + EPS)
    return (np.mean(a) - np.mean(b)) / sp


def normalize_columns(df):
    # Normalize weird column names from relics (logM∗ etc.)
    # We'll keep originals but add safe aliases.
    rename_map = {}

    if "Redshift (z)" in df.columns:
        rename_map["Redshift (z)"] = "z_struct"
    if "Idade Estrutural (logρ)" in df.columns:
        rename_map["Idade Estrutural (logρ)"] = "log_rho"
    if "Re (kpc)" in df.columns:
        rename_map["Re (kpc)"] = "Re_kpc"
    if "logM∗" in df.columns:
        rename_map["logM∗"] = "logMstar"
    if "σ (km/s)" in df.columns:
        rename_map["σ (km/s)"] = "sigma_kms"

    df = df.rename(columns=rename_map)
    return df


def load_random(file_path):
    df = pd.read_csv(file_path)

    # Detect peak column
    if "r_peak_Mpc" in df.columns:
        peak_col = "r_peak_Mpc"
    elif "r_peak_mpc" in df.columns:
        peak_col = "r_peak_mpc"
    else:
        raise ValueError("Random file missing r_peak column (expected r_peak_Mpc).")

    # Detect redshift column
    if "Z_center" in df.columns:
        z_col = "Z_center"
    elif "Z" in df.columns:
        z_col = "Z"
    elif "z" in df.columns:
        z_col = "z"
    else:
        z_col = None

    # Keep only what we need + optional extras
    keep = ["ID", peak_col]
    if z_col: keep.append(z_col)
    if "environment" in df.columns: keep.append("environment")
    if "mass_proxy_min" in df.columns: keep.append("mass_proxy_min")

    df = df[keep].copy()
    df = df.rename(columns={peak_col: "r_peak"})
    if z_col:
        df = df.rename(columns={z_col: "z"})
    else:
        df["z"] = np.nan

    df["sample"] = "random"
    return df


# ============================================================
# LOAD DATA
# ============================================================
relics = pd.read_csv(RELICS_FILE)
relics = normalize_columns(relics)

# relic peak column is dist_pico_mpc
if "dist_pico_mpc" not in relics.columns:
    raise ValueError("Relics file missing dist_pico_mpc")

relics = relics.copy()
relics["r_peak"] = relics["dist_pico_mpc"].astype(float)
relics["z"] = relics["z_lens"].astype(float) if "z_lens" in relics.columns else np.nan
relics["sample"] = "relic"

# optional: keep structural age
if "log_rho" not in relics.columns:
    # fallback: try original name
    if "Idade Estrutural (logρ)" in relics.columns:
        relics["log_rho"] = relics["Idade Estrutural (logρ)"]
    else:
        relics["log_rho"] = np.nan

random_df = load_random(RANDOM_FILE)

# Filter to plotting range + finite
relic_peaks = relics["r_peak"].values
rand_peaks  = random_df["r_peak"].values

relic_peaks = relic_peaks[np.isfinite(relic_peaks)]
rand_peaks  = rand_peaks[np.isfinite(rand_peaks)]

# ============================================================
# STATS SUMMARY
# ============================================================
mode_relic = kde_mode(relic_peaks, KDE_GRID)
mode_rand  = kde_mode(rand_peaks, KDE_GRID)

stats_rows = []

def push_stats(label, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    stats_rows.append({
        "sample": label,
        "n": len(x),
        "mean": float(np.mean(x)) if len(x) else np.nan,
        "median": float(np.median(x)) if len(x) else np.nan,
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "mode_kde": float(kde_mode(x, KDE_GRID)) if len(x) else np.nan,
        "q16": float(np.quantile(x, 0.16)) if len(x) else np.nan,
        "q84": float(np.quantile(x, 0.84)) if len(x) else np.nan,
        "min": float(np.min(x)) if len(x) else np.nan,
        "max": float(np.max(x)) if len(x) else np.nan,
    })

push_stats("relic", relic_peaks)
push_stats("random", rand_peaks)

# KS test (optional)
ks_stat, ks_p = np.nan, np.nan
if SCIPY_OK and len(relic_peaks) > 2 and len(rand_peaks) > 10:
    ks = ks_2samp(relic_peaks, rand_peaks, alternative="two-sided", mode="auto")
    ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)

effect_d = cohen_d(relic_peaks, rand_peaks)

comparison = pd.DataFrame(stats_rows)
comparison["ks_stat_relic_vs_random"] = ks_stat
comparison["ks_p_relic_vs_random"] = ks_p
comparison["cohen_d_relic_minus_random"] = effect_d

comparison.to_csv(os.path.join(OUTDIR, "comparison_stats.csv"), index=False)
print("[OK] Saved:", os.path.join(OUTDIR, "comparison_stats.csv"))

# ============================================================
# FIG 1 — OVERLAY HIST + KDE
# ============================================================
plt.figure(figsize=(8.5, 5.5))
plt.hist(rand_peaks, bins=NBINS, range=(XMIN, XMAX), density=True, alpha=0.45, label="Random (10k)")
plt.hist(relic_peaks, bins=NBINS, range=(XMIN, XMAX), density=True, alpha=0.65, label="Relics (14)")

kde_rand  = gaussian_kde_1d(rand_peaks, KDE_GRID)
kde_relic = gaussian_kde_1d(relic_peaks, KDE_GRID)

plt.plot(KDE_GRID, kde_rand, lw=2.0, label="Random KDE")
plt.plot(KDE_GRID, kde_relic, lw=2.0, label="Relics KDE")

plt.axvline(mode_rand,  lw=2.0, linestyle="--", label=f"Random mode ≈ {mode_rand:.2f} Mpc")
plt.axvline(mode_relic, lw=2.0, linestyle="--", label=f"Relics mode ≈ {mode_relic:.2f} Mpc")

plt.xlim(XMIN, XMAX)
plt.xlabel("Dispersion peak radius  r_peak  [Mpc]")
plt.ylabel("Density")
plt.title("Peak Radius Distribution: Relics vs Random Anchors")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig1_peak_distribution_overlay.png"), dpi=220)
plt.close()
print("[OK] Saved fig1")

# ============================================================
# FIG 2 — ECDF (no binning bias)
# ============================================================
xr, yr = ecdf(rand_peaks)
xe, ye = ecdf(relic_peaks)

plt.figure(figsize=(7.8, 5.2))
plt.step(xr, yr, where="post", lw=2.0, label="Random (ECDF)")
plt.step(xe, ye, where="post", lw=2.0, label="Relics (ECDF)")
plt.xlim(XMIN, XMAX)
plt.ylim(0, 1)
plt.xlabel("r_peak [Mpc]")
plt.ylabel("Cumulative fraction")
plt.title("Empirical CDF: Relics vs Random")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig2_ecdf.png"), dpi=220)
plt.close()
print("[OK] Saved fig2")

# ============================================================
# FIG 3 — r_peak vs redshift (scatter)
# ============================================================
plt.figure(figsize=(8.5, 5.5))
if np.isfinite(random_df["z"]).any():
    plt.scatter(random_df["z"], random_df["r_peak"], s=12, alpha=0.25, label="Random (10k)")
plt.scatter(relics["z"], relics["r_peak"], s=70, alpha=0.95, label="Relics (14)")

plt.xlabel("Redshift  z")
plt.ylabel("r_peak [Mpc]")
plt.title("Peak Radius vs Redshift")
plt.xlim(0, max(np.nanmax(random_df["z"].values), np.nanmax(relics["z"].values)) + 0.05)
plt.ylim(XMIN, XMAX)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "fig3_peak_vs_redshift.png"), dpi=220)
plt.close()
print("[OK] Saved fig3")

# ============================================================
# PHYSICAL OCCUPANCY TABLE (no statistics)
# ============================================================

def occupancy_table(label, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    return {
        "sample": label,
        "n": n,
        "frac_r_lt_0p3": np.mean(x < 0.3),
        "frac_0p3_1p0": np.mean((x >= 0.3) & (x < 1.0)),
        "frac_1p0_2p5": np.mean((x >= 1.0) & (x < 2.5)),
        "frac_gt_2p5": np.mean(x >= 2.5),
        "r_max": float(np.max(x)) if n else np.nan,
        "r_median": float(np.median(x)) if n else np.nan,
    }

occ_rows = []
occ_rows.append(occupancy_table("random", rand_peaks))
occ_rows.append(occupancy_table("relics", relic_peaks))

occ_df = pd.DataFrame(occ_rows)
occ_df.to_csv(os.path.join(OUTDIR, "occupancy_physical.csv"), index=False)
print("[OK] Saved: occupancy_physical.csv")

print("\nDONE ✅")
print(f"Outputs in: {OUTDIR}")
if not SCIPY_OK:
    print("Note: scipy not found — KS test skipped (plots/stats still valid).")