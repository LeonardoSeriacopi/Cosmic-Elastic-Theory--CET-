# -- coding: utf-8 --
"""
Usage:
  python finalize_los_analysis.py posterior_with_density.csv results_los6 \
      --events "GW150914_095045,GW151012_095443,GW170104_101158,GW170823_131358,GW191103_012549,GW191109_010717"

Notes:
- Outputs ONLY PNG images (no PDF).
- All figure labels/titles are in ENGLISH.
- Works with statsmodels if available; otherwise falls back to scikit-learn.
"""

import sys, os, math, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # ensure headless-safe
import matplotlib.pyplot as plt

# Try statsmodels; fallback to sklearn
USE_SM = True
try:
    import statsmodels.api as sm
except Exception:
    USE_SM = False
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

# ----------------------------- CLI ---------------------------------

INFILE = sys.argv[1] if len(sys.argv) > 1 else "posterior_with_density.csv"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "results_los6"
EVARG  = None

# allow --events "a,b,c"
for i, arg in enumerate(sys.argv):
    if arg.strip().lower() == "--events" and i + 1 < len(sys.argv):
        EVARG = sys.argv[i + 1]

os.makedirs(OUTDIR, exist_ok=True)

# --------------------------- helpers --------------------------------

def deg_or_rad(theta_series: pd.Series) -> str:
    """Heuristic: if max(theta) > 2*pi, assume degrees."""
    try:
        m = np.nanmax(pd.to_numeric(theta_series, errors="coerce").values)
    except Exception:
        return "rad"
    return "deg" if (isinstance(m, (int, float)) and m > 2 * math.pi) else "rad"

def pick_los_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Prefer 'losdens_galaxies_per_seg' if present.
    Else try losdens_galaxy_count / losdens_nseg.
    """
    if "losdens_galaxies_per_seg" in df.columns:
        return pd.to_numeric(df["losdens_galaxies_per_seg"], errors="coerce")
    gc = pd.to_numeric(df.get("losdens_galaxy_count", np.nan), errors="coerce")
    ns = pd.to_numeric(df.get("losdens_nseg", np.nan), errors="coerce")
    ratio = gc / ns
    return ratio.replace([np.inf, -np.inf], np.nan)

def load_and_prepare(infile: str, events_keep=None) -> pd.DataFrame:
    df = pd.read_csv(infile, low_memory=False)
    if events_keep:
        df = df[df["event"].isin(events_keep)].copy()

    # Target: SNR
    if "snr_net" in df.columns:
        df = df.rename(columns={"snr_net": "SNR"})
    if "SNR" not in df.columns:
        raise RuntimeError("Missing 'snr_net' (or SNR) column.")

    # cos(theta_jn)
    if "theta_jn" not in df.columns:
        raise RuntimeError("Missing 'theta_jn' column.")
    theta = pd.to_numeric(df["theta_jn"], errors="coerce")
    mode = deg_or_rad(theta)
    th = np.deg2rad(theta) if mode == "deg" else theta
    df["cos_theta_jn"] = np.cos(th)

    # log10(DL)
    if "luminosity_distance" not in df.columns:
        raise RuntimeError("Missing 'luminosity_distance' column.")
    DL = pd.to_numeric(df["luminosity_distance"], errors="coerce").replace(0, np.nan)
    df["log_DL"] = np.log10(DL)

    # chirp mass
    if "chirp_mass" not in df.columns:
        raise RuntimeError("Missing 'chirp_mass' column.")
    df["chirp_mass"] = pd.to_numeric(df["chirp_mass"], errors="coerce")

    # LOS ratio
    df["LOS_ratio"] = pick_los_ratio(df)

    # Clean
    df["SNR"] = pd.to_numeric(df["SNR"], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["SNR", "cos_theta_jn", "log_DL", "chirp_mass", "LOS_ratio"])

    return df

def standardize(series_or_df):
    return (series_or_df - series_or_df.mean()) / series_or_df.std(ddof=0)

def partial_r2(y, X_full, X_reduced):
    """
    Partial R^2 for the removed term(s): R2_full - R2_reduced.
    Returns (delta_R2, R2_full_adj).
    """
    if USE_SM:
        m_full = sm.OLS(y, sm.add_constant(X_full)).fit()
        m_red  = sm.OLS(y, sm.add_constant(X_reduced)).fit()
        return float(m_full.rsquared - m_red.rsquared), float(m_full.rsquared_adj)
    else:
        lrF = LinearRegression().fit(X_full, y)
        r2F = r2_score(y, lrF.predict(X_full))
        lrR = LinearRegression().fit(X_reduced, y)
        r2R = r2_score(y, lrR.predict(X_reduced))
        # pseudo-adj only for full returned above (we return plain R2 as proxy if sklearn)
        return float(r2F - r2R), float(r2F)

def permute_pvalue(y, X_base, x_los, nperm=300, seed=42):
    """
    Permutation test on ΔR² gained by adding LOS_ratio.
    Returns (observed_delta_R2, p_value).
    """
    rng = np.random.default_rng(seed)
    # observed stat
    if USE_SM:
        base_fit = sm.OLS(y, sm.add_constant(X_base)).fit()
        r2_base = base_fit.rsquared
        r2_full = sm.OLS(y, sm.add_constant(pd.concat([X_base, x_los], axis=1))).fit().rsquared
        stat_obs = r2_full - r2_base
    else:
        base = LinearRegression().fit(X_base, y)
        r2_base = r2_score(y, base.predict(X_base))
        full = LinearRegression().fit(pd.concat([X_base, x_los], axis=1), y)
        stat_obs = r2_score(y, full.predict(pd.concat([X_base, x_los], axis=1))) - r2_base

    cnt = 0
    for _ in range(nperm):
        xperm = x_los.sample(frac=1.0, random_state=int(rng.integers(0, 1e9))).reset_index(drop=True)
        yv = y.reset_index(drop=True)
        Xp = X_base.reset_index(drop=True)
        if USE_SM:
            r2b = sm.OLS(yv, sm.add_constant(Xp)).fit().rsquared
            r2f = sm.OLS(yv, sm.add_constant(pd.concat([Xp, xperm], axis=1))).fit().rsquared
            r2p = r2f - r2b
        else:
            r2b = r2_score(yv, LinearRegression().fit(Xp, yv).predict(Xp))
            Xpf = pd.concat([Xp, xperm], axis=1)
            r2f = r2_score(yv, LinearRegression().fit(Xpf, yv).predict(Xpf))
            r2p = r2f - r2b
        if r2p >= stat_obs - 1e-12:
            cnt += 1
    pval = (cnt + 1) / (nperm + 1)
    return float(stat_obs), float(pval)

# ------------------------------ main --------------------------------

def main():
    events_keep = None
    if EVARG:
        events_keep = [e.strip() for e in EVARG.split(",") if e.strip()]

    print(f"[INFO] Reading {INFILE}")
    df = load_and_prepare(INFILE, events_keep)
    print(f"[OK] Valid samples: {len(df):,}")
    if events_keep:
        print(f"[OK] Events: {sorted(set(df['event']))}")

    # Features & target
    cols = ["log_DL", "cos_theta_jn", "chirp_mass", "LOS_ratio"]
    X = df[cols].copy()
    y = df["SNR"].astype(float)

    # Standardize (for comparable betas)
    Xz = standardize(X)
    yz = standardize(y)

    results = {}

    # OLS on standardized data
    if USE_SM:
        fit = sm.OLS(yz, sm.add_constant(Xz)).fit()
        results["R2_adj_std"] = float(fit.rsquared_adj)
        coef = fit.params.to_dict()
        pvals = fit.pvalues.to_dict()
        with open(os.path.join(OUTDIR, "ols_std_summary.txt"), "w", encoding="utf-8") as f:
            f.write(fit.summary().as_text())
    else:
        lr = LinearRegression().fit(Xz, yz)
        yhat = lr.predict(Xz)
        r2 = r2_score(yz, yhat)
        n, k = Xz.shape
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)
        results["R2_adj_std"] = float(r2_adj)
        # reconstruct an intercept on standardized space (mean zero ~ intercept ~ 0)
        coef = {"const": float(- (lr.coef_ @ Xz.mean()) + yz.mean())}
        coef.update({c: float(b) for c, b in zip(cols, lr.coef_)})
        pvals = {k: np.nan for k in ["const"] + cols}

    # save OLS results
    pd.DataFrame([{"beta_" + k: v for k, v in coef.items()}]).to_csv(
        os.path.join(OUTDIR, "ols_std_coefs.csv"), index=False
    )
    with open(os.path.join(OUTDIR, "ols_std_pvalues.json"), "w", encoding="utf-8") as f:
        json.dump(pvals, f, indent=2)

    # Partial R^2 for LOS
    Xbase = X[["log_DL", "cos_theta_jn", "chirp_mass"]]
    Xfull = Xbase.join(X[["LOS_ratio"]])
    dR2, R2_full = partial_r2(y, Xfull, Xbase)
    results["delta_R2_LOS"] = float(dR2)
    results["R2_full_adj_or_plain"] = float(R2_full)

    # Permutation test of ΔR^2 (adding LOS)
    stat_obs, p_perm = permute_pvalue(y, Xbase, X[["LOS_ratio"]], nperm=300)
    results["perm_deltaR2_obs"] = float(stat_obs)
    results["perm_pvalue"] = float(p_perm)

    # Per-event summary
    per_event = (
        df.groupby("event")
        .agg(
            n=("SNR", "size"),
            SNR_p50=("SNR", "median"),
            LOS_p50=("LOS_ratio", "median"),
            logDL_p50=("log_DL", "median"),
            cosT_p50=("cos_theta_jn", "median"),
            mass_p50=("chirp_mass", "median"),
        )
        .reset_index()
        .sort_values("n", ascending=False)
    )
    per_event.to_csv(os.path.join(OUTDIR, "per_event_summary.csv"), index=False)

    # --------- Plots (PNG only, labels in ENGLISH!) ----------
    # Scatter: SNR vs LOS per event
    try:
        plt.figure(figsize=(7, 4.5))
        for ev, dsub in df.groupby("event"):
            plt.scatter(
                dsub["LOS_ratio"], dsub["SNR"],
                s=6, alpha=0.25, label=ev
            )
        plt.xlabel("LOS galaxies per segment")
        plt.ylabel("Network SNR (ringdown proxy)")
        plt.title("SNR vs LOS (sample)")
        # shrink legend if many events
        ncol = 2 if df["event"].nunique() <= 8 else 3
        plt.legend(markerscale=2, fontsize=7, ncol=ncol, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "scatter_SNR_vs_LOS.png"), dpi=180)
        plt.close()
    except Exception:
        pass

    # Bar: standardized betas
    try:
        labels = ["log_DL", "cos_theta_jn", "chirp_mass", "LOS_ratio"]
        betas = [coef.get(k, np.nan) for k in labels]
        colors = ["#ff7f0e", "#2ca02c", "#1f77b4", "#d62728"]
        plt.figure(figsize=(6.5, 3.5))
        plt.bar(labels, betas, color=colors)
        plt.axhline(0, color="k", lw=0.8)
        plt.ylabel("Standardized beta (OLS)")
        plt.title("Predictors of SNR (standardized OLS)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, "standardized_betas.png"), dpi=180)
        plt.close()
    except Exception:
        pass

    # Save summary.json
    with open(os.path.join(OUTDIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Console short summary
    print("===== SHORT SUMMARY =====")
    print(f"Adj. R^2 (standardized): {results['R2_adj_std']:.3f}")
    print(f"ΔR^2 by adding LOS:       {results['delta_R2_LOS']:.4f}")
    print(f"Permutation ΔR^2:         {results['perm_deltaR2_obs']:.4f}  p≈{results['perm_pvalue']:.3f}")
    print(f"[OK] Outputs saved in:    {OUTDIR}")

if __name__ == "__main__":
    main()