# run_bootstrap_perm.py
# Uso:
#   python run_bootstrap_perm.py --csv posterior_with_density.csv --outdir results_boot --nboot 500 --nperm 500
import os, argparse, numpy as np, pandas as pd
import matplotlib.pyplot as plt

USE_SM = True
try:
    import statsmodels.api as sm
except Exception:
    USE_SM = False
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

def pick_col(df, options):
    for c in options:
        if c in df.columns:
            return c
    return None

def prepare(df):
    snr_col = pick_col(df, ["snr_net","SNR","network_matched_filter_snr","network_optimal_snr"])
    if not snr_col: raise RuntimeError("SNR column not found.")
    df = df.copy()
    df["SNR"] = pd.to_numeric(df[snr_col], errors="coerce")
    dl_col = pick_col(df, ["luminosity_distance","DL_p50","comoving_distance"])
    if not dl_col: raise RuntimeError("Distance column not found.")
    df["log_DL"] = np.log10(pd.to_numeric(df[dl_col], errors="coerce").replace(0, np.nan))
    th = pd.to_numeric(df["theta_jn"], errors="coerce")
    mode = "deg" if np.nanmax(th) and np.nanmax(th) > 2*np.pi else "rad"
    if mode == "deg": th = np.deg2rad(th)
    df["cos_theta_jn"] = np.cos(th)
    mc_col = pick_col(df, ["chirp_mass","chirp_mass_source","mass_chirp"])
    if not mc_col: raise RuntimeError("chirp_mass column not found.")
    df["chirp_mass"] = pd.to_numeric(df[mc_col], errors="coerce")
    dens_col = pick_col(df, ["losdens_mass_density_mean","losdens_lum_density_mean","losdens_galaxies_per_seg"])
    if not dens_col: raise RuntimeError("No density/LOS column found.")
    df["DENS"] = pd.to_numeric(df[dens_col], errors="coerce")
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["SNR","log_DL","cos_theta_jn","chirp_mass","DENS"])
    return df

def adj_r2_sklearn(y, yhat, k):
    from sklearn.metrics import r2_score
    n = y.shape[0]
    r2 = r2_score(y, yhat)
    return 1 - (1-r2)*(n-1)/(n-k-1)

def delta_r2(y, Xbase, Xfull):
    if USE_SM:
        mB = sm.OLS(y, sm.add_constant(Xbase)).fit()
        mF = sm.OLS(y, sm.add_constant(Xfull)).fit()
        return float(mF.rsquared - mB.rsquared), float(mF.rsquared_adj)
    else:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        yhatB = LinearRegression().fit(Xbase, y).predict(Xbase)
        yhatF = LinearRegression().fit(Xfull, y).predict(Xfull)
        n = y.shape[0]
        kB = Xbase.shape[1]
        kF = Xfull.shape[1]
        r2B = r2_score(y, yhatB); r2F = r2_score(y, yhatF)
        r2B_adj = 1 - (1-r2B)*(n-1)/(n-kB-1)
        r2F_adj = 1 - (1-r2F)*(n-1)/(n-kF-1)
        return float(r2F_adj - r2B_adj), float(r2F_adj)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="results_boot")
    ap.add_argument("--nboot", type=int, default=500)
    ap.add_argument("--nperm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False)
    df = prepare(df)

    y = df["SNR"].astype(float).reset_index(drop=True)
    Xb = df[["log_DL","cos_theta_jn","chirp_mass"]].astype(float).reset_index(drop=True)
    Xf = pd.concat([Xb, df[["DENS"]].astype(float).reset_index(drop=True)], axis=1)

    # Observed ΔR²
    dR2_obs, R2F_obs = delta_r2(y, Xb, Xf)

    # Bootstrap ΔR²
    rng = np.random.default_rng(args.seed)
    boots = []
    n = len(y)
    for _ in range(args.nboot):
        idx = rng.integers(0, n, n)
        dR2, _ = delta_r2(y.iloc[idx], Xb.iloc[idx], Xf.iloc[idx])
        boots.append(dR2)
    pd.DataFrame({"deltaR2": boots}).to_csv(os.path.join(args.outdir,"bootstrap_deltaR2.csv"), index=False)

    # Permutation test
    geq = 0
    for _ in range(args.nperm):
        idx = rng.permutation(n)
        dR2p, _ = delta_r2(y, Xb, pd.concat([Xb, df[["DENS"]].iloc[idx].astype(float).reset_index(drop=True)], axis=1))
        if dR2p >= dR2_obs - 1e-12:
            geq += 1
    p_perm = (geq + 1) / (args.nperm + 1)

    # Histograma
    plt.figure(figsize=(6,3.5))
    plt.hist(boots, bins=40, alpha=0.75, color="#1f77b4")
    plt.axvline(dR2_obs, color="red", lw=2, label=f"Observed ΔR² = {dR2_obs:.4f}")
    plt.xlabel("ΔR² (full – base)")
    plt.ylabel("Count")
    plt.title("Bootstrap distribution of ΔR²")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir,"hist_deltaR2.png"), dpi=160)
    plt.close()

    # resumo
    with open(os.path.join(args.outdir,"perm_test.txt"),"w") as f:
        f.write(f"Observed ΔR² (add DENS): {dR2_obs:.6f}\n")
        f.write(f"Permutation p-value: {p_perm:.6f}  (nperm={args.nperm})\n")

    with open(os.path.join(args.outdir,"summary.txt"),"w") as f:
        f.write(f"Samples: {n}\nObserved ΔR²: {dR2_obs:.6f}\nPermutation p≈{p_perm:.6f}\n")

if __name__ == "__main__":
    main()