# run_cap_model.py
# Uso:
#   python run_cap_model.py --csv posterior_with_density.csv --outdir results_cap --grid 25
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
    densX = pick_col(df, ["losdens_mass_density_mean","losdens_galaxies_per_seg"])
    df["Xcap"] = pd.to_numeric(df[densX], errors="coerce") if densX else np.nan
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["SNR","log_DL","cos_theta_jn","chirp_mass"])
    return df

def adj_r2(y, yhat, k):
    n = y.shape[0]
    ssr = np.sum((y - yhat)**2)
    sst = np.sum((y - np.mean(y))**2)
    r2 = 1 - ssr/sst if sst>0 else 0.0
    return 1 - (1-r2)*(n-1)/(n-k-1)

def aic_bic(y, yhat, k):
    n = y.shape[0]
    resid = y - yhat
    sigma2 = np.mean(resid**2)
    # Gaussian loglik ~ -n/2 [log(2πσ2)+1]
    ll = -0.5*n*(np.log(2*np.pi*sigma2)+1) if sigma2>0 else -1e9
    AIC = 2*k - 2*ll
    BIC = np.log(n)*k - 2*ll
    return AIC, BIC

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="results_cap")
    ap.add_argument("--grid", type=int, default=25, help="grid resolution for c0,c1")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    df = pd.read_csv(args.csv, low_memory=False)
    df = prepare(df)
    X = df[["log_DL","cos_theta_jn","chirp_mass"]].astype(float)
    y = df["SNR"].astype(float)

    # Base fit
    if USE_SM:
        mB = sm.OLS(y, sm.add_constant(X)).fit()
        yhatB = mB.predict(sm.add_constant(X))
        kB = int(len(mB.params))  # const + 3
        # Nonlinear (para referência)
        yhat = yhatB
        mN = sm.OLS(y, sm.add_constant(pd.concat([X, pd.Series(yhat**2, name="base_sq")], axis=1))).fit()
        yhatN = mN.predict(sm.add_constant(pd.concat([X, pd.Series(yhat**2, name="base_sq")], axis=1)))
        kN = int(len(mN.params))
    else:
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score
        lr = LinearRegression().fit(X, y)
        yhatB = lr.predict(X)
        kB = 4
        X2 = pd.concat([X, pd.Series(yhatB**2, name="base_sq")], axis=1)
        lr2 = LinearRegression().fit(X2, y)
        yhatN = lr2.predict(X2)
        kN = 5

    r2B = adj_r2(y, yhatB, kB-1)
    AICB, BICB = aic_bic(y, yhatB, kB)

    r2N = adj_r2(y, yhatN, kN-1)
    AICN, BICN = aic_bic(y, yhatN, kN)

    # CAP scan (se houver Xcap)
    out_rows = []
    if df["Xcap"].notna().any():
        Xcap = df["Xcap"].fillna(df["Xcap"].median())
        c0_grid = np.linspace(np.percentile(yhatB, 10), np.percentile(yhatB, 90), args.grid)
        c1_grid = np.linspace(-2.0, 2.0, args.grid)

        best = dict(AIC=1e99)
        for c0 in c0_grid:
            for c1 in c1_grid:
                hmax = c0 + c1*Xcap
                ycap = np.minimum(yhatB, hmax)
                r2C = adj_r2(y, ycap, 2)  # efetivos: c0,c1
                AICC, BICC = aic_bic(y, ycap, 2)
                out_rows.append(dict(c0=float(c0), c1=float(c1), R2_adj=float(r2C), AIC=float(AICC), BIC=float(BICC)))
                if AICC < best["AIC"]:
                    best = dict(c0=float(c0), c1=float(c1), R2_adj=float(r2C), AIC=float(AICC), BIC=float(BICC))

        pd.DataFrame(out_rows).to_csv(os.path.join(args.outdir,"cap_scan.csv"), index=False)
        with open(os.path.join(args.outdir,"best_cap.txt"),"w") as f:
            f.write(f"Best cap (by AIC): c0={best['c0']:.3f}, c1={best['c1']:.3f}, R2_adj={best['R2_adj']:.4f}, AIC={best['AIC']:.2f}, BIC={best['BIC']:.2f}\n")
        # plots: obs vs pred (base) e obs vs pred (cap best)
        plt.figure(figsize=(5,4))
        plt.scatter(yhatB, y, s=4, alpha=0.3)
        plt.xlabel("Predicted SNR (base)")
        plt.ylabel("Observed SNR")
        plt.title("Observed vs Predicted (base)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"obs_vs_pred_base.png"), dpi=160); plt.close()

        ycap_best = np.minimum(yhatB, best["c0"] + best["c1"]*Xcap)
        plt.figure(figsize=(5,4))
        plt.scatter(ycap_best, y, s=4, alpha=0.3, label="with cap")
        plt.xlabel("Predicted SNR (capped)")
        plt.ylabel("Observed SNR")
        plt.title("Observed vs Predicted (cap)")
        plt.tight_layout(); plt.savefig(os.path.join(args.outdir,"obs_vs_pred_cap.png"), dpi=160); plt.close()
    else:
        with open(os.path.join(args.outdir,"best_cap.txt"),"w") as f:
            f.write("No Xcap column found; skipped cap scan.\n")

    # resumo txt
    with open(os.path.join(args.outdir,"summary.txt"),"w") as f:
        f.write("=== BASE ===\n")
        f.write(f"R2_adj={r2B:.4f}  AIC={AICB:.2f}  BIC={BICB:.2f}\n")
        f.write("=== NONLINEAR ===\n")
        f.write(f"R2_adj={r2N:.4f}  AIC={AICN:.2f}  BIC={BICN:.2f}\n")
        if df["Xcap"].notna().any() and out_rows:
            bestA = min(out_rows, key=lambda r:r["AIC"])
            f.write("=== CAP (best AIC in grid) ===\n")
            f.write(f"R2_adj={bestA['R2_adj']:.4f}  AIC={bestA['AIC']:.2f}  BIC={bestA['BIC']:.2f}  (c0={bestA['c0']:.3f}, c1={bestA['c1']:.3f})\n")

if __name__ == "__main__":
    main()