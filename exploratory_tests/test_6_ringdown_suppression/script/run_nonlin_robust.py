# run_nonlin_robust.py
# Uso:
#   python run_nonlin_robust.py --csv posterior_with_density.csv --outdir results_nonlin
import os, argparse, json, numpy as np, pandas as pd
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
    # SNR
    snr_col = pick_col(df, ["snr_net","SNR","network_matched_filter_snr","network_optimal_snr"])
    if not snr_col: raise RuntimeError("SNR column not found.")
    df = df.copy()
    df["SNR"] = pd.to_numeric(df[snr_col], errors="coerce")
    # distance
    dl_col = pick_col(df, ["luminosity_distance","DL_p50","comoving_distance"])
    if not dl_col: raise RuntimeError("Distance column not found.")
    df["log_DL"] = np.log10(pd.to_numeric(df[dl_col], errors="coerce").replace(0, np.nan))
    # orientation
    th_col = pick_col(df, ["theta_jn"])
    if not th_col: raise RuntimeError("theta_jn column not found.")
    th = pd.to_numeric(df[th_col], errors="coerce")
    # detectar deg vs rad
    mode = "deg" if np.nanmax(th) and np.nanmax(th) > 2*np.pi else "rad"
    if mode == "deg": th = np.deg2rad(th)
    df["cos_theta_jn"] = np.cos(th)
    # mass
    mc_col = pick_col(df, ["chirp_mass","chirp_mass_source","mass_chirp"])
    if not mc_col: raise RuntimeError("chirp_mass column not found.")
    df["chirp_mass"] = pd.to_numeric(df[mc_col], errors="coerce")
    # redshift
    z_col = pick_col(df, ["redshift","z_p50"])
    df["redshift"] = pd.to_numeric(df[z_col], errors="coerce") if z_col else np.nan
    # densities
    dens_col = pick_col(df, ["losdens_mass_density_mean","losdens_lum_density_mean","losdens_galaxies_per_seg"])
    df["density_feature"] = pd.to_numeric(df[dens_col], errors="coerce") if dens_col else np.nan
    # dropna core
    df = df.replace([np.inf,-np.inf], np.nan).dropna(subset=["SNR","log_DL","cos_theta_jn","chirp_mass"])
    return df

def fit_base_and_nonlin(y, X):
    # base: y ~ const + X
    if USE_SM:
        m_base = sm.OLS(y, sm.add_constant(X)).fit()
        yhat = m_base.predict(sm.add_constant(X))
        Xnl = pd.DataFrame({"base_sq": yhat**2}, index=X.index)
        m_nonlin = sm.OLS(y, sm.add_constant(pd.concat([X, Xnl], axis=1))).fit()
        return dict(
            r2_adj_base=float(m_base.rsquared_adj),
            r2_adj_nonlin=float(m_nonlin.rsquared_adj),
            delta_r2=float(m_nonlin.rsquared_adj - m_base.rsquared_adj),
            beta_nl=float(m_nonlin.params.get("base_sq", np.nan)),
            p_nl=float(m_nonlin.pvalues.get("base_sq", np.nan))
        )
    else:
        lrB = LinearRegression().fit(X, y)
        yhat = lrB.predict(X)
        r2B = r2_score(y, yhat)
        n, k = X.shape
        r2B_adj = 1 - (1-r2B)*(n-1)/(n-k-1)
        X2 = pd.concat([X, pd.Series(yhat**2, name="base_sq", index=X.index)], axis=1)
        lrN = LinearRegression().fit(X2, y)
        yhatN = lrN.predict(X2)
        k2 = X2.shape[1]
        r2N = r2_score(y, yhatN)
        r2N_adj = 1 - (1-r2N)*(n-1)/(n-k2-1)
        beta_nl = float(lrN.coef_[-1])
        return dict(
            r2_adj_base=float(r2B_adj),
            r2_adj_nonlin=float(r2N_adj),
            delta_r2=float(r2N_adj - r2B_adj),
            beta_nl=beta_nl,
            p_nl=np.nan
        )

def terciles(series):
    qs = series.quantile([0.3333, 0.6667]).values
    bins = [-np.inf, qs[0], qs[1], np.inf]
    labels = ["low","mid","high"]
    return pd.cut(series, bins=bins, labels=labels)

def run_split(df, split_name, key_series):
    res = []
    groups = terciles(key_series.dropna())
    df2 = df.loc[groups.index].copy()
    df2["_grp"] = groups
    for g, d in df2.groupby("_grp"):
        if d.shape[0] < 500:  # mínimo pra estabilidade
            res.append(dict(split=split_name, group=str(g), n=int(d.shape[0]), r2_adj_base=np.nan, r2_adj_nonlin=np.nan, delta_r2=np.nan, beta_nl=np.nan, p_nl=np.nan))
            continue
        X = d[["log_DL","cos_theta_jn","chirp_mass"]].astype(float)
        y = d["SNR"].astype(float)
        r = fit_base_and_nonlin(y, X)
        r.update(dict(split=split_name, group=str(g), n=int(d.shape[0])))
        res.append(r)
    return res

def barplot(df, out_png, title):
    plt.figure(figsize=(6,3.5))
    for split, d in df.groupby("split"):
        xs = np.arange(d.shape[0])
        plt.bar(xs + (0.0 if split.endswith("chirp") else 0.28 if split.endswith("redshift") else 0.56),
                d["delta_r2"], width=0.28, label=split)
    plt.axhline(0, color="k", lw=0.8)
    plt.xticks([0.28, 1.28, 2.28], ["low","mid","high"])
    plt.ylabel("ΔR² (nonlinear – base)")
    plt.title(title)
    plt.legend(frameon=False, fontsize=8, ncol=1)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", default="results_nonlin")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv, low_memory=False)
    df = prepare(df)

    results = []
    # splits
    results += run_split(df, "split_chirp", df["chirp_mass"])
    if df["redshift"].notna().any():
        results += run_split(df, "split_redshift", df["redshift"])
    if df["density_feature"].notna().any():
        results += run_split(df, "split_density", df["density_feature"])

    outcsv = os.path.join(args.outdir, "robust_summary.csv")
    pd.DataFrame(results).to_csv(outcsv, index=False)

    # gráficos por split (se existirem)
    dfres = pd.DataFrame(results).dropna(subset=["delta_r2"])
    if (dfres["split"]=="split_chirp").any():
        barplot(dfres[dfres["split"]=="split_chirp"], os.path.join(args.outdir,"deltaR2_by_split_chirp.png"),
                "Nonlinear gain by chirp-mass tertiles")
    if (dfres["split"]=="split_redshift").any():
        barplot(dfres[dfres["split"]=="split_redshift"], os.path.join(args.outdir,"deltaR2_by_split_redshift.png"),
                "Nonlinear gain by redshift tertiles")
    if (dfres["split"]=="split_density").any():
        barplot(dfres[dfres["split"]=="split_density"], os.path.join(args.outdir,"deltaR2_by_split_density.png"),
                "Nonlinear gain by density tertiles")

    with open(os.path.join(args.outdir,"README.txt"),"w",encoding="utf-8") as f:
        f.write("Nonlinear robustness finished.\nSaved: robust_summary.csv and PNG plots.\n")

if __name__ == "__main__":
    main()