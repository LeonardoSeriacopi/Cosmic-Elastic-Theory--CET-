#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, rankdata

def load_df(path, z_col, dmu_col, dens_col):
    df = pd.read_csv(path)
    for c in [z_col, dmu_col, dens_col]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in {os.path.basename(path)}. Found: {list(df.columns)[:12]} ...")
    df = df[[z_col, dmu_col, dens_col]].dropna()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df = df.rename(columns={z_col:"z", dmu_col:"dmu", dens_col:"dens"})
    return df

def detrend_by_z_quantiles(x, z, qbins=10):
    # detrend by quantiles of z
    qs = np.linspace(0, 1, qbins+1)
    edges = np.quantile(z, qs)
    edges[0]  = min(edges[0], z.min()) - 1e-12
    edges[-1] = max(edges[-1], z.max()) + 1e-12
    idx = np.digitize(z, edges) - 1  # 0..qbins-1
    xm = np.zeros_like(x, dtype=float)
    for b in range(qbins):
        m = np.median(x[idx==b]) if np.any(idx==b) else 0.0
        xm[idx==b] = x[idx==b] - m
    return xm, edges

def partial_spearman(x, y, z):
    # Partial Spearman: residualize ranks of x and y against ranks of z
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    A = np.vstack([np.ones_like(rz), rz]).T
    bx = np.linalg.lstsq(A, rx, rcond=None)[0]
    by = np.linalg.lstsq(A, ry, rcond=None)[0]
    rx_res = rx - A @ bx
    ry_res = ry - A @ by
    r, p = spearmanr(rx_res, ry_res)
    return r, p

def ols_with_z(x, y, z):
    # Multiple regression: y = a + b*x + c*z
    X = np.column_stack([np.ones_like(x), x, z])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat
    n, k = X.shape
    s2 = (resid @ resid) / (n - k)
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    b_x, se_x = beta[1], se[1]
    t_x = b_x / se_x if se_x>0 else np.nan
    from math import erf, sqrt
    p_x = 2*(1-0.5*(1+erf(abs(t_x)/np.sqrt(2)))) if np.isfinite(t_x) else np.nan
    return dict(beta0=beta[0], beta_x=b_x, beta_z=beta[2], se_x=se_x, t_x=t_x, p_x=p_x)

def scatter_with_trend(x, y, z, title, xlabel, ylabel, outpng):
    plt.figure(figsize=(8,6))
    sc = plt.scatter(x, y, s=18, c=z, alpha=0.85)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    cb = plt.colorbar(sc); cb.set_label("z")
    A = np.vstack([np.ones_like(x), x]).T
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    plt.plot(xs, b[0] + b[1]*xs)
    plt.tight_layout()
    plt.savefig(outpng, dpi=130); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Correlation Δμ vs LOS density controlling for z (raw, detrended, partial).")
    ap.add_argument("--merged_lcdm", required=True, help="CSV with columns z, dmu and dens_LOS_Kw (e.g. merged_for_tests.csv)")
    ap.add_argument("--merged_cet",  required=True, help="Equivalent CSV for CET (e.g. merged_for_tests_CET.csv)")
    ap.add_argument("--z_col", default="zSN", help="name of the redshift column")
    ap.add_argument("--dens_col", default="dens_LOS_Kw", help="LOS density column (K-weighted)")
    ap.add_argument("--dmu_col_lcdm", default="dmu", help="residual column (LCDM)")
    ap.add_argument("--dmu_col_cet",  default="dmu_CET", help="residual column (CET)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--qbins", type=int, default=10, help="number of z quantiles for detrending")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dfL = load_df(args.merged_lcdm, args.z_col, args.dmu_col_lcdm, args.dens_col)
    dfC = load_df(args.merged_cet,  args.z_col, args.dmu_col_cet,  args.dens_col)

    out_rows = []

    for tag, df in [("LCDM", dfL), ("CET", dfC)]:
        z   = df["z"].values.astype(float)
        dst = df["dens"].values.astype(float)
        dmu = df["dmu"].values.astype(float)

        r_raw, p_raw = spearmanr(dst, dmu)
        dst_dt, edges = detrend_by_z_quantiles(dst, z, qbins=args.qbins)
        r_dt, p_dt = spearmanr(dst_dt, dmu)
        r_par, p_par = partial_spearman(dst, dmu, z)
        reg = ols_with_z(dst, dmu, z)

        scatter_with_trend(dst,    dmu, z, f"{tag}: Δμ vs dens_LOS_Kw (raw)",
                           "dens_LOS_Kw", "Δμ (mag)",
                           os.path.join(args.outdir, f"{tag}_scatter_raw.png"))
        scatter_with_trend(dst_dt, dmu, z, f"{tag}: Δμ vs dens_LOS_Kw (detrended by z)",
                           "dens_LOS_Kw (z-detrended)", "Δμ (mag)",
                           os.path.join(args.outdir, f"{tag}_scatter_detrended.png"))

        out_rows.append({
            "model": tag,
            "N": len(df),
            "spearman_raw_r": r_raw, "spearman_raw_p": p_raw,
            "spearman_detrended_r": r_dt, "spearman_detrended_p": p_dt,
            "spearman_partial_r": r_par, "spearman_partial_p": p_par,
            "reg_beta_x": reg["beta_x"], "reg_se_x": reg["se_x"],
            "reg_t_x": reg["t_x"], "reg_p_x": reg["p_x"],
            "reg_beta_z": reg["beta_z"]
        })

    summary = pd.DataFrame(out_rows)
    out_csv = os.path.join(args.outdir, "los_zcorrect_summary.csv")
    summary.to_csv(out_csv, index=False)
    print(summary.to_string(index=False))
    print(f"\n[ok] Written: {out_csv}")
    print(f"Plots: *_scatter_raw.png and *_scatter_detrended.png in {args.outdir}")
    
if __name__ == "__main__":
    main()