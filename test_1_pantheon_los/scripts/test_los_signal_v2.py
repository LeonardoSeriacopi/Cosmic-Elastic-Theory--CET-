#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LOS vs. Hubble residual test using standard LCDM distance.
d_L_LCDM computed via trapezoidal integration of 1/E(z).

Inputs:
  - Pantheon CSV (must contain: ID, z, mu_obs)
  - los_by_sn.csv (must contain: SN_ID and LOS density column)

Outputs (in --outdir):
  - merged_for_tests.csv       (merged table with dmu)
  - los_signal_summary.csv     (summary statistics)
  - los_binned_dmu.csv         (binned results)
  - dmu_vs_<los_col>.png       (scatter + regression line)
"""

import os, argparse
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, linregress
from numpy import trapezoid as trapz
import matplotlib.pyplot as plt

C_KMS = 299792.458

def Ez_LCDM(z, Om, Ol):
    Or = 0.0
    Ok = 1.0 - Om - Ol - Or
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + Ok*(1+z)**2 + Ol)

def mu_th_lcdm(z, H0, Om, Ol):
    if z <= 0: 
        z = 1e-6
    zs = np.linspace(0, z, 2048)
    Ez = Ez_LCDM(zs, Om, Ol)
    dc = (C_KMS/H0)*trapz(1.0/Ez, zs)  # Mpc
    dm = (1+z)*dc
    return 5*np.log10(dm*1e6) - 5  # mag

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pantheon", required=True)
    ap.add_argument("--loscsv",   required=True)
    ap.add_argument("--outdir",   required=True)
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--Om", type=float, default=0.3)
    ap.add_argument("--Ol", type=float, default=0.7)
    ap.add_argument("--id_col", default="CID")
    ap.add_argument("--z_col",  default="zCMB")
    ap.add_argument("--mu_col", default="MU_SH0ES")
    ap.add_argument("--los_col", default=None,
                    help="Column name in los_by_sn.csv (e.g., dens_LOS_Kw or dens_LOS).")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    pan = pd.read_csv(args.pantheon)
    los = pd.read_csv(args.loscsv)

    # map ID/z/mu in Pantheon
    for c in [args.id_col, args.z_col, args.mu_col]:
        if c not in pan.columns:
            raise ValueError(f"Column '{c}' not found in Pantheon file.")
    pan_sub = pan[[args.id_col, args.z_col, args.mu_col]].copy().rename(
        columns={args.id_col:"SN_ID", args.z_col:"z", args.mu_col:"mu_obs"}
    )

    # choose LOS column
    los_candidates = []
    if args.los_col:
        los_candidates = [args.los_col]
    else:
        los_candidates = ["dens_LOS_Kw", "dens_LOS", "LOS_score"]

    los_col = None
    for c in los_candidates:
        if c in los.columns:
            los_col = c
            break
    if los_col is None:
        raise ValueError(f"None of these columns found in {args.loscsv}: {los_candidates}")

    # merge
    if "SN_ID" not in los.columns:
        raise ValueError("SN_ID not found in los_by_sn.csv.")
    keep_cols = ["SN_ID", los_col] + [c for c in ["n_pairs"] if c in los.columns]
    m = pd.merge(pan_sub, los[keep_cols], on="SN_ID", how="inner")

    # compute μ_LCDM and dmu
    m["mu_th"] = m["z"].apply(lambda z: mu_th_lcdm(float(z), args.H0, args.Om, args.Ol))
    m["dmu"]   = m["mu_obs"] - m["mu_th"]

    # clean and check N
    m = m.replace([np.inf, -np.inf], np.nan).dropna(subset=["z","dmu",los_col])
    N = len(m)
    if N < 20:
        print(f"[warn] Very low N after merge: {N}")

    # Spearman and regression
    rho, pval = spearmanr(m[los_col], m["dmu"])
    lr = linregress(m[los_col].values, m["dmu"].values)

    # save summary
    summary = {
        "N": N,
        "los_col": los_col,
        "spearman_rho": rho,
        "spearman_p": pval,
        "lin_slope": lr.slope,
        "lin_intercept": lr.intercept,
        "lin_r": lr.rvalue,
        "lin_p": lr.pvalue,
        "lin_stderr": lr.stderr
    }
    pd.DataFrame([summary]).to_csv(os.path.join(args.outdir, "los_signal_summary.csv"), index=False)

    # binned by terciles
    try:
        terc = pd.qcut(m[los_col], q=3, labels=["low","mid","high"])
    except Exception:
        terc = pd.cut(m[los_col], bins=3, labels=["low","mid","high"])
    mb = m.copy()
    mb["LOS_bin"] = terc
    bin_stats = (mb.groupby("LOS_bin")
                   .agg(N=("dmu","size"),
                        dmu_mean=("dmu","mean"),
                        dmu_std=("dmu","std"),
                        z_mean=("z","mean"))
                   .reset_index())
    bin_stats["dmu_sem"] = bin_stats["dmu_std"]/np.sqrt(bin_stats["N"])
    bin_stats.to_csv(os.path.join(args.outdir, "los_binned_dmu.csv"), index=False)

    # figure
    plt.figure(figsize=(6,4))
    plt.scatter(m[los_col], m["dmu"], s=10, alpha=0.5)
    xs = np.linspace(m[los_col].min(), m[los_col].max(), 200)
    plt.plot(xs, lr.slope*xs + lr.intercept)
    plt.xlabel(los_col)
    plt.ylabel("dmu (mag)")
    plt.title(f"dmu vs {los_col} | Spearman ρ={rho:.2f} (p={pval:.2g})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, f"dmu_vs_{los_col}.png"), dpi=160)

    m.to_csv(os.path.join(args.outdir, "merged_for_tests.csv"), index=False)
    print(f"[ok] N={N} | Spearman ρ={rho:.3f} (p={pval:.2g}) | slope={lr.slope:.4g}")
    print("[ok] Outputs:", args.outdir)

if __name__ == "__main__":
    main()