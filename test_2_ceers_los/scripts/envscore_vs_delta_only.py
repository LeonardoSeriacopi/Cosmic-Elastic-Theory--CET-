#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de controle: ambiente bruto (env_score_0_100) vs resíduo (DeltaD_resid).

- Se --resid-col existir no CSV, usa diretamente essa coluna.
- Caso contrário, tenta calcular: DL_obs_Mpc - DL_lcdm_Mpc.
- Salva tabelas (Spearman global, por tercis/quartis, e por z) e 1 figura.
"""

import argparse, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, theilslopes

def winsorize(v: np.ndarray, p: float):
    if not p or p <= 0: return v
    lo, hi = np.nanquantile(v, [p/100.0, 1.0 - p/100.0])
    w = v.copy()
    w[w < lo] = lo
    w[w > hi] = hi
    return w

def rolling_median_xy(x, y, frac=0.15, min_points=20):
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < max(min_points,5): return None, None
    ord_ = np.argsort(x, kind="mergesort")
    x = x[ord_]; y = y[ord_]
    n = x.size
    win = max(int(np.ceil(frac*n)), min_points)
    if win % 2 == 0: win += 1
    half = win // 2
    ymed = np.full(n, np.nan)
    for i in range(n):
        i0 = max(0, i-half); i1 = min(n, i+half+1)
        ymed[i] = np.nanmedian(y[i0:i1])
    return x, ymed

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--env-col", default="env_score_0_100")
    ap.add_argument("--resid-col", default="DeltaD_resid",
                    help="se existir no CSV, será usada diretamente")
    ap.add_argument("--z-col", default="z_best_new")
    ap.add_argument("--winsor-x", type=float, default=0.0)
    ap.add_argument("--winsor-y", type=float, default=1.0)
    ap.add_argument("--out-prefix", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.env_col not in df.columns:
        raise SystemExit(f"[ERR] coluna de ambiente não encontrada: {args.env_col}")

    # --- pegar resíduo ---
    if args.resid_col in df.columns:
        resid = pd.to_numeric(df[args.resid_col], errors="coerce").to_numpy()
    else:
        # fallback: tentar DL_obs_Mpc - DL_lcdm_Mpc
        if "DL_obs_Mpc" in df.columns and "DL_lcdm_Mpc" in df.columns:
            resid = pd.to_numeric(df["DL_obs_Mpc"], errors="coerce").to_numpy() \
                    - pd.to_numeric(df["DL_lcdm_Mpc"], errors="coerce").to_numpy()
        else:
            raise SystemExit("[ERR] não há --resid-col nem colunas DL_obs_Mpc/DL_lcdm_Mpc para calcular o resíduo.")

    env = pd.to_numeric(df[args.env_col], errors="coerce").to_numpy()
    z = pd.to_numeric(df[args.z_col], errors="coerce").to_numpy() if args.z_col in df.columns else np.full_like(env, np.nan)

    # limpeza + winsorizações
    m = np.isfinite(env) & np.isfinite(resid)
    env = env[m]; resid = resid[m]; z = z[m]
    if env.size < 10: raise SystemExit("[ERR] amostra muito pequena após limpeza.")

    if args.winsor_x > 0: env_w = winsorize(env, args.winsor_x)
    else: env_w = env.copy()
    if args.winsor_y > 0: resid_w = winsorize(resid, args.winsor_y)
    else: resid_w = resid.copy()

    # --- correlações globais ---
    rs, ps = spearmanr(env_w, resid_w)
    rp, pp = pearsonr(env_w, resid_w)
    pd.DataFrame([dict(spearman_r=rs, spearman_p=ps, pearson_r=rp, pearson_p=pp, n=int(env_w.size))]) \
      .to_csv(f"{args.out_prefix}_spearman_global.csv", index=False)

    # --- bins por tercis/quartis ---
    out_rows_t = []
    terc_edges = np.nanquantile(env, [0, 1/3, 2/3, 1])
    t_idx = np.digitize(env, terc_edges, right=True) - 1
    for k, label in enumerate(["T1","T2","T3"]):
        sel = (t_idx == k)
        if sel.sum() == 0: continue
        yk = resid[sel]
        out_rows_t.append(dict(bin=label, n=int(sel.sum()),
                               resid_mean=float(np.nanmean(yk)),
                               resid_median=float(np.nanmedian(yk)),
                               resid_std=float(np.nanstd(yk))))
    pd.DataFrame(out_rows_t).to_csv(f"{args.out_prefix}_resid_by_env_terciles.csv", index=False)

    out_rows_q = []
    quart_edges = np.nanquantile(env, [0, .25, .5, .75, 1])
    q_idx = np.digitize(env, quart_edges, right=True) - 1
    for k, label in enumerate(["Q1","Q2","Q3","Q4"]):
        sel = (q_idx == k)
        if sel.sum() == 0: continue
        yk = resid[sel]
        out_rows_q.append(dict(bin=label, n=int(sel.sum()),
                               resid_mean=float(np.nanmean(yk)),
                               resid_median=float(np.nanmedian(yk)),
                               resid_std=float(np.nanstd(yk))))
    pd.DataFrame(out_rows_q).to_csv(f"{args.out_prefix}_resid_by_env_quartiles.csv", index=False)

    # --- por z (baixo/alto) se disponível ---
    if np.isfinite(z).any():
        zmed = np.nanmedian(z)
        rows = []
        for name, sel in [("low-z", z <= zmed), ("high-z", z > zmed)]:
            e = env_w[sel & np.isfinite(z)]
            r = resid_w[sel & np.isfinite(z)]
            if e.size >= 8:
                rsz, psz = spearmanr(e, r)
                rows.append(dict(z_bin=name, n=int(e.size),
                                 spearman_r=float(rsz), spearman_p=float(psz)))
        if rows:
            pd.DataFrame(rows).to_csv(f"{args.out_prefix}_spearman_by_z.csv", index=False)

    # --- figura única ---
    xs, ymed = rolling_median_xy(env_w, resid_w, frac=0.2, min_points=20)
    try:
        slope, inter, lo_s, hi_s = theilslopes(resid_w, env_w)
    except Exception:
        slope = inter = np.nan, np.nan

    plt.figure(figsize=(5.6,4.2), dpi=140)
    plt.scatter(env_w, resid_w, s=14, alpha=0.55)
    if np.isfinite(slope).all():
        xl = np.linspace(np.nanmin(env_w), np.nanmax(env_w), 300)
        yl = slope*xl + inter
        plt.plot(xl, yl, linewidth=2)
    if xs is not None:
        plt.plot(xs, ymed, linewidth=2)
    plt.xlabel(args.env_col)
    plt.ylabel("DeltaD_resid")
    plt.title(f"env vs resíduo  | Spearman={rs:.3f} (p={ps:.2g})")
    plt.tight_layout()
    outpng = f"{args.out_prefix}_env_vs_resid.png"
    plt.savefig(outpng); plt.close()
    print(f"[OK] global: Spearman={rs:.3f} (p={ps:.2g}) | Pearson={rp:.3f} (p={pp:.2g})")
    print(f"[OK] tabelas: {args.out_prefix}_spearman_global.csv, _resid_by_env_*.csv")
    print(f"[OK] figura: {outpng}")

if __name__ == "__main__":
    main()