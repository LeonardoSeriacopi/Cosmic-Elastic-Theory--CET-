#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise contínua de ΔD_resid e |ΔD_resid| vs métricas LOS,
com estratificação opcional por faixas de z (sem tercis/quartis).

Exemplo:
  python analyze_los_vs_residuals_stratified.py ^
    --csv ceers_los_poly_v2_norm.csv ^
    --zcol z_best_new ^
    --xcols los_mean_w,los_median_w,los_sum_norm,los_mean_norm,los_median_norm ^
    --ycols DeltaD_resid ^
    --also-abs 1 ^
    --z-bins 0,1;1,2;2,99 ^
    --logx 1 ^
    --winsor-x 1 --winsor-y 1 ^
    --smooth-pct 0.15 ^
    --plot-dir figs_strat ^
    --out-csv resumo_strat.csv
"""

import argparse
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr, theilslopes

# ---------------------- utils ----------------------

def safe_name(s: str) -> str:
    """Torna um nome seguro para Windows (remove \/:*?"<>|)."""
    return re.sub(r'[\\/:*?"<>|]+', '_', s)

def winsorize(v: np.ndarray, p: float):
    if not p or p <= 0:
        return v
    lo, hi = np.nanquantile(v, [p/100.0, 1.0 - p/100.0])
    w = v.copy()
    w[w < lo] = lo
    w[w > hi] = hi
    return w

def rolling_median_xy(x: np.ndarray, y: np.ndarray, frac: float = 0.15, min_points: int = 20):
    """
    Curva suave contínua sem bins fixos:
    - ordena por x
    - calcula mediana móvel de y com janela = max(min_points, ceil(frac * n))
    """
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < max(min_points, 5):
        return None, None
    order = np.argsort(x, kind="mergesort")
    x_sorted = x[order]
    y_sorted = y[order]
    n = x_sorted.size
    win = max(int(np.ceil(frac * n)), min_points)
    if win % 2 == 0:
        win += 1  # janela ímpar
    half = win // 2
    y_med = np.full(n, np.nan)
    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        y_med[i] = np.nanmedian(y_sorted[i0:i1])
    return x_sorted, y_med

# ------------------ análise principal ------------------

def analyze_one(df: pd.DataFrame, xcol: str, ycol: str, label: str,
                logx: bool, winsor_x: float, winsor_y: float,
                smooth_pct: float, plot_dir: str):
    if xcol not in df.columns:
        print(f"[WARN] coluna X ausente: {xcol} (pulando)")
        return None

    x = df[xcol].to_numpy(float)
    y = df[ycol].to_numpy(float)

    if logx:
        x = np.log10(np.clip(x, 1e-12, None))
        xlab = f"log10({xcol})"
    else:
        xlab = xcol

    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 5:
        print(f"[WARN] poucos pontos após limpeza em {xcol}")
        return None

    if winsor_x and winsor_x > 0:
        x = winsorize(x, winsor_x)
    if winsor_y and winsor_y > 0:
        y = winsorize(y, winsor_y)

    rs, ps = spearmanr(x, y)
    rp, pp = pearsonr(x, y)

    try:
        slope, inter, lo_s, hi_s = theilslopes(y, x)  # y = slope*x + inter
    except Exception as e:
        print(f"[WARN] theilslopes falhou em {xcol}: {e}")
        slope = inter = lo_s = hi_s = np.nan

    xs, ymed = rolling_median_xy(x, y, frac=smooth_pct, min_points=20)

    out = dict(
        xcol=xcol, ycol=label, n=int(x.size),
        spearman_r=float(rs), spearman_p=float(ps),
        pearson_r=float(rp), pearson_p=float(pp),
        theilsen_slope=float(slope), theilsen_intercept=float(inter),
        theilsen_slope_lo=float(lo_s), theilsen_slope_hi=float(hi_s)
    )

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fig = plt.figure(figsize=(6.0, 4.2), dpi=140)
        plt.scatter(x, y, s=10, alpha=0.5)
        if np.isfinite(slope) and np.isfinite(inter):
            xl = np.linspace(np.nanmin(x), np.nanmax(x), 200)
            yl = slope * xl + inter
            plt.plot(xl, yl, linewidth=2)
        if xs is not None:
            plt.plot(xs, ymed, linewidth=2)
        ttl = f"{label} vs {xlab}\nSpearman={rs:.3f} (p={ps:.2g}) | Theil–Sen slope={slope:.3g}"
        plt.title(ttl)
        plt.xlabel(xlab)
        plt.ylabel(label)
        plt.tight_layout()
        png = os.path.join(plot_dir, f"{safe_name(label)}_vs_{safe_name(xcol)}.png")
        plt.savefig(png)
        plt.close()
        out["plot"] = png

    return out

def parse_zbins(zbins_str: str):
    if not zbins_str:
        return []
    out = []
    chunks = [c for c in zbins_str.split(";") if c.strip()]
    for ch in chunks:
        a, b = ch.split(",")
        out.append((float(a), float(b)))
    return out

def main():
    ap = argparse.ArgumentParser(description="ΔD_resid e |ΔD_resid| vs LOS (contínuo, com faixas de z)")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--zcol", required=True)
    ap.add_argument("--xcols", required=True, help="ex.: los_mean_w,los_median_w,los_sum_norm")
    ap.add_argument("--ycols", default="DeltaD_resid", help="ex.: DeltaD_resid")
    ap.add_argument("--also-abs", type=int, default=1, help="1=analisar |y| também (default)")
    ap.add_argument("--z-bins", dest="z_bins", default="", help="faixas z tipo '0,1;1,2;2,99'")
    ap.add_argument("--logx", type=int, default=1)
    ap.add_argument("--winsor-x", type=float, default=0.0)
    ap.add_argument("--winsor-y", type=float, default=0.0)
    ap.add_argument("--smooth-pct", type=float, default=0.15)
    ap.add_argument("--plot-dir", default="")
    ap.add_argument("--out-csv", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    xcols = [c.strip() for c in args.xcols.split(",") if c.strip()]
    ycols = [c.strip() for c in args.ycols.split(",") if c.strip()]
    zbins = parse_zbins(args.z_bins)
    z = df[args.zcol].to_numpy(float)

    def run_block(df_block: pd.DataFrame, label_suffix: str):
        rows = []
        for ycol in ycols:
            ylab_main = ycol + label_suffix
            # bloco principal (y normal)
            for xcol in xcols:
                res = analyze_one(df_block, xcol, ycol, ylab_main,
                                  bool(args.logx), args.winsor_x, args.winsor_y,
                                  args.smooth_pct, args.plot_dir)
                if res:
                    rows.append(res)
            # bloco com |y|
            if args.also_abs:
                tmp = df_block.copy()
                y_abs_col = ylab_main + "|abs|"
                tmp[y_abs_col] = np.abs(tmp[ycol].to_numpy(float))
                for xcol in xcols:
                    res = analyze_one(tmp, xcol, y_abs_col, y_abs_col,
                                      bool(args.logx), args.winsor_x, args.winsor_y,
                                      args.smooth_pct, args.plot_dir)
                    if res:
                        rows.append(res)
        return rows

    all_rows = []
    # Global (todas as linhas)
    all_rows += run_block(df, label_suffix="")

    # Por faixas de z
    for (za, zb) in zbins:
        sel = (z >= za) & (z < zb) & np.isfinite(z)
        nsel = int(sel.sum())
        if nsel < 10:
            print(f"[WARN] pouca amostra em z∈[{za},{zb}) -> n={nsel} (pulando)")
            continue
        label = f" [z∈[{za},{zb})]"
        all_rows += run_block(df[sel].copy(), label_suffix=label)

    if args.out_csv:
        pd.DataFrame(all_rows).to_csv(args.out_csv, index=False)
        print(f"[OK] resumo salvo em: {args.out_csv}")

if __name__ == "__main__":
    main()