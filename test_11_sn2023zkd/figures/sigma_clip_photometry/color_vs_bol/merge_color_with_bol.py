#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge color (g-r) with bolometric curve by nearest time matching.

Inputs:
  --colors : CSV from your color script (needs time & color column)
  --bol    : CSV from bolometric script (needs time & Lbol or Fbol)
  --outdir : output directory for merged CSV and plot (or use --out and --plot to override)
Options:
  --tol_days / --time_tol : tolerance (days) for nearest-time merge (default 1.5)
  --smooth : rolling window (pts) to smooth color (0 disables)
  --out    : optional path for merged CSV (otherwise saved to outdir)
  --plot   : optional path for PNG figure (otherwise saved to outdir)

The script is tolerant to column names:
  Colors CSV: time -> MJD/mjd/JD/jd/Date/date ; color -> color_gr/g-r/color
  Bol CSV   : time -> MJD/mjd/JD/jd/Date/date ; value -> Lbol_erg_s or Fbol_erg_s_cm2
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def pick_col(df, candidates, required=True):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    if required:
        raise SystemExit(f"Missing any of required columns: {candidates}")
    return None

def nearest_merge(colors, bol, tol_days=1.5):
    """
    Nearest-time merge: for each color epoch find nearest bol epoch within tol_days.
    Returns merged frame with columns:
      MJD_color, color_gr, color_err (if present), MJD_bol, BOL
    """
    c = colors.copy()
    b = bol.copy()

    c = c.sort_values("MJD").reset_index(drop=True)
    b = b.sort_values("MJD").reset_index(drop=True)

    # Cartesian + nearest (robust for moderate table sizes)
    c["__k"] = 1; b["__k"] = 1
    m = c.merge(b, on="__k").drop(columns="__k")
    m["dt"] = (m["MJD_x"] - m["MJD_y"]).abs()
    m = m[m["dt"] <= float(tol_days)]
    if m.empty:
        return pd.DataFrame()

    # pick nearest bol for each color epoch
    m = (m.sort_values(["MJD_x","dt"])
           .drop_duplicates(subset=["MJD_x"], keep="first")
           .rename(columns={"MJD_x":"MJD_color","MJD_y":"MJD_bol"}))

    # keep tidy columns
    keep_cols = ["MJD_color","color_gr","MJD_bol","BOL"]
    if "color_err" in m.columns:
        keep_cols.insert(2, "color_err")
    return m[keep_cols].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--colors", required=True, help="colors CSV (from g-r script)")
    ap.add_argument("--bol",    required=True, help="bolometric CSV")
    ap.add_argument("--outdir", required=True, help="output directory (used if --out/--plot not given)")
    ap.add_argument("--tol_days", type=float, default=None, help="tolerance (days) for time matching")
    ap.add_argument("--time_tol", type=float, default=None, help="alias for --tol_days")
    ap.add_argument("--smooth", type=int, default=0, help="rolling window for smoothing color; 0 disables")
    ap.add_argument("--out",  default=None, help="optional full path for merged CSV")
    ap.add_argument("--plot", default=None, help="optional full path for PNG figure")
    args = ap.parse_args()

    # resolve tolerance
    tol = (args.time_tol if args.time_tol is not None else
           (args.tol_days if args.tol_days is not None else 1.5))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # load colors
    cdf = pd.read_csv(args.colors)
    time_c = pick_col(cdf, ["MJD","mjd","JD","jd","Date","date"])
    col_c  = pick_col(cdf, ["color_gr","g-r","color"])
    err_c  = pick_col(cdf, ["color_err","err_color","g-r_err","color error"], required=False)
    cdf = cdf.rename(columns={time_c:"MJD", col_c:"color_gr"})
    if err_c: cdf = cdf.rename(columns={err_c:"color_err"})

    # optional smoothing on color (by points, not by days)
    cdf = cdf.sort_values("MJD").reset_index(drop=True)
    if args.smooth and args.smooth > 1:
        cdf["color_gr"] = (cdf["color_gr"]
                           .rolling(args.smooth, center=True, min_periods=max(1,args.smooth//2))
                           .median())

    # load bol
    bdf = pd.read_csv(args.bol)
    time_b = pick_col(bdf, ["MJD","mjd","JD","jd","Date","date"])
    # choose Lbol if present, else Fbol
    val_b  = None
    for cand in ["Lbol_erg_s", "Fbol_erg_s_cm2", "Lbol", "Fbol"]:
        if cand in bdf.columns:
            val_b = cand
            break
    if val_b is None:
        raise SystemExit("Bolometric CSV must contain one of: Lbol_erg_s, Fbol_erg_s_cm2, Lbol, Fbol")

    bdf = bdf.rename(columns={time_b:"MJD", val_b:"BOL"})

    # merge
    merged = nearest_merge(cdf[["MJD","color_gr"] + (["color_err"] if "color_err" in cdf.columns else [])],
                           bdf[["MJD","BOL"]],
                           tol_days=tol)
    if merged.empty:
        raise SystemExit(f"No matches within tolerance ({tol} d). Try increasing --tol_days/--time_tol.")

    # save merged CSV
    out_csv = Path(args.out) if args.out else (outdir / "merged_colors_bol.csv")
    merged.to_csv(out_csv, index=False)

    # plot
    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(9,7), sharex=False)

    # Top: bolometric
    ax1.plot(bdf["MJD"], bdf["BOL"], "o-", ms=3, alpha=0.85)
    ax1.set_xlabel("MJD")
    ax1.set_ylabel("Lbol [erg s$^{-1}$]" if "Lbol" in (val_b or "") or "Lbol_erg_s" in (val_b or "") else "Fbol [erg s$^{-1}$ cm$^{-2}$]")
    ax1.grid(alpha=0.3)
    ax1.set_title("Bolometric curve")

    # Bottom: color (invert y)
    ax2.errorbar(merged["MJD_color"], merged["color_gr"],
                 yerr=merged["color_err"] if "color_err" in merged.columns and merged["color_err"].notna().any() else None,
                 fmt="o", ms=3, alpha=0.9)
    ax2.set_xlabel("MJD")
    ax2.set_ylabel("g − r  [mag]")
    ax2.invert_yaxis()
    ax2.grid(alpha=0.3)
    ax2.set_title("Color evolution (paired to nearest bolometric epochs)")

    fig.tight_layout()
    out_png = Path(args.plot) if args.plot else (outdir / "plot_color_vs_bol.png")
    plt.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"[ok] merged CSV: {out_csv}")
    print(f"[ok] plot:       {out_png}")
    print(f"[summary] matches: {len(merged)} | tolerance: {tol} d | smooth window: {args.smooth}")

if __name__ == "__main__":
    main()