#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute ZTF g-r color from binned/clipped photometry by pairing g and r points
within a time tolerance (MJD). Robust to column-name variants.

Input  CSV needs columns like:
- MJD or mjd
- band  (expects 'ztf_g' and/or 'ztf_r'; other bands are ignored)
- mag or mag_corr (will prefer mag_corr if present)
- err or mag_err or e_mag (optional; used to propagate color uncertainties)

Usage example (Windows):
  python phot_colors_gr_pair.py ^
    --inp "C:\...\saida_gagliano_clip_5\phot_all_clean_bin3d_clip_A.csv" ^
    --out "C:\...\saida_gagliano_clip_5\colors_gr.csv" ^
    --time_tol 0.5
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def pick_col(df, candidates, required=False):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Required column not found. Tried: {candidates}")
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="Input CSV (binned/clipped photometry)")
    ap.add_argument("--out", required=True, help="Output CSV for g-r colors")
    ap.add_argument("--time_tol", type=float, default=0.5,
                    help="Pairing tolerance in days for MJD (default 0.5)")
    ap.add_argument("--plot", default=None,
                    help="Optional PNG path to save color vs. MJD plot")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)

    # Normalize essential columns
    mjd_col  = pick_col(df, ["MJD", "mjd"], required=True)
    band_col = pick_col(df, ["band", "filter", "Band"], required=True)
    mag_col  = pick_col(df, ["mag_corr", "mag_cal", "mag"], required=True)
    err_col  = pick_col(df, ["mag_err", "err", "e_mag", "magerr"], required=False)

    # Keep only ZTF g/r
    df = df.copy()
    df[band_col] = df[band_col].astype(str).str.strip().str.lower()
    df = df[df[band_col].isin(["ztf_g", "ztf_r"])].reset_index(drop=True)

    if df.empty:
        raise SystemExit("No ZTF g/r points found in the input CSV.")

    # Build g/r tables
    cols_keep = [mjd_col, mag_col]
    if err_col: cols_keep.append(err_col)

    dfg = df[df[band_col] == "ztf_g"][cols_keep].rename(
        columns={mjd_col: "mjd", mag_col: "mag_g", (err_col or "none"): "err_g"})
    dfr = df[df[band_col] == "ztf_r"][cols_keep].rename(
        columns={mjd_col: "mjd", mag_col: "mag_r", (err_col or "none"): "err_r"})

    # merge_asof requires sorted by key
    dfg = dfg.sort_values("mjd").reset_index(drop=True)
    dfr = dfr.sort_values("mjd").reset_index(drop=True)

    # As-of merge: for each g find nearest r within tolerance
    merged = pd.merge_asof(
        dfg, dfr, on="mjd", direction="nearest",
        tolerance=pd.Timedelta(args.time_tol, unit="D") if False else args.time_tol
    )
    # pandas merge_asof works with numeric directly (not Timedelta) when key is float.
    # Keep only rows that found a partner
    merged = merged.dropna(subset=["mag_r"]).copy()

    if merged.empty:
        raise SystemExit("No g–r pairs within the specified time tolerance.")

    # Compute color and uncertainty
    merged["color_gr"] = merged["mag_g"] - merged["mag_r"]
    if "err_g" in merged.columns and "err_r" in merged.columns:
        if merged["err_g"].notna().any() and merged["err_r"].notna().any():
            merged["color_err"] = np.sqrt(merged["err_g"].fillna(0.0)**2 +
                                          merged["err_r"].fillna(0.0)**2)
        else:
            merged["color_err"] = np.nan
    else:
        merged["color_err"] = np.nan

    # Reorder columns nicely
    outcols = ["mjd", "mag_g", "mag_r", "color_gr", "color_err"]
    for c in ["err_g", "err_r"]:
        if c in merged.columns:
            outcols.insert(3, c)  # place errors before color
    merged = merged[outcols].copy()

    # Save CSV
    outpath = Path(args.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(outpath, index=False)
    print(f"Saved {len(merged)} g–r pairs -> {outpath}")

    # Optional plot
    if args.plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                       gridspec_kw={"height_ratios":[2,1]})
        # lower mag = brighter; just show as-is
        ax1.scatter(merged["mjd"], merged["mag_g"], s=16, label="g")
        ax1.scatter(merged["mjd"], merged["mag_r"], s=16, label="r")
        ax1.invert_yaxis()
        ax1.set_ylabel("Magnitude")
        ax1.legend()

        ax2.errorbar(merged["mjd"], merged["color_gr"],
                     yerr=merged["color_err"] if merged["color_err"].notna().any() else None,
                     fmt="o", ms=3)
        ax2.axhline(0, lw=1, alpha=0.3)
        ax2.set_xlabel("MJD")
        ax2.set_ylabel("g − r [mag]")

        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"Plot saved to {args.plot}")

if __name__ == "__main__":
    main()