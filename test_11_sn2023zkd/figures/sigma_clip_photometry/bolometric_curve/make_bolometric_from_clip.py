#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a bolometric curve from cleaned/binned photometry (ATLAS+ZTF), with time-binning.
- Bin by time (e.g., 1.0 day), take median per band within each bin.
- Prefer ZTF g/r; else ATLAS cyan/orange; else the two most populated bands.
- Blackbody fit per bin (>=2 bands) in f_nu space; derive Fbol and (optionally) Lbol.

Usage example:
  python make_bolometric_from_clip.py --inp phot_all_clean_bin3d_clip_A.csv --outdir out --dl_mpc 253 --bin_days 1.0
"""

import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# cgs constants
h=6.62607015e-27; kB=1.380649e-16; c=2.99792458e10; sigma=5.670374419e-5; pi=math.pi

# Effective wavelengths (meters)
LAMBDA_M = {
    "ztf_g": 4723e-10, "ztf_r": 6339e-10,
    "cyan":  5330e-10, "orange": 6790e-10,
    "g": 4723e-10, "r": 6339e-10, "c": 5330e-10, "o": 6790e-10
}

def norm_band(b):
    if not isinstance(b, str): return None
    s = b.strip().lower().replace("-", "_")
    s = s.replace("ztf-","ztf_").replace("ztf ","ztf_").replace("atlas_","")
    # common aliases
    if s in ("g","r","cyan","orange","ztf_g","ztf_r"): return s
    if s in ("ztf_g","g","fid1","filter_g"): return "ztf_g" if s=="ztf_g" else "g"
    if s in ("ztf_r","r","fid2","filter_r"): return "ztf_r" if s=="ztf_r" else "r"
    if s in ("c","cyan"): return "cyan"
    if s in ("o","orange"): return "orange"
    return s

def band_lambda(b):
    if b is None: return np.nan
    key = b
    if key not in LAMBDA_M:
        # map loose 'g','r' to ZTF passbands by default
        if key == "g": key = "ztf_g"
        if key == "r": key = "ztf_r"
        if key == "c": key = "cyan"
        if key == "o": key = "orange"
    return LAMBDA_M.get(key, np.nan)

def abmag_to_fnu(mag):
    return 10.0**(-0.4*(mag + 48.60))  # erg/s/cm^2/Hz

def planck_nu(nu, T):
    x = (h*nu)/(kB*T)
    x = np.clip(x, 1e-6, 700)  # numerical safety
    return (2.0*h*nu**3 / c**2) / np.expm1(x)

def fit_tA(nu, fnu, w):
    # coarse+fine grid improves robustness and keeps speed reasonable
    Tgrid = np.unique(np.concatenate([
        np.linspace(3000,12000,181),
        np.linspace(12000,25000,65)
    ]))
    best=(np.nan,np.nan,np.inf)
    for T in Tgrid:
        B = planck_nu(nu, T)
        den = np.sum(w*B*B)
        if den<=0: continue
        A = np.sum(w*fnu*B)/den
        chi2 = np.sum(w*(fnu - A*B)**2)
        if chi2<best[2]: best=(float(T),float(A),float(chi2))
    return best

def choose_subset(df_bin):
    """Prefer ZTF g/r; else ATLAS cyan/orange; else top-2 bands by count."""
    bands = set(df_bin["band_norm"])
    if {"ztf_g","ztf_r"} <= bands:
        return df_bin[df_bin["band_norm"].isin(["ztf_g","ztf_r"])]
    if {"cyan","orange"} <= bands:
        return df_bin[df_bin["band_norm"].isin(["cyan","orange"])]
    top = (df_bin["band_norm"].value_counts().index.tolist())[:2]
    return df_bin[df_bin["band_norm"].isin(top)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="cleaned/clip CSV (e.g., phot_all_clean_bin3d_clip_A.csv)")
    ap.add_argument("--outdir", required=True, help="output folder")
    ap.add_argument("--dl_mpc", type=float, default=None, help="luminosity distance (Mpc) to compute Lbol")
    ap.add_argument("--bin_days", type=float, default=1.0, help="time-bin width in days (default 1.0)")
    args = ap.parse_args()

    outdir=Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.inp)

    # flexible columns
    cols = {c.lower(): c for c in df.columns}
    time_col = cols.get("mjd") or cols.get("jd") or cols.get("date")
    mag_col  = cols.get("mag") or cols.get("m")
    band_col = cols.get("band") or cols.get("filter") or cols.get("fid")
    merr_col = cols.get("mag_err") or cols.get("magerr") or cols.get("err_mag") or cols.get("err")

    if not time_col: raise SystemExit("Time column not found (need MJD/JD/Date).")
    if not mag_col:  raise SystemExit("Magnitude column not found (need mag).")
    if not band_col: raise SystemExit("Band column not found (need band/filter/fid).")

    df = df.rename(columns={time_col:"mjd", mag_col:"mag", band_col:"band"})
    if merr_col: df = df.rename(columns={merr_col:"mag_err"})

    # clean
    df = df.dropna(subset=["mjd","mag","band"]).copy()
    df["band_norm"] = df["band"].apply(norm_band)
    df = df.dropna(subset=["band_norm"]).copy()

    # time-binning
    binw = float(args.bin_days)
    df["bin_id"] = np.floor(df["mjd"]/binw)*binw

    # median per band within each bin
    agg = {"mag":"median"}
    if "mag_err" in df.columns:
        agg["mag_err"] = "median"
    df_b = (df.groupby(["bin_id","band_norm"], as_index=False)
              .agg(agg))

    # attach wavelengths and frequencies
    df_b["lambda_m"] = df_b["band_norm"].apply(band_lambda)
    df_b = df_b[np.isfinite(df_b["lambda_m"])].copy()
    df_b["nu"] = c/(df_b["lambda_m"]*100.0)
    df_b["fnu"] = abmag_to_fnu(df_b["mag"])

    rows=[]
    for bini, d in df_b.groupby("bin_id"):
        sub = choose_subset(d.copy())
        if len(sub) < 2:  # still not enough bands
            continue

        nu  = sub["nu"].to_numpy(float)
        fnu = sub["fnu"].to_numpy(float)

        if "mag_err" in sub.columns and np.isfinite(sub["mag_err"]).any():
            dfnu = np.log(10)*0.4 * fnu * np.clip(sub["mag_err"].to_numpy(float), 1e-4, 5.0)
            w = 1.0/np.clip(dfnu,1e-30,np.inf)**2
        else:
            w = np.ones_like(fnu)

        T,A,chi2 = fit_tA(nu,fnu,w)
        if not (np.isfinite(T) and np.isfinite(A) and A>0):
            continue

        # Fbol = A * sigma * T^4 (see derivation in the previous version)
        Fbol = A * sigma * (T**4)

        out = {"MJD": float(bini + 0.5*binw),  # bin center
               "Fbol_erg_s_cm2": float(Fbol),
               "T_K": float(T),
               "A_norm": float(A),
               "n_bands": int(len(sub))}
        if args.dl_mpc and args.dl_mpc>0:
            DL_cm = args.dl_mpc*3.085677581e24
            out["Lbol_erg_s"] = 4*pi*(DL_cm**2)*Fbol
        rows.append(out)

    if not rows:
        raise SystemExit("No time-bins with >=2 useful bands. Try increasing --bin_days (e.g., 2.0) or check band labels.")

    bol = pd.DataFrame(rows).sort_values("MJD").reset_index(drop=True)
    bol_path = outdir/"bolometric_curve.csv"
    bol.to_csv(bol_path, index=False)

    # quick plot
    plt.figure(figsize=(8,5))
    y = bol["Lbol_erg_s"] if "Lbol_erg_s" in bol.columns else bol["Fbol_erg_s_cm2"]
    plt.plot(bol["MJD"], y, "o-")
    plt.xlabel("MJD")
    plt.ylabel("Lbol [erg s$^{-1}$]" if "Lbol_erg_s" in bol.columns else "Fbol [erg s$^{-1}$ cm$^{-2}$]")
    plt.title(f"Bolometric curve (time-binned {binw:.2f} d; blackbody per bin)")
    plt.tight_layout()
    fig_path = outdir/"fig_bolometric_curve.png"
    plt.savefig(fig_path, dpi=160)
    print(f"Generated:\n  {bol_path}\n  {fig_path}")
    print(f"Bins: {len(bol)} | T(K): {bol['T_K'].min():.0f}–{bol['T_K'].max():.0f}")

if __name__ == "__main__":
    main()