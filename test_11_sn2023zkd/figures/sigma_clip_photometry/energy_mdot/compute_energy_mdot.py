#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute radiated energy from a bolometric curve and estimate Mdot(t) from CSM interaction.

Inputs:
  --bol: bolometric_curve.csv (must have columns:
        MJD and Lbol_erg_s   OR   MJD and Fbol_erg_s_cm2 with --dl_mpc)
Options:
  --dl_mpc    : luminosity distance in Mpc (only needed if file has Fbol, not Lbol)
  --vs_kms    : shock velocity [km/s] (default 6000)
  --vw_kms    : wind velocity  [km/s] (default 1000)
  --epsilon   : conversion efficiency in L ≈ 0.5*epsilon*(Mdot/vw)*vs^3 (default 0.3)
  --outdir    : output folder
  --peaks     : optional windows to integrate energy per peak (e.g., "60240,60270;60340,60380")

Outputs:
  - energy_mdot_summary.txt
  - mdot_timeseries.csv     (MJD, Lbol, E_rad_cum, Mdot)
  - fig_Lbol_Ecum_Mdot.png  (three panels)
"""

import argparse, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

MSUN_G     = 1.98847e33
SEC_DAY    = 86400.0
CM_PER_MPC = 3.085677581e24

def integrate_trapz(x, y):
    return float(np.trapz(y, x))

def parse_peak_windows(s):
    wins=[]
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk: continue
        a,b = chunk.split(",")
        wins.append((float(a), float(b)))
    return wins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bol", required=True, help="bolometric_curve.csv")
    ap.add_argument("--dl_mpc", type=float, default=None, help="distance (Mpc) if only Fbol is provided")
    ap.add_argument("--vs_kms", type=float, default=6000.0)
    ap.add_argument("--vw_kms", type=float, default=1000.0)
    ap.add_argument("--epsilon", type=float, default=0.3)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--peaks", default=None, help='e.g. "60240,60270;60340,60380"')
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.bol)

    # pick columns
    col_time = "MJD" if "MJD" in df.columns else ("mjd" if "mjd" in df.columns else None)
    if col_time is None: raise SystemExit("Time column not found (need MJD).")
    col_L = "Lbol_erg_s" if "Lbol_erg_s" in df.columns else None
    col_F = "Fbol_erg_s_cm2" if "Fbol_erg_s_cm2" in df.columns else None
    if (col_L is None) and (col_F is None):
        raise SystemExit("Need Lbol_erg_s or Fbol_erg_s_cm2 in the CSV.")

    df = df.sort_values(col_time).reset_index(drop=True)

    # If only Fbol given, convert to Lbol
    if col_L is None:
        if not args.dl_mpc or args.dl_mpc <= 0:
            raise SystemExit("Provide --dl_mpc to convert Fbol to Lbol.")
        DL_cm = args.dl_mpc * CM_PER_MPC
        df["Lbol_erg_s"] = 4*math.pi * (DL_cm**2) * df[col_F]
        col_L = "Lbol_erg_s"

    # arrays
    t = df[col_time].to_numpy(float)
    L = df[col_L].to_numpy(float)

    # cumulative radiated energy E_rad = ∫ L dt (in seconds)
    t_sec = (t - t.min()) * SEC_DAY
    E_rad = cumtrapz(L, t_sec, initial=0.0)   # <-- correção

    # Mdot(t) from L = 0.5 * epsilon * (Mdot/vw) * vs^3
    vs = args.vs_kms * 1e5  # km/s -> cm/s
    vw = args.vw_kms * 1e5
    eps = args.epsilon
    denom = 0.5 * eps * (vs**3) / vw  # L = denom * Mdot  ->  Mdot = L/denom
    if denom <= 0:
        raise SystemExit("Non-positive denominator in Mdot formula. Check vs, vw, epsilon.")
    mdot_cgs = L / denom  # g/s
    mdot_msun_yr = mdot_cgs * (SEC_DAY*365.25) / MSUN_G

    # Save series
    out_csv = outdir / "mdot_timeseries.csv"
    pd.DataFrame({
        "MJD": t,
        "Lbol_erg_s": L,
        "E_rad_erg_cum": E_rad,
        "Mdot_Msun_per_yr": mdot_msun_yr
    }).to_csv(out_csv, index=False)

    # Integrals
    E_total = float(E_rad[-1])

    # Optional per-peak integrals
    peak_report = []
    if args.peaks:
        wins = parse_peak_windows(args.peaks)
        for (a,b) in wins:
            sel = (t>=a) & (t<=b)
            if np.count_nonzero(sel) >= 2:
                tsec_w = (t[sel] - t[sel].min()) * SEC_DAY
                Ew = integrate_trapz(tsec_w, L[sel])
                peak_report.append((a,b,Ew))
            else:
                peak_report.append((a,b,float("nan")))

    # Summary
    out_txt = outdir / "energy_mdot_summary.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("=== Energy & Mdot Summary ===\n")
        f.write(f"N points: {len(df)}\n")
        f.write(f"E_total (erg): {E_total:.3e}\n")
        f.write(f"Assumptions: vs={args.vs_kms:.0f} km/s, vw={args.vw_kms:.0f} km/s, epsilon={args.epsilon:.2f}\n")
        if args.peaks:
            f.write("\nPer-peak energy (erg):\n")
            for (a,b,Ew) in peak_report:
                f.write(f"  MJD [{a:.2f}, {b:.2f}] : {Ew:.3e}\n")
        f.write("\nMdot range (Msun/yr):\n")
        f.write(f"  min={np.nanmin(mdot_msun_yr):.3e}, max={np.nanmax(mdot_msun_yr):.3e}\n")

    # Plot
    fig, axs = plt.subplots(3,1, figsize=(9,9), sharex=False)
    axs[0].plot(t, L, "o-", ms=3)
    axs[0].set_ylabel("Lbol [erg s$^{-1}$]")
    axs[0].set_title("Bolometric luminosity")
    axs[0].grid(alpha=0.3)

    axs[1].plot(t, E_rad, "o-", ms=3)
    axs[1].set_ylabel("E_cum [erg]")
    axs[1].set_title("Cumulative radiated energy")
    axs[1].grid(alpha=0.3)

    axs[2].plot(t, mdot_msun_yr, "o-", ms=3)
    axs[2].set_xlabel("MJD")
    axs[2].set_ylabel(r"$\dot M$ [M$_\odot$/yr]")
    axs[2].set_title("Mass-loss rate (interaction model)")
    axs[2].grid(alpha=0.3)

    fig.tight_layout()
    figp = outdir / "fig_Lbol_Ecum_Mdot.png"
    plt.savefig(figp, dpi=180)

    print(f"[ok] E_total = {E_total:.3e} erg")
    print(f"[ok] saved: {out_csv}")
    print(f"[ok] saved: {out_txt}")
    print(f"[ok] saved: {figp}")

if __name__ == "__main__":
    main()