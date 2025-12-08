#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import sin, cos, acos, radians, pi


def angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)

    delta_ra = ra1 - ra2
    cos_theta = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(delta_ra)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.degrees(np.arccos(cos_theta))
    return theta


def safe_float(x):
    """Retorna float(x) ou NaN caso seja inválido ('deg', '---', texto, etc)."""
    try:
        return float(x)
    except:
        return np.nan


def main():

    infile = "2mass.tsv"
    print("="*80)
    print(f"[INFO] Lendo catálogo 2MASS recortado: {infile}")

    # NÃO confiar no dtypes automáticos → ler tudo como string
    df = pd.read_csv(infile, sep="\t", comment="#", dtype=str)

    print(f"[INFO] Linhas totais no arquivo: {len(df)}")

    # Aplica safe_float a RA e Dec
    df["RA"]  = df["RAJ2000"].apply(safe_float)
    df["DEC"] = df["DEJ2000"].apply(safe_float)

    # Remove entradas inválidas
    df = df.dropna(subset=["RA", "DEC"])

    print(f"[INFO] Linhas com RA/Dec válidos após limpeza: {len(df)}")

    ra  = df["RA"].astype(float).values
    dec = df["DEC"].astype(float).values

    # Cold Spot center (ICRS)
    ra_cs_deg  = 48.2999
    dec_cs_deg = -20.4373

    radial_edges_deg = np.array([0, 2, 4, 6, 8, 10])

    print("[INFO] Calculando distâncias angulares...")
    theta_deg = angular_separation_deg(ra, dec, ra_cs_deg, dec_cs_deg)
    df["theta_deg"] = theta_deg

    print(f"[INFO] theta_deg min/max: {theta_deg.min():.2f} – {theta_deg.max():.2f}")

    results = []

    for i in range(len(radial_edges_deg)-1):
        r_in  = radial_edges_deg[i]
        r_out = radial_edges_deg[i+1]

        mask = (theta_deg > r_in) & (theta_deg <= r_out)
        N = np.count_nonzero(mask)

        area = pi * (r_out**2 - r_in**2)
        dens = N / area
        sigma = np.sqrt(N) / area if N>0 else 0

        results.append((r_in, r_out, N, area, dens, sigma))

    print("\n[RESULTADO] Perfil radial")
    print(" r_in  r_out   N     area      dens       sigma")
    for r in results:
        print(f" {r[0]:4.1f}  {r[1]:4.1f}  {r[2]:6d}  {r[3]:8.2f}  {r[4]:9.2f}  {r[5]:9.2f}")


if __name__ == "__main__":
    main()