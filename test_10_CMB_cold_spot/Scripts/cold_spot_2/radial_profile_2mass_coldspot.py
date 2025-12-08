#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from math import pi
from astropy.coordinates import SkyCoord
import astropy.units as u


def angular_separation_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
    Distância angular em graus entre (ra1, dec1) e (ra2, dec2)
    usando a fórmula do cosseno esférico.
    """
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)

    delta_ra = ra1 - ra2
    cos_theta = (
        np.sin(dec1) * np.sin(dec2)
        + np.cos(dec1) * np.cos(dec2) * np.cos(delta_ra)
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.degrees(np.arccos(cos_theta))
    return theta


def safe_float(x):
    """Retorna float(x) ou NaN caso seja inválido ('deg', '---', texto, etc)."""
    try:
        return float(x)
    except Exception:
        return np.nan


def main():

    infile = "2mass.tsv"
    print("=" * 80)
    print(f"[INFO] Lendo catálogo 2MASS recortado: {infile}")

    # Lê tudo como string para evitar problemas com 'deg', '---', etc
    df = pd.read_csv(infile, sep="\t", comment="#", dtype=str)

    print(f"[INFO] Linhas totais no arquivo: {len(df)}")

    # Aplica safe_float a RA e Dec
    if "RAJ2000" not in df.columns or "DEJ2000" not in df.columns:
        raise RuntimeError(
            "[ERRO] Colunas 'RAJ2000'/'DEJ2000' não encontradas no arquivo. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    df["RA"] = df["RAJ2000"].apply(safe_float)
    df["DEC"] = df["DEJ2000"].apply(safe_float)

    # Remove entradas inválidas
    df = df.dropna(subset=["RA", "DEC"])

    print(f"[INFO] Linhas com RA/Dec válidos após limpeza: {len(df)}")

    ra = df["RA"].astype(float).values
    dec = df["DEC"].astype(float).values

    # ----------------------------------------------------
    # Centro do SEGUNDO Cold Spot em coordenadas galácticas
    # ----------------------------------------------------
    L_CS_DEG = 208.4349
    B_CS_DEG = -55.8545

    cs_gal = SkyCoord(l=L_CS_DEG * u.deg,
                      b=B_CS_DEG * u.deg,
                      frame="galactic")
    cs_icrs = cs_gal.icrs
    ra_cs_deg = cs_icrs.ra.deg
    dec_cs_deg = cs_icrs.dec.deg

    print("[INFO] Centro do SEGUNDO Cold Spot:")
    print(f"  Galáctico: l = {L_CS_DEG:.4f} deg, b = {B_CS_DEG:.4f} deg")
    print(f"  ICRS/J2000: RA = {ra_cs_deg:.4f} deg, Dec = {dec_cs_deg:.4f} deg")

    radial_edges_deg = np.array([0, 2, 4, 6, 8, 10])

    print("[INFO] Calculando distâncias angulares...")
    theta_deg = angular_separation_deg(ra, dec, ra_cs_deg, dec_cs_deg)
    df["theta_deg"] = theta_deg

    print(f"[INFO] theta_deg min/max: {theta_deg.min():.2f} – {theta_deg.max():.2f}")

    results = []

    for i in range(len(radial_edges_deg) - 1):
        r_in = radial_edges_deg[i]
        r_out = radial_edges_deg[i + 1]

        mask = (theta_deg > r_in) & (theta_deg <= r_out)
        N = np.count_nonzero(mask)

        area = pi * (r_out**2 - r_in**2)
        dens = N / area if area > 0 else np.nan
        sigma = np.sqrt(N) / area if N > 0 else 0.0

        results.append((r_in, r_out, N, area, dens, sigma))

    print("\n[RESULTADO] Perfil radial (SEGUNDO Cold Spot)")
    print(" r_in  r_out   N     area      dens       sigma")
    for r in results:
        print(f" {r[0]:4.1f}  {r[1]:4.1f}  {r[2]:6d}  {r[3]:8.2f}  {r[4]:9.2f}  {r[5]:9.2f}")


if __name__ == "__main__":
    main()