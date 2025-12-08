#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte Carlo radial density test around the SECOND CMB Cold Spot
using 2MASS K-band counts.

Agora atualizado com as coordenadas do segundo cold spot:
  l = 208.4349 deg
  b = -55.8545 deg
  RA = 49.755 deg
  Dec = -18.350 deg
"""

import numpy as np
import pandas as pd
import os

try:
    from scipy.stats import chi2
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------
# PARÂMETROS PRINCIPAIS
# ---------------------------------------------------------

INPUT_FILE = "2mass.tsv"

KMIN = 8.0
KMAX = 16.0

RADIAL_BINS = np.array([0, 2, 4, 6, 8, 10], dtype=float)

# ----------- NOVO COLD SPOT -----------
# Coordenadas galácticas
L_CS = 208.4349
B_CS = -55.8545

# Coordenadas ICRS (convertidas corretamente)
RA_CS  = 49.7550
DEC_CS = -18.3500
# ---------------------------------------

N_SIM = 10000
RANDOM_SEED = 12345


# ---------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------

def angsep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)

    cos_theta = (np.sin(dec1)*np.sin(dec2) +
                 np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_theta))


def build_radial_profile(theta_deg, Kmag, radial_bins, kmin, kmax):
    mask_mag = (Kmag >= kmin) & (Kmag < kmax)
    theta = theta_deg[mask_mag]

    n_bins = len(radial_bins) - 1
    r_centers = 0.5*(radial_bins[:-1] + radial_bins[1:])
    counts = np.zeros(n_bins, dtype=int)
    areas = np.zeros(n_bins)
    density = np.zeros(n_bins)
    sigma = np.zeros(n_bins)

    for i in range(n_bins):
        r_in = radial_bins[i]
        r_out = radial_bins[i+1]

        mask = (theta >= r_in) & (theta < r_out)
        N = np.sum(mask)
        area = np.pi * (r_out**2 - r_in**2)

        counts[i] = N
        areas[i] = area

        if N > 0:
            density[i] = N / area
            sigma[i] = np.sqrt(N) / area
        else:
            density[i] = 0
            sigma[i] = np.inf

    return r_centers, counts, areas, density, sigma


def fit_constant_model(dens, errs):
    w = 1 / errs**2
    rho0 = np.sum(w * dens) / np.sum(w)
    chi2_val = np.sum(w * (dens - rho0)**2)
    dof = len(dens) - 1
    chi2_red = chi2_val / dof
    return rho0, chi2_val, chi2_red, dof


def fit_linear_model(r, dens, errs):
    w = 1 / errs**2
    Sw = np.sum(w)
    Swr = np.sum(w*r)
    Swd = np.sum(w*dens)
    Swr2 = np.sum(w*r*r)
    Swrd = np.sum(w*r*dens)

    Delta = Sw*Swr2 - Swr**2

    a = (Swr2*Swd - Swr*Swrd) / Delta
    b = (Sw*Swrd - Swr*Swd) / Delta

    sigma_a = np.sqrt(Swr2 / Delta)
    sigma_b = np.sqrt(Sw / Delta)

    model = a + b*r
    chi2_val = np.sum(w * (dens - model)**2)
    dof = len(r) - 2
    chi2_red = chi2_val / dof

    return a, b, sigma_a, sigma_b, chi2_val, chi2_red, dof


# ---------------------------------------------------------
# SCRIPT PRINCIPAL
# ---------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)

    print("="*60)
    print("[INFO] Lendo catálogo:", INPUT_FILE)

    df = pd.read_csv(INPUT_FILE, sep="\t", comment="#", dtype=str)
    print("[INFO] Linhas totais:", len(df))

    for col in ["RAJ2000", "DEJ2000", "Kmag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["RAJ2000", "DEJ2000", "Kmag"])

    ra = df["RAJ2000"].values
    dec = df["DEJ2000"].values
    k = df["Kmag"].values

    theta_deg = angsep_deg(ra, dec, RA_CS, DEC_CS)

    print("[INFO] theta range: %.2f – %.2f deg" %
          (theta_deg.min(), theta_deg.max()))

    # Perfil radial observado
    r_mid, N_obs, areas, dens_obs, err_obs = build_radial_profile(
        theta_deg, k, RADIAL_BINS, KMIN, KMAX
    )

    # Modelo constante
    rho0, chi2_const_obs, chi2_red_const, dof_const = fit_constant_model(
        dens_obs, err_obs
    )

    # Ajuste linear
    a_obs, b_obs, sa_obs, sb_obs, chi2_lin, chi2_red_lin, dof_lin = \
        fit_linear_model(r_mid, dens_obs, err_obs)

    print("[INFO] inclinação observada b =", b_obs)

    # Monte Carlo
    lambdas = rho0 * areas
    b_sims = []
    chi2_sims = []

    print("[INFO] Rodando Monte Carlo...")

    for i in range(N_SIM):
        Nsim = np.random.poisson(lambdas)
        N_eff = np.where(Nsim > 0, Nsim, 1)
        dens_sim = Nsim / areas
        err_sim = np.sqrt(N_eff) / areas

        _, b_sim, _, _, chi2_sim, _, _ = fit_linear_model(
            r_mid, dens_sim, err_sim
        )
        b_sims.append(b_sim)

        w = 1 / err_sim**2
        chi2_sims.append(np.sum(w * (dens_sim - rho0)**2))

    b_sims = np.array(b_sims)
    chi2_sims = np.array(chi2_sims)

    p_slope = np.mean(np.abs(b_sims) >= np.abs(b_obs))
    p_chi2 = np.mean(chi2_sims >= chi2_const_obs)

    print("[RESULT] b_obs =", b_obs)
    print("[RESULT] p_mc_slope =", p_slope)
    print("[RESULT] p_mc_chi2 =", p_chi2)

    print("[OK] Finalizado.")


if __name__ == "__main__":
    main()