#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Monte Carlo radial density test around the CMB Cold Spot
using 2MASS K-band counts.

- Lê 2mass.tsv (catálogo recortado ao redor do Cold Spot, formato VizieR TSV)
- Usa Kmag em [8, 16]
- Bins radiais: [0, 2, 4, 6, 8, 10] graus
- Ajusta:
    * modelo constante: dens(r) = rho0
    * modelo linear:    dens(r) = a + b r
- Gera N_SIM realizações Poisson sob o modelo constante e
  compara b_obs e chi2_const_obs com as distribuições simuladas.

Saídas:
- Impressão no terminal
- Arquivo texto: 2mass_mc_coldspot_results.txt
"""

import numpy as np
import pandas as pd
import os

# SciPy é opcional: só para p-valor analítico do chi2
try:
    from scipy.stats import chi2
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------
# Parâmetros principais
# ---------------------------------------------------------

INPUT_FILE = "2mass.tsv"   # catálogo recortado
KMIN = 8.0
KMAX = 16.0

# Bins radiais (em graus)
RADIAL_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Centro do Cold Spot (coordenadas galácticas e equatoriais já convertidas)
L_CS = 209.0     # deg (galáctico)
B_CS = -57.0     # deg (galáctico)
RA_CS = 48.2999  # deg (ICRS)
DEC_CS = -20.4373  # deg

# Número de simulações Monte Carlo
N_SIM = 10000

# Semente para reprodutibilidade
RANDOM_SEED = 12345


# ---------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------

def angsep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
    Distância angular entre dois pontos na esfera (em graus),
    usando lei dos cossenos esférica.
    """
    ra1 = np.deg2rad(ra1_deg)
    dec1 = np.deg2rad(dec1_deg)
    ra2 = np.deg2rad(ra2_deg)
    dec2 = np.deg2rad(dec2_deg)

    cos_theta = (np.sin(dec1) * np.sin(dec2) +
                 np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    # proteção numérica
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.rad2deg(theta)


def build_radial_profile(theta_deg, Kmag, radial_bins, kmin, kmax):
    """
    Constrói perfil radial de contagens e densidades para
    objetos com Kmag em [kmin, kmax].

    Retorna:
        r_centers, N, areas, densidades, erros
    """
    # Seleciona faixa de magnitude
    mask_mag = (Kmag >= kmin) & (Kmag < kmax)
    theta = theta_deg[mask_mag]

    n_bins = len(radial_bins) - 1
    r_centers = 0.5 * (radial_bins[:-1] + radial_bins[1:])
    counts = np.zeros(n_bins, dtype=np.int64)
    areas = np.zeros(n_bins, dtype=float)
    densities = np.zeros(n_bins, dtype=float)
    errors = np.zeros(n_bins, dtype=float)

    for i in range(n_bins):
        r_in = radial_bins[i]
        r_out = radial_bins[i+1]

        mask_bin = (theta >= r_in) & (theta < r_out)
        N_i = np.sum(mask_bin)
        counts[i] = N_i

        # Área do anel em deg^2: pi * (r_out^2 - r_in^2)
        area_i = np.pi * (r_out**2 - r_in**2)
        areas[i] = area_i

        if N_i > 0:
            dens_i = N_i / area_i
            sigma_i = np.sqrt(N_i) / area_i
        else:
            dens_i = 0.0
            sigma_i = np.inf

        densities[i] = dens_i
        errors[i] = sigma_i

    return r_centers, counts, areas, densities, errors


def fit_constant_model(densities, errors):
    """
    Ajuste de modelo constante dens(r) = rho0 via mínimos quadrados ponderados.

    Retorna:
        rho0, chi2, chi2_red, dof
    """
    w = 1.0 / (errors**2)
    rho0 = np.sum(w * densities) / np.sum(w)
    chi2_val = np.sum(w * (densities - rho0)**2)
    dof = len(densities) - 1  # 1 parâmetro ajustado
    chi2_red = chi2_val / dof if dof > 0 else np.nan
    return rho0, chi2_val, chi2_red, dof


def fit_linear_model(r, densities, errors):
    """
    Ajuste linear dens(r) = a + b r via mínimos quadrados ponderados.

    Retorna:
        a, b, sigma_a, sigma_b, chi2, chi2_red, dof
    """
    w = 1.0 / (errors**2)
    x = r
    y = densities

    Sw = np.sum(w)
    Swx = np.sum(w * x)
    Swy = np.sum(w * y)
    Swx2 = np.sum(w * x * x)
    Swxy = np.sum(w * x * y)

    Delta = Sw * Swx2 - Swx**2
    a = (Swy * Swx2 - Swx * Swxy) / Delta
    b = (Sw * Swxy - Swx * Swy) / Delta

    # variâncias
    var_a = Swx2 / Delta
    var_b = Sw / Delta
    sigma_a = np.sqrt(var_a)
    sigma_b = np.sqrt(var_b)

    # chi2 do ajuste linear
    model = a + b * x
    chi2_val = np.sum(w * (y - model)**2)
    dof = len(y) - 2
    chi2_red = chi2_val / dof if dof > 0 else np.nan

    return a, b, sigma_a, sigma_b, chi2_val, chi2_red, dof


# ---------------------------------------------------------
# Script principal
# ---------------------------------------------------------

def main():
    np.random.seed(RANDOM_SEED)

    print("="*57)
    print("[INFO] Lendo catálogo 2MASS recortado:", INPUT_FILE)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Arquivo {INPUT_FILE} não encontrado.")

    # Lê TSV do VizieR, ignorando linhas de comentário
    df = pd.read_csv(INPUT_FILE, sep="\t", comment="#", dtype=str, low_memory=False)
    n_total = len(df)
    print("[INFO] Linhas totais no arquivo:", n_total)

    # Limpeza básica: remove linhas com "deg", "----" etc. nas colunas de coordenadas
    for col in ["RAJ2000", "DEJ2000", "Kmag"]:
        if col not in df.columns:
            raise KeyError(f"Coluna {col} não encontrada em {INPUT_FILE}.")

    mask_valid = (~df["RAJ2000"].isin(["RAJ2000", "----------", "deg", ""])) & \
                 (~df["DEJ2000"].isin(["DEJ2000", "----------", "deg", ""])) & \
                 (~df["Kmag"].isin(["Kmag", "------", "", "     "]))

    df = df[mask_valid].copy()
    n_valid = len(df)
    print("[INFO] Linhas com RA/Dec e Kmag válidos:", n_valid)

    # Converte para float
    ra = df["RAJ2000"].astype(float).values
    dec = df["DEJ2000"].astype(float).values
    Kmag = df["Kmag"].astype(float).values

    # Distância angular ao centro do Cold Spot
    theta_deg = angsep_deg(ra, dec, RA_CS, DEC_CS)
    print("[INFO] theta_deg min/max: %.2f – %.2f deg" %
          (theta_deg.min(), theta_deg.max()))

    # Perfil radial observado
    print("="*57)
    print("[INFO] Construindo perfil radial observado (K em [%.1f, %.1f])..." %
          (KMIN, KMAX))

    r_centers, counts_obs, areas_obs, dens_obs, err_obs = build_radial_profile(
        theta_deg, Kmag, RADIAL_BINS, KMIN, KMAX
    )

    print("[INFO] Perfil radial bruto:")
    print(" r_in  r_out   N      area_deg2   dens[1/deg2]   sigma")
    for i in range(len(r_centers)):
        r_in = RADIAL_BINS[i]
        r_out = RADIAL_BINS[i+1]
        print(" %4.1f  %4.1f  %7d   %8.2f    %10.3f   %7.3f" %
              (r_in, r_out, counts_obs[i], areas_obs[i],
               dens_obs[i], err_obs[i]))

    # Ajuste constante (modelo isotrópico simples)
    rho0, chi2_const_obs, chi2_red_const_obs, dof_const = fit_constant_model(
        dens_obs, err_obs
    )

    print("="*57)
    print("[RESULT] Ajuste modelo constante (baseline tipo LCDM local):")
    print("  rho0_const       = %.3f sources/deg^2" % rho0)
    print("  chi2_const       = %.3f" % chi2_const_obs)
    print("  chi2_red_const   = %.3f" % chi2_red_const_obs)
    print("  dof_const        = %d" % dof_const)
    if HAVE_SCIPY:
        p_const = chi2.sf(chi2_const_obs, dof_const)
        print("  p_value_const    = %.3e" % p_const)
    else:
        print("  p_value_const    = (SciPy não disponível)")

    # Ajuste linear dens(r) = a + b r
    a_obs, b_obs, sigma_a_obs, sigma_b_obs, chi2_lin_obs, chi2_red_lin_obs, dof_lin = \
        fit_linear_model(r_centers, dens_obs, err_obs)

    print("="*57)
    print("[RESULT] Ajuste linear dens(r) = a + b r:")
    print("  a (intercepto)   = %.3f ± %.3f sources/deg^2" %
          (a_obs, sigma_a_obs))
    print("  b (inclinação)   = %.3f ± %.3f sources/deg^2/deg" %
          (b_obs, sigma_b_obs))
    print("  significancia_b  = %.2f sigma" % (b_obs / sigma_b_obs))
    print("  chi2_linear      = %.3f" % chi2_lin_obs)
    print("  chi2_red_linear  = %.3f" % chi2_red_lin_obs)
    print("  dof_linear       = %d" % dof_lin)
    if HAVE_SCIPY:
        p_lin = chi2.sf(chi2_lin_obs, dof_lin)
        print("  p_value_linear   = %.3e" % p_lin)
    else:
        print("  p_value_linear   = (SciPy não disponível)")

    # -----------------------------------------------------
    # Monte Carlo sob o modelo constante
    # -----------------------------------------------------
    print("="*57)
    print("[INFO] Iniciando Monte Carlo com N_SIM = %d..." % N_SIM)
    print("       Hipótese nula: densidade constante rho0 = %.3f" % rho0)

    lambdas = rho0 * areas_obs  # contagens esperadas em cada anel

    b_sims = np.zeros(N_SIM, dtype=float)
    chi2_const_sims = np.zeros(N_SIM, dtype=float)

    for i_sim in range(N_SIM):
        # Sorteia contagens Poisson em cada anel
        N_sim = np.random.poisson(lam=lambdas)

        # Evita divisão por zero na estimativa de erro
        N_eff = np.where(N_sim > 0, N_sim, 1)

        dens_sim = N_sim / areas_obs
        err_sim = np.sqrt(N_eff) / areas_obs

        # slope sob a simulação
        a_sim, b_sim, _, _, chi2_lin_sim, _, _ = fit_linear_model(
            r_centers, dens_sim, err_sim
        )
        b_sims[i_sim] = b_sim

        # chi2 da simulação contra modelo constante com rho0 fixo
        w_sim = 1.0 / (err_sim**2)
        chi2_const_sim = np.sum(w_sim * (dens_sim - rho0)**2)
        chi2_const_sims[i_sim] = chi2_const_sim

        # feedback ocasional
        if (i_sim + 1) % max(1, (N_SIM // 10)) == 0:
            print("[INFO] Monte Carlo: %5d / %5d simulações concluídas..."
                  % (i_sim + 1, N_SIM))

    # Estatísticas das distribuições simuladas
    mean_b = np.mean(b_sims)
    std_b = np.std(b_sims)
    mean_chi2 = np.mean(chi2_const_sims)
    std_chi2 = np.std(chi2_const_sims)

    # p-valores empíricos
    p_mc_slope = np.mean(np.abs(b_sims) >= np.abs(b_obs))
    p_mc_chi2 = np.mean(chi2_const_sims >= chi2_const_obs)

    print("="*57)
    print("[RESULT] Monte Carlo – distribuição de slopes b:")
    print("  mean(b_sim)      = %.3f sources/deg^2/deg" % mean_b)
    print("  std(b_sim)       = %.3f sources/deg^2/deg" % std_b)
    print("  b_obs            = %.3f" % b_obs)
    print("  p_mc_slope       = %.3e" % p_mc_slope)

    print("="*57)
    print("[RESULT] Monte Carlo – distribuição de chi2_const:")
    print("  mean(chi2_sim)   = %.3f" % mean_chi2)
    print("  std(chi2_sim)    = %.3f" % std_chi2)
    print("  chi2_const_obs   = %.3f" % chi2_const_obs)
    print("  p_mc_chi2        = %.3e" % p_mc_chi2)

    # -----------------------------------------------------
    # Salva resultados em arquivo texto
    # -----------------------------------------------------
    outfile = "2mass_mc_coldspot_results.txt"
    with open(outfile, "w") as f:
        f.write("Monte Carlo radial density test around the CMB Cold Spot\n")
        f.write("Input catalog : %s\n" % INPUT_FILE)
        f.write("Cold Spot center (RA,Dec) = (%.4f, %.4f) deg\n" %
                (RA_CS, DEC_CS))
        f.write("Radial bins (deg): %s\n" % RADIAL_BINS.tolist())
        f.write("Kmag range: [%.1f, %.1f]\n" % (KMIN, KMAX))
        f.write("\nObserved radial profile:\n")
        f.write("r_in_deg r_out_deg N area_deg2 density sigma\n")
        for i in range(len(r_centers)):
            r_in = RADIAL_BINS[i]
            r_out = RADIAL_BINS[i+1]
            f.write("%.2f %.2f %d %.5f %.5f %.5f\n" %
                    (r_in, r_out, counts_obs[i], areas_obs[i],
                     dens_obs[i], err_obs[i]))

        f.write("\nConstant model fit:\n")
        f.write("rho0_const = %.6f\n" % rho0)
        f.write("chi2_const = %.6f\n" % chi2_const_obs)
        f.write("chi2_red_const = %.6f\n" % chi2_red_const_obs)
        f.write("dof_const = %d\n" % dof_const)
        if HAVE_SCIPY:
            f.write("p_value_const = %.6e\n" % chi2.sf(chi2_const_obs, dof_const))
        else:
            f.write("p_value_const = (SciPy not available)\n")

        f.write("\nLinear model fit (density = a + b * r):\n")
        f.write("a = %.6f ± %.6f\n" % (a_obs, sigma_a_obs))
        f.write("b = %.6f ± %.6f\n" % (b_obs, sigma_b_obs))
        f.write("significance_b = %.3f sigma\n" % (b_obs / sigma_b_obs))
        f.write("chi2_linear = %.6f\n" % chi2_lin_obs)
        f.write("chi2_red_linear = %.6f\n" % chi2_red_lin_obs)
        f.write("dof_linear = %d\n" % dof_lin)
        if HAVE_SCIPY:
            f.write("p_value_linear = %.6e\n" % chi2.sf(chi2_lin_obs, dof_lin))
        else:
            f.write("p_value_linear = (SciPy not available)\n")

        f.write("\nMonte Carlo summary (N_SIM = %d):\n" % N_SIM)
        f.write("mean_b_sim = %.6f\n" % mean_b)
        f.write("std_b_sim  = %.6f\n" % std_b)
        f.write("b_obs      = %.6f\n" % b_obs)
        f.write("p_mc_slope = %.6e\n" % p_mc_slope)
        f.write("mean_chi2_sim = %.6f\n" % mean_chi2)
        f.write("std_chi2_sim  = %.6f\n" % std_chi2)
        f.write("chi2_const_obs = %.6f\n" % chi2_const_obs)
        f.write("p_mc_chi2 = %.6e\n" % p_mc_chi2)

    print("="*57)
    print("[INFO] Resultados detalhados salvos em:", outfile)
    print("[OK] Fim do Monte Carlo.")


if __name__ == "__main__":
    main()