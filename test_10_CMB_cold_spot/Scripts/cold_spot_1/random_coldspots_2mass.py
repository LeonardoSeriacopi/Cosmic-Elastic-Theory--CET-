#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Random Cold Spots test em torno do CMB Cold Spot usando 2MASS recortado.

- Usa o catálogo 2MASS recortado (2mass.tsv), já centrado em um cone de 10°.
- Calcula o perfil radial observado em torno do Cold Spot real.
- Gera centros aleatórios dentro do mesmo cone (amostrando posições de galáxias).
- Para cada centro aleatório:
    - Calcula o perfil radial de densidade.
    - Ajusta modelo constante (tipo LCDM local).
    - Ajusta modelo linear dens(r) = a + b r.
- Compara o slope observado b_obs com a distribuição de slopes gerados aleatoriamente.
- Compara também o chi2 do modelo constante observado com a distribuição de chi2 sob isotropia.

Saída: estatísticas no terminal e arquivo de resumo:
    random_coldspots_2mass_results.txt
"""

import numpy as np
import pandas as pd
from math import radians, sin, cos, acos, pi

# ---------------------------
# Configurações principais
# ---------------------------

INFILE = "2mass.tsv"

# Centro do Cold Spot em coordenadas galácticas (já usado antes)
L_CS = 209.0  # deg
B_CS = -57.0  # deg

# Centro do Cold Spot em equatorial (ICRS)
# (valor que você já usou/testou antes)
RA_CS = 48.2999   # deg
DEC_CS = -20.4373 # deg

R_MAX_DEG = 10.0
RADIAL_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

KMIN = 8.0
KMAX = 16.0

N_SIM = 5000  # número de centros aleatórios para Monte Carlo


# ---------------------------
# Funções auxiliares
# ---------------------------

def ang_sep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
    Distância angular em graus entre dois pontos na esfera (RA, Dec), em graus.
    Usa fórmula do cosseno esférico com clamp numérico.
    """
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)

    cos_theta = (
        np.sin(dec1) * np.sin(dec2) +
        np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    return np.degrees(theta)


def build_radial_profile(ra_deg, dec_deg, kmag, ra0_deg, dec0_deg,
                         radial_bins, kmin, kmax):
    """
    Constrói perfil radial (densidade vs raio) ao redor de (ra0, dec0).

    Retorna:
        r_centers_deg, densities, sigmas, Ns
    onde:
        densities = N_bin / area_bin
        sigmas = sqrt(N_bin) / area_bin  (erro Poisson em densidade)
    """
    # Seleciona faixa de magnitude
    mask_mag = (kmag >= kmin) & (kmag < kmax)
    ra_sel = ra_deg[mask_mag]
    dec_sel = dec_deg[mask_mag]

    # Distância angular até o centro
    theta = ang_sep_deg(ra_sel, dec_sel, ra0_deg, dec0_deg)

    # Considera apenas até R_MAX
    mask_r = theta <= radial_bins[-1]
    theta = theta[mask_r]

    # Perfil radial
    r_in = radial_bins[:-1]
    r_out = radial_bins[1:]
    r_centers = 0.5 * (r_in + r_out)

    # Áreas dos anéis (em deg^2): pi (r2^2 - r1^2)
    area = pi * (r_out**2 - r_in**2)

    densities = []
    sigmas = []
    Ns = []

    for rin, rout, a in zip(r_in, r_out, area):
        m_bin = (theta > rin) & (theta <= rout)
        N = np.sum(m_bin)
        Ns.append(N)
        if N > 0:
            dens = N / a
            sigma = np.sqrt(N) / a
        else:
            dens = 0.0
            sigma = np.inf   # erro infinito => peso zero
        densities.append(dens)
        sigmas.append(sigma)

    return r_centers, np.array(densities), np.array(sigmas), np.array(Ns)


def fit_constant_density(densities, sigmas):
    """
    Ajuste de densidade constante (rho0) com pesos 1/sigma^2.
    Retorna rho0, chi2, dof.
    """
    w = np.zeros_like(densities)
    valid = np.isfinite(sigmas) & (sigmas > 0)
    w[valid] = 1.0 / (sigmas[valid]**2)

    # Média ponderada
    Sw = np.sum(w)
    if Sw <= 0:
        return np.nan, np.nan, 0

    rho0 = np.sum(w * densities) / Sw

    # Chi^2
    chi2 = np.sum(w * (densities - rho0)**2)
    dof = np.sum(valid) - 1

    return rho0, chi2, dof


def fit_linear_density(r_centers, densities, sigmas):
    """
    Ajuste linear dens(r) = a + b r com pesos 1/sigma^2.
    Retorna:
        a, b, sigma_a, sigma_b, chi2, dof
    """
    valid = np.isfinite(sigmas) & (sigmas > 0)
    r = r_centers[valid]
    d = densities[valid]
    s = sigmas[valid]
    if len(r) < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    w = 1.0 / (s**2)

    Sw = np.sum(w)
    Swr = np.sum(w * r)
    Swd = np.sum(w * d)
    Swr2 = np.sum(w * r**2)
    Swrd = np.sum(w * r * d)

    Delta = Sw * Swr2 - Swr**2
    if Delta == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, 0)

    a = (Swr2 * Swd - Swr * Swrd) / Delta
    b = (Sw * Swrd - Swr * Swd) / Delta

    # Erros
    sigma_a = np.sqrt(Swr2 / Delta)
    sigma_b = np.sqrt(Sw / Delta)

    # Chi^2
    d_model = a + b * r
    chi2 = np.sum(w * (d - d_model)**2)
    dof = len(r) - 2

    return a, b, sigma_a, sigma_b, chi2, dof


# ---------------------------
# Script principal
# ---------------------------

def main():
    print("=========================================================")
    print(f"[INFO] Lendo catálogo 2MASS recortado: {INFILE}")

    # Lê o TSV; comenta linhas que começam com '#'
    df = pd.read_csv(INFILE, sep="\t", comment="#")

    # Limpa RA/Dec/Kmag
    for col in ["RAJ2000", "DEJ2000", "Kmag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["RAJ2000", "DEJ2000", "Kmag"])
    ra = df["RAJ2000"].values
    dec = df["DEJ2000"].values
    kmag = df["Kmag"].values

    print(f"[INFO] Linhas com RA/Dec/Kmag válidos: {len(df)}")
    theta_cs = ang_sep_deg(ra, dec, RA_CS, DEC_CS)
    print(f"[INFO] theta_deg vs Cold Spot: min={theta_cs.min():.2f} deg, max={theta_cs.max():.2f} deg")
    print("=========================================================")

    # --------------------------------------
    # Perfil radial OBSERVADO (Cold Spot real)
    # --------------------------------------
    print("[INFO] Construindo perfil radial observado (K em [8.0, 16.0])...")
    r_centers, dens_obs, sigma_obs, N_obs = build_radial_profile(
        ra, dec, kmag, RA_CS, DEC_CS,
        RADIAL_BINS, KMIN, KMAX
    )

    print("[INFO] Perfil radial observado:")
    print(" r_in  r_out   N      area_deg2   dens[1/deg2]   sigma")
    r_in = RADIAL_BINS[:-1]
    r_out = RADIAL_BINS[1:]
    area = pi * (r_out**2 - r_in**2)
    for rin, rout, N, dens, sig, a in zip(r_in, r_out, N_obs, dens_obs, sigma_obs, area):
        print(f" {rin:4.1f}  {rout:4.1f}  {N:7d}  {a:9.2f}  {dens:10.3f}  {sig:8.3f}")

    # Ajuste modelo constante (LCDM local)
    rho0_const, chi2_const, dof_const = fit_constant_density(dens_obs, sigma_obs)
    chi2_red_const = chi2_const / dof_const if dof_const > 0 else np.nan

    # Ajuste linear
    a_obs, b_obs, sa_obs, sb_obs, chi2_lin_obs, dof_lin_obs = fit_linear_density(
        r_centers, dens_obs, sigma_obs
    )
    chi2_red_lin_obs = chi2_lin_obs / dof_lin_obs if dof_lin_obs > 0 else np.nan
    sig_b_obs = b_obs / sb_obs if sb_obs > 0 else np.nan

    print("=========================================================")
    print("[RESULT] Ajuste modelo constante (baseline tipo LCDM local):")
    print(f"  rho0_const       = {rho0_const:.3f} sources/deg^2")
    print(f"  chi2_const       = {chi2_const:.3f}")
    print(f"  chi2_red_const   = {chi2_red_const:.3f}")
    print(f"  dof_const        = {dof_const:d}")
    print("=========================================================")
    print("[RESULT] Ajuste linear dens(r) = a + b r (Cold Spot real):")
    print(f"  a_obs            = {a_obs:.3f} +/- {sa_obs:.3f} sources/deg^2")
    print(f"  b_obs            = {b_obs:.3f} +/- {sb_obs:.3f} sources/deg^2/deg")
    print(f"  significancia_b  = {sig_b_obs:.2f} sigma")
    print(f"  chi2_linear      = {chi2_lin_obs:.3f}")
    print(f"  chi2_red_linear  = {chi2_red_lin_obs:.3f}")
    print(f"  dof_linear       = {dof_lin_obs:d}")
    print("=========================================================")

    # --------------------------------------
    # Monte Carlo: centros aleatórios
    # --------------------------------------
    print(f"[INFO] Iniciando Monte Carlo com N_SIM = {N_SIM} centros aleatórios...")
    slopes_sim = []
    chi2_const_sim = []
    n_fail = 0

    # Pré-cálculo: área dos anéis já definida acima
    for i in range(N_SIM):
        # Escolhe aleatoriamente um centro entre as próprias galáxias
        idx = np.random.randint(0, len(ra))
        ra0 = ra[idx]
        dec0 = dec[idx]

        # Perfil radial ao redor desse centro aleatório
        rc, dens_sim, sigma_sim, N_sim_bins = build_radial_profile(
            ra, dec, kmag, ra0, dec0,
            RADIAL_BINS, KMIN, KMAX
        )

        # Se algum bin não tiver N>0, ainda é aceitável (sigma = inf => peso zero),
        # mas se TODOS estiverem ruins, essa simulação é descartada.
        if np.all(~np.isfinite(dens_sim)) or np.all(~np.isfinite(sigma_sim)):
            n_fail += 1
            continue

        # Ajuste constante
        rho0_s, chi2_s, dof_s = fit_constant_density(dens_sim, sigma_sim)
        if dof_s <= 0 or not np.isfinite(chi2_s):
            n_fail += 1
            continue

        # Ajuste linear
        a_s, b_s, sa_s, sb_s, chi2_lin_s, dof_lin_s = fit_linear_density(
            rc, dens_sim, sigma_sim
        )
        if dof_lin_s <= 0 or not np.isfinite(b_s):
            n_fail += 1
            continue

        slopes_sim.append(b_s)
        chi2_const_sim.append(chi2_s)

        if (i+1) % 500 == 0:
            print(f"[INFO] Monte Carlo: {i+1} / {N_SIM} simulações concluídas...")

    slopes_sim = np.array(slopes_sim)
    chi2_const_sim = np.array(chi2_const_sim)
    n_ok = len(slopes_sim)

    print("=========================================================")
    print("[RESULT] Monte Carlo – resumo dos centros aleatórios:")
    print(f"  Simulações pedidas:   {N_SIM}")
    print(f"  Simulações válidas:   {n_ok}")
    print(f"  Simulações descartadas: {n_fail}")
    print("=========================================================")

    if n_ok == 0:
        print("[ERRO] Nenhuma simulação Monte Carlo válida. Verifique o catálogo ou parâmetros.")
        return

    # Estatísticas da distribuição de slopes
    mean_b = slopes_sim.mean()
    std_b = slopes_sim.std(ddof=1)

    # Estatísticas da distribuição de chi2_const
    mean_chi2 = chi2_const_sim.mean()
    std_chi2 = chi2_const_sim.std(ddof=1)

    # p-value empírico para slope: prob(b_sim >= b_obs) (teste unilateral)
    p_mc_slope = np.mean(slopes_sim >= b_obs)

    # p-value empírico para chi2_const: prob(chi2_sim >= chi2_const_obs)
    p_mc_chi2 = np.mean(chi2_const_sim >= chi2_const)

    print("[RESULT] Monte Carlo – distribuição de slopes b:")
    print(f"  mean(b_sim)      = {mean_b:.3f} sources/deg^2/deg")
    print(f"  std(b_sim)       = {std_b:.3f} sources/deg^2/deg")
    print(f"  b_obs            = {b_obs:.3f} sources/deg^2/deg")
    print(f"  p_mc_slope       = {p_mc_slope:.3e}")
    print("=========================================================")
    print("[RESULT] Monte Carlo – distribuição de chi2_const:")
    print(f"  mean(chi2_sim)   = {mean_chi2:.3f}")
    print(f"  std(chi2_sim)    = {std_chi2:.3f}")
    print(f"  chi2_const_obs   = {chi2_const:.3f}")
    print(f"  p_mc_chi2        = {p_mc_chi2:.3e}")
    print("=========================================================")

    # Salva resumo em arquivo
    outfile = "random_coldspots_2mass_results.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("Random Cold Spots test around the CMB Cold Spot (2MASS cutout)\n")
        f.write("-------------------------------------------------------------\n\n")
        f.write(f"Input catalog : {INFILE}\n")
        f.write(f"Cold Spot (RA,Dec) = ({RA_CS:.4f}, {DEC_CS:.4f}) deg\n")
        f.write(f"Radial bins (deg)  : {RADIAL_BINS.tolist()}\n")
        f.write(f"Kmag range         : [{KMIN:.1f}, {KMAX:.1f}]\n\n")

        f.write("Observed radial profile (Cold Spot):\n")
        f.write("Bin_r_in  Bin_r_out   N      area_deg2    dens[1/deg2]   sigma\n")
        for rin, rout, N, dens, sig, a in zip(r_in, r_out, N_obs, dens_obs, sigma_obs, area):
            f.write(f"{rin:6.2f}  {rout:6.2f}  {N:7d}  {a:9.2f}  {dens:12.3f}  {sig:8.3f}\n")
        f.write("\n")
        f.write("Constant-density fit (LCDM-like baseline):\n")
        f.write(f"  rho0_const     = {rho0_const:.3f} sources/deg^2\n")
        f.write(f"  chi2_const     = {chi2_const:.3f}\n")
        f.write(f"  chi2_red_const = {chi2_red_const:.3f}\n")
        f.write(f"  dof_const      = {dof_const:d}\n\n")

        f.write("Linear fit dens(r) = a + b r (Cold Spot):\n")
        f.write(f"  a_obs          = {a_obs:.3f} +/- {sa_obs:.3f}\n")
        f.write(f"  b_obs          = {b_obs:.3f} +/- {sb_obs:.3f}\n")
        f.write(f"  significance_b = {sig_b_obs:.2f} sigma\n")
        f.write(f"  chi2_linear    = {chi2_lin_obs:.3f}\n")
        f.write(f"  chi2_red_lin   = {chi2_red_lin_obs:.3f}\n")
        f.write(f"  dof_linear     = {dof_lin_obs:d}\n\n")

        f.write("Monte Carlo summary (random centers within the same 2MASS cutout):\n")
        f.write(f"  N_SIM requested = {N_SIM}\n")
        f.write(f"  N_SIM valid     = {n_ok}\n")
        f.write(f"  N_SIM discarded = {n_fail}\n\n")

        f.write("Monte Carlo distribution of slopes b:\n")
        f.write(f"  mean(b_sim)    = {mean_b:.3f}\n")
        f.write(f"  std(b_sim)     = {std_b:.3f}\n")
        f.write(f"  b_obs          = {b_obs:.3f}\n")
        f.write(f"  p_mc_slope     = {p_mc_slope:.3e}\n\n")

        f.write("Monte Carlo distribution of chi2_const:\n")
        f.write(f"  mean(chi2_sim) = {mean_chi2:.3f}\n")
        f.write(f"  std(chi2_sim)  = {std_chi2:.3f}\n")
        f.write(f"  chi2_const_obs = {chi2_const:.3f}\n")
        f.write(f"  p_mc_chi2      = {p_mc_chi2:.3e}\n")

    print(f"[INFO] Resultados salvos em: {outfile}")


if __name__ == "__main__":
    main()