#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

# ---------------------------------------------------------
# Configurações básicas
# ---------------------------------------------------------

INFILE = "2mass.tsv"  # catálogo recortado que você já está usando
OUTFILE = "regressao_2mass_lcdm_results.txt"

# Centro do Cold Spot em coordenadas galácticas
L_CS = 209.0  # graus
B_CS = -57.0  # graus

# Bins radiais em graus (0–10 deg, como nos testes anteriores)
RADIAL_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Cortes em magnitude K
KMIN = 8.0
KMAX = 16.0


# ---------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------

def load_2mass_subset(infile):
    """
    Lê o catálogo 2MASS recortado (2mass.tsv) e retorna arrays
    limpos de RA, Dec e Kmag.
    """

    print("=========================================================")
    print(f"[INFO] Lendo catálogo 2MASS recortado: {infile}")

    df = pd.read_csv(infile, sep="\t", comment="#")
    print(f"[INFO] Linhas totais no arquivo: {len(df)}")

    # Limpeza de RA/Dec
    for col in ["RAJ2000", "DEJ2000"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["RAJ2000", "DEJ2000"])

    # Kmag pode ter valores ausentes ou "null"
    if "Kmag" in df.columns:
        df["Kmag"] = pd.to_numeric(df["Kmag"], errors="coerce")
    else:
        raise RuntimeError("Coluna 'Kmag' não encontrada em 2mass.tsv.")

    df = df.dropna(subset=["Kmag"])

    print(f"[INFO] Linhas com RA/Dec e Kmag válidos: {len(df)}")

    ra = df["RAJ2000"].values.astype(float)
    dec = df["DEJ2000"].values.astype(float)
    kmag = df["Kmag"].values.astype(float)

    return ra, dec, kmag


def compute_theta_deg(ra, dec, l_cs_deg, b_cs_deg):
    """
    Calcula a distância angular (em graus) de cada objeto até o Cold Spot,
    dado em coordenadas galácticas (l_cs, b_cs).
    """
    # Centro do CS em galácticas
    cs_gal = SkyCoord(l=l_cs_deg * u.deg, b=b_cs_deg * u.deg, frame="galactic")
    # Converter para equatorial (ICRS)
    cs_icrs = cs_gal.icrs

    # Catálogo em ICRS
    cat = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # Distância angular
    theta = cat.separation(cs_icrs).deg
    return theta


def radial_profile(theta_deg, k_mag, radial_bins, kmin, kmax):
    """
    Monta perfil radial de contagens e densidade angular
    para objetos com Kmag em [kmin, kmax].
    Retorna:
      r_mid  : centro de cada anel (deg)
      density: densidade (sources/deg^2)
      sigma  : erro na densidade (Poisson)
      counts : contagens em cada anel
      areas  : área de cada anel (deg^2)
    """
    mask_mag = (k_mag >= kmin) & (k_mag < kmax)
    theta_sel = theta_deg[mask_mag]

    counts = []
    density = []
    sigma = []
    areas = []
    r_mid = []

    for r_in, r_out in zip(radial_bins[:-1], radial_bins[1:]):
        mask_bin = (theta_sel >= r_in) & (theta_sel < r_out)
        N = np.sum(mask_bin)
        area = np.pi * (r_out**2 - r_in**2)  # área em deg^2

        if N > 0 and area > 0:
            dens = N / area
            err = np.sqrt(N) / area
        else:
            dens = 0.0
            err = np.inf

        counts.append(N)
        density.append(dens)
        sigma.append(err)
        areas.append(area)
        r_mid.append(0.5 * (r_in + r_out))

    return np.array(r_mid), np.array(density), np.array(sigma), np.array(counts), np.array(areas)


def fit_constant_model(density, sigma):
    """
    Ajuste de modelo de densidade constante (baseline tipo LCDM)
    usando média ponderada pelas incertezas sigma.
    Retorna:
      rho0      : densidade constante ajustada
      chi2_val  : chi2 do ajuste
      dof       : graus de liberdade
      p_value   : p-valor
    """
    w = 1.0 / sigma**2
    rho0 = np.sum(w * density) / np.sum(w)

    chi2_val = np.sum(((density - rho0) / sigma)**2)
    dof = len(density) - 1
    p = chi2.sf(chi2_val, dof)

    return rho0, chi2_val, dof, p


def regressao_linear(r, density, sigma):
    """
    Regressão linear ponderada: density(r) = a + b * r
    Retorna:
      a, b         : intercepto e inclinação
      sigma_a,b    : incertezas dos parâmetros
      chi2_val     : chi2 do ajuste
      dof          : graus de liberdade
      p_value      : p-valor
    """
    w = 1.0 / sigma**2
    Sw = np.sum(w)
    Swr = np.sum(w * r)
    Swr2 = np.sum(w * r * r)
    Swd = np.sum(w * density)
    Swrd = np.sum(w * r * density)

    Delta = Sw * Swr2 - Swr**2

    a = (Swr2 * Swd - Swr * Swrd) / Delta
    b = (Sw * Swrd - Swr * Swd) / Delta

    sigma_a = np.sqrt(Swr2 / Delta)
    sigma_b = np.sqrt(Sw / Delta)

    model = a + b * r
    chi2_val = np.sum(((density - model) / sigma)**2)
    dof = len(r) - 2
    p = chi2.sf(chi2_val, dof)

    return a, b, sigma_a, sigma_b, chi2_val, dof, p


# ---------------------------------------------------------
# Script principal
# ---------------------------------------------------------

def main():
    ra, dec, kmag = load_2mass_subset(INFILE)

    # Distância angular até o Cold Spot
    theta_deg = compute_theta_deg(ra, dec, L_CS, B_CS)
    print(f"[INFO] theta_deg min/max: {theta_deg.min():.2f} – {theta_deg.max():.2f} deg")
    print("=========================================================")

    # Perfil radial para todo o intervalo KMIN–KMAX
    r_mid, density, sigma, counts, areas = radial_profile(
        theta_deg, kmag, RADIAL_BINS, KMIN, KMAX
    )

    print("[INFO] Perfil radial bruto (K in [{:.1f}, {:.1f}]):".format(KMIN, KMAX))
    print(" r_in  r_out   N      area_deg2   dens[1/deg2]   sigma")
    for i in range(len(r_mid)):
        r_in = RADIAL_BINS[i]
        r_out = RADIAL_BINS[i+1]
        print(f" {r_in:4.1f}  {r_out:4.1f}  {counts[i]:6d}  {areas[i: i+1][0]:10.2f}  {density[i]:10.3f}  {sigma[i]:8.3f}")

    # Ajuste do modelo constante (baseline tipo LCDM)
    rho0, chi2_const, dof_const, p_const = fit_constant_model(density, sigma)
    print("=========================================================")
    print("[RESULT] Ajuste modelo constante (baseline tipo LCDM):")
    print(f"  rho0_const       = {rho0:.3f} sources/deg^2")
    print(f"  chi2_const       = {chi2_const:.3f}")
    print(f"  chi2_red_const   = {chi2_const/dof_const:.3f}")
    print(f"  dof_const        = {dof_const}")
    print(f"  p_value_const    = {p_const:.3e}")

    # Ajuste linear (possível assinatura dissipativa tipo CET)
    a, b, sigma_a, sigma_b, chi2_lin, dof_lin, p_lin = regressao_linear(
        r_mid, density, sigma
    )

    print("=========================================================")
    print("[RESULT] Ajuste linear density(r) = a + b * r:")
    print(f"  a (intercepto)   = {a:.3f} ± {sigma_a:.3f} sources/deg^2")
    print(f"  b (inclinação)   = {b:.3f} ± {sigma_b:.3f} sources/deg^2/deg")
    print(f"  significancia b  = {b/sigma_b:.2f} sigma")
    print(f"  chi2_linear      = {chi2_lin:.3f}")
    print(f"  chi2_red_linear  = {chi2_lin/dof_lin:.3f}")
    print(f"  dof_linear       = {dof_lin}")
    print(f"  p_value_linear   = {p_lin:.3e}")

    # Comparação de modelos: constante vs linear
    # (não estritamente aninhados, mas dá para olhar a diferença de chi2)
    delta_chi2 = chi2_const - chi2_lin
    delta_dof = dof_const - dof_lin  # deve ser 1
    if delta_dof > 0:
        p_delta = chi2.sf(delta_chi2, delta_dof)
    else:
        p_delta = np.nan

    print("=========================================================")
    print("[RESULT] Comparação modelos (constante vs linear):")
    print(f"  delta_chi2       = {delta_chi2:.3f}")
    print(f"  delta_dof        = {delta_dof}")
    print(f"  p_value_delta    = {p_delta:.3e}")
    print("=========================================================")

    # -----------------------------------------------------
    # Salvar em arquivo de resultados (ASCII puro)
    # -----------------------------------------------------
    with open(OUTFILE, "w", encoding="utf-8") as f:
        f.write("2MASS radial density around CMB Cold Spot\n")
        f.write("Input catalog : {}\n".format(INFILE))
        f.write("Cold Spot (l,b) = ({:.1f}, {:.1f}) deg\n".format(L_CS, B_CS))
        f.write("Radial bins (deg): {}\n".format(list(RADIAL_BINS)))
        f.write("Kmag range used : [{:.1f}, {:.1f}]\n".format(KMIN, KMAX))
        f.write("\n")
        f.write("Radial profile (all K in range):\n")
        f.write("Bin_r_in  Bin_r_out   N      area_deg2    dens[1/deg2]   sigma\n")
        for i in range(len(r_mid)):
            r_in = RADIAL_BINS[i]
            r_out = RADIAL_BINS[i+1]
            f.write(
                f"{r_in:6.2f}  {r_out:6.2f}  {counts[i]:7d}  {areas[i]:12.3f}  {density[i]:12.3f}  {sigma[i]:8.3f}\n"
            )

        f.write("\n")
        f.write("Constant density model (LCDM-like baseline):\n")
        f.write("  rho0_const       = {:.3f} sources/deg^2\n".format(rho0))
        f.write("  chi2_const       = {:.3f}\n".format(chi2_const))
        f.write("  chi2_red_const   = {:.3f}\n".format(chi2_const/dof_const))
        f.write("  dof_const        = {}\n".format(dof_const))
        f.write("  p_value_const    = {:.3e}\n".format(p_const))

        f.write("\n")
        f.write("Linear model: density(r) = a + b * r\n")
        f.write("  a (intercepto)   = {:.3f} +/- {:.3f} sources/deg^2\n".format(a, sigma_a))
        f.write("  b (inclinação)   = {:.3f} +/- {:.3f} sources/deg^2/deg\n".format(b, sigma_b))
        f.write("  significancia_b  = {:.2f} sigma\n".format(b/sigma_b))
        f.write("  chi2_linear      = {:.3f}\n".format(chi2_lin))
        f.write("  chi2_red_linear  = {:.3f}\n".format(chi2_lin/dof_lin))
        f.write("  dof_linear       = {}\n".format(dof_lin))
        f.write("  p_value_linear   = {:.3e}\n".format(p_lin))

        f.write("\n")
        f.write("Model comparison (constant vs linear):\n")
        f.write("  delta_chi2       = {:.3f}\n".format(delta_chi2))
        f.write("  delta_dof        = {}\n".format(delta_dof))
        f.write("  p_value_delta    = {:.3e}\n".format(p_delta))

    print(f"[INFO] Resultados gravados em: {OUTFILE}")


if __name__ == "__main__":
    main()