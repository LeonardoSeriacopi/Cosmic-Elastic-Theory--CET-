#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Testa se o déficit/excesso de fontes ao redor do Cold Spot
depende da magnitude K (2MASS).

Entrada:
    - 2mass.tsv  (catalogo recortado que voce ja baixou do VizieR)

Saída:
    - mag_stratified_2mass_results.txt
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

# ------------------------------------------------------------
# Parâmetros principais
# ------------------------------------------------------------

INFILE = "2mass.tsv"

# Centro do Cold Spot (coordenadas galácticas, as mesmas que usamos antes)
L_CS = 209.0   # deg
B_CS = -57.0   # deg

# Bins radiais (em graus)
RADIAL_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Faixas de magnitude K para estratificação
MAG_BINS = [
    (8.0, 12.0, "bright"),
    (12.0, 14.0, "mid"),
    (14.0, 16.0, "faint"),
]

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------

def load_2mass_catalog(infile):
    """
    Lê o catálogo 2MASS recortado (TSV) e retorna um DataFrame
    apenas com colunas limpas de RA, Dec, Kmag.
    """
    print("=========================================================")
    print(f"[INFO] Lendo catálogo 2MASS recortado: {infile}")
    df = pd.read_csv(infile, sep="\t", comment="#")
    print(f"[INFO] Linhas totais no arquivo: {len(df)}")

    # Limpeza de RA/Dec
    for col in ["RAJ2000", "DEJ2000"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Kmag
    if "Kmag" not in df.columns:
        raise RuntimeError("Coluna 'Kmag' nao encontrada no 2MASS TSV.")

    df["Kmag"] = pd.to_numeric(df["Kmag"], errors="coerce")

    mask_valid = df["RAJ2000"].notna() & df["DEJ2000"].notna() & df["Kmag"].notna()
    df = df.loc[mask_valid].copy()

    print(f"[INFO] Linhas com RA/Dec/Kmag validos apos limpeza: {len(df)}")
    return df


def compute_theta_deg(df, l_cs, b_cs):
    """
    Calcula a distancia angular (em graus) de cada fonte
    ao centro do Cold Spot, dado em (l,b) galactico.
    """
    # Centro do Cold Spot em galactico
    cs_gal = SkyCoord(l=l_cs * u.deg, b=b_cs * u.deg, frame="galactic")
    # Catalogo em equatorial
    cat = SkyCoord(ra=df["RAJ2000"].values * u.deg,
                   dec=df["DEJ2000"].values * u.deg,
                   frame="icrs")
    sep = cs_gal.icrs.separation(cat)  # graus
    theta_deg = sep.deg
    print(f"[INFO] theta_deg min/max: {theta_deg.min():.2f} – {theta_deg.max():.2f}")
    return theta_deg


def radial_profile(theta_deg, r_bins):
    """
    Calcula o perfil radial de contagens:
    - r_bins: array de limites de anéis (ex: [0,2,4,...])
    
    Retorna:
    - r_in, r_out, N, area_deg2, density, sigma_density
    """
    counts = []
    areas = []
    densities = []
    errors = []

    for i in range(len(r_bins) - 1):
        r_in = r_bins[i]
        r_out = r_bins[i + 1]
        # mascara bin
        mask = (theta_deg >= r_in) & (theta_deg < r_out)
        N = np.sum(mask)
        # area do anel
        area = np.pi * (r_out**2 - r_in**2)  # deg^2
        dens = N / area
        # Poisson: sigma_N = sqrt(N), sigma_dens = sigma_N / area
        sigma = np.sqrt(N) / area if N > 0 else np.inf

        counts.append(N)
        areas.append(area)
        densities.append(dens)
        errors.append(sigma)

    return (
        np.array(r_bins[:-1]),
        np.array(r_bins[1:]),
        np.array(counts),
        np.array(areas),
        np.array(densities),
        np.array(errors),
    )


def fit_constant_density(densities, errors):
    """
    Ajuste de uma densidade constante usando mínimos quadrados
    ponderados por 1/sigma^2 (equivalente ao MLE Poisson aproximado).
    """
    # Pesos
    w = 1.0 / (errors**2)
    # Média ponderada
    rho0 = np.sum(w * densities) / np.sum(w)
    return rho0


def chi2_constant_model(densities, errors, rho0):
    """
    Calcula o chi^2 para o modelo de densidade constante rho0.
    """
    chi2_val = np.sum(((densities - rho0) / errors)**2)
    dof = len(densities) - 1  # um parâmetro (rho0)
    chi2_red = chi2_val / dof if dof > 0 else np.nan
    pval = 1.0 - chi2.cdf(chi2_val, dof)
    return chi2_val, chi2_red, dof, pval


# ------------------------------------------------------------
# Script principal
# ------------------------------------------------------------

def main():
    # 1) Ler catálogo
    df = load_2mass_catalog(INFILE)

    # 2) Calcular theta_deg para todas as fontes
    theta_deg = compute_theta_deg(df, L_CS, B_CS)

    # 3) Abrir arquivo de saída
    outname = "mag_stratified_2mass_results.txt"
    f = open(outname, "w", encoding="utf-8")

    f.write("Mag-stratified radial density test around the CMB Cold Spot\n")
    f.write(f"Input catalog : {INFILE}\n")
    f.write(f"Cold Spot (l,b) = ({L_CS}, {B_CS}) deg\n")
    f.write(f"Radial bins (deg): {RADIAL_BINS.tolist()}\n")
    f.write("Magnitude bins (Kmag):\n")
    for mmin, mmax, label in MAG_BINS:
        f.write(f"  {label}: {mmin} <= K < {mmax}\n")
    f.write("\n")

    print("=========================================================")
    print("[INFO] Iniciando testes por faixa de magnitude K")
    print("=========================================================")

    for mmin, mmax, label in MAG_BINS:
        print("---------------------------------------------------------")
        print(f"[INFO] Faixa de magnitude: {label}  ({mmin} <= K < {mmax})")
        mask_mag = (df["Kmag"] >= mmin) & (df["Kmag"] < mmax)

        if np.sum(mask_mag) == 0:
            print("[AVISO] Nenhuma fonte nessa faixa de magnitude; pulando.")
            f.write(f"[{label}] Nenhuma fonte nessa faixa de magnitude; pulando.\n\n")
            continue

        theta_sub = theta_deg[mask_mag]

        (r_in, r_out, N, area, dens, sigma) = radial_profile(theta_sub, RADIAL_BINS)

        print("[INFO] Perfil radial:")
        f.write(f"=== Faixa {label}: {mmin} <= K < {mmax} ===\n")
        f.write("Bin_r_in  Bin_r_out   N      area_deg2    dens[1/deg2]   sigma\n")

        for i in range(len(r_in)):
            print(
                f"Bin {r_in[i]:.1f}-{r_out[i]:.1f} deg | "
                f"N={N[i]:6d} | dens={dens[i]:8.2f} | sigma={sigma[i]:7.2f}"
            )
            f.write(
                f"{r_in[i]:6.2f}  {r_out[i]:6.2f}  {N[i]:7d}  "
                f"{area[i]:10.2f}  {dens[i]:12.3f}  {sigma[i]:10.3f}\n"
            )

        # Ajuste de densidade constante
        rho0 = fit_constant_density(dens, sigma)
        chi2_val, chi2_red, dof, pval = chi2_constant_model(dens, sigma, rho0)

        print(f"[RESULT] Faixa {label} - densidade constante ajustada = {rho0:.2f} fontes/deg^2")
        print(f"         chi2 = {chi2_val:.3f}, chi2_red = {chi2_red:.3f}, dof = {dof}, p-value = {pval:.3e}")

        f.write("\n[Constant density fit]\n")
        f.write(f"rho0 = {rho0:.3f} sources/deg^2\n")
        f.write(f"chi2 = {chi2_val:.3f}\n")
        f.write(f"chi2_red = {chi2_red:.3f}\n")
        f.write(f"dof = {dof}\n")
        f.write(f"p_value = {pval:.3e}\n\n")

    f.close()
    print("=========================================================")
    print(f"[OK] Resultados salvos em: {outname}")
    print("=========================================================")


if __name__ == "__main__":
    main()