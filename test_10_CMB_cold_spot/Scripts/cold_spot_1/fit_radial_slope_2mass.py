#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
fit_radial_slope_2mass.py

Ajusta modelos de densidade radial em torno do CMB Cold Spot
usando o catálogo 2MASS recortado (2mass.tsv).

- Usa o centro do Cold Spot em coordenadas galácticas (l,b) = (209, -57) deg,
  convertido para equatorial (RA,Dec) ≈ (48.2999, -20.4373) deg.
- Constrói perfil radial em bins: [0–2], [2–4], [4–6], [6–8], [8–10] deg.
- Calcula densidade de fontes e erro (Poisson: sqrt(N)/area).
- Ajusta um modelo linear: dens(r) = rho0 + a * r
  para:
    (i) todas as fontes,
    (ii) bright   (8 <= K < 12),
    (iii) mid     (12 <= K < 14),
    (iv) faint    (14 <= K < 16).
- Imprime resultados no terminal e salva em '2mass_radial_slope_results.txt'.
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.stats import chi2

# -------------------------------
# Configurações básicas
# -------------------------------

INFILE = "2mass.tsv"   # catálogo recortado do Vizier
OUTTXT = "2mass_radial_slope_results.txt"

# Centro do Cold Spot (galáctico) e conversão para equatorial
l_cs = 209.0
b_cs = -57.0

cs_gal = SkyCoord(l=l_cs * u.deg, b=b_cs * u.deg, frame="galactic")
cs_eq = cs_gal.icrs
RA_CS = cs_eq.ra.deg
DEC_CS = cs_eq.dec.deg

# Bins radiais em graus
RADIAL_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# Faixas de magnitude em K
MAG_BINS = {
    "all":  (None, None),
    "bright": (8.0, 12.0),
    "mid":    (12.0, 14.0),
    "faint":  (14.0, 16.0),
}

# -------------------------------
# Funções auxiliares
# -------------------------------

def big_sep(a, ch="="):
    print(ch * a)


def compute_theta_deg(ra_deg, dec_deg, ra_cs, dec_cs):
    """
    Calcula a separação angular em graus entre (ra,dec) e (ra_cs,dec_cs)
    usando SkyCoord (mais robusto do que aproximação de pequeno ângulo).
    """
    coords = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    cs = SkyCoord(ra=ra_cs * u.deg, dec=dec_cs * u.deg, frame="icrs")
    sep = coords.separation(cs).deg
    return sep


def radial_profile(theta_deg, mask_mag, radial_bins):
    """
    Constrói perfil radial em bins dados por radial_bins.
    Retorna:
      r_in, r_out, r_center, N, area_deg2, dens, sigma.
    """
    r_in = radial_bins[:-1]
    r_out = radial_bins[1:]
    r_center = 0.5 * (r_in + r_out)

    N = []
    area_deg2 = []
    dens = []
    sigma = []

    for rin, rout in zip(r_in, r_out):
        m = (theta_deg >= rin) & (theta_deg < rout) & mask_mag
        n_bin = np.sum(m)
        N.append(n_bin)

        # área anular em deg^2: pi (r_out^2 - r_in^2)
        area = np.pi * (rout**2 - rin**2)
        area_deg2.append(area)

        if n_bin > 0:
            d = n_bin / area
            s = np.sqrt(n_bin) / area
        else:
            d = 0.0
            s = np.inf

        dens.append(d)
        sigma.append(s)

    return (r_in, r_out, r_center,
            np.array(N), np.array(area_deg2),
            np.array(dens), np.array(sigma))


def weighted_linear_fit(x, y, yerr):
    """
    Ajuste linear ponderado y = a + b x,
    pesos w = 1 / yerr^2.

    Retorna:
      a, b, sigma_a, sigma_b, chi2_val, chi2_red, dof, p_value
    """
    # mascara para bins com erro finito
    m = np.isfinite(yerr) & (yerr > 0)
    x = x[m]
    y = y[m]
    yerr = yerr[m]

    if len(x) < 2:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    w = 1.0 / (yerr**2)
    W = np.sum(w)
    Wx = np.sum(w * x)
    Wy = np.sum(w * y)
    Wxx = np.sum(w * x * x)
    Wxy = np.sum(w * x * y)

    denom = (W * Wxx - Wx**2)
    if denom == 0:
        return (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0, np.nan)

    # coeficientes
    a = (Wxx * Wy - Wx * Wxy) / denom   # intercepto
    b = (W * Wxy - Wx * Wy) / denom    # slope

    # incertezas
    sigma_a = np.sqrt(Wxx / denom)
    sigma_b = np.sqrt(W / denom)

    # chi^2
    y_model = a + b * x
    chi2_val = np.sum(((y - y_model) / yerr)**2)
    dof = len(x) - 2
    chi2_red = chi2_val / dof if dof > 0 else np.nan
    p_value = 1.0 - chi2.cdf(chi2_val, dof) if dof > 0 else np.nan

    return a, b, sigma_a, sigma_b, chi2_val, chi2_red, dof, p_value


# -------------------------------
# Programa principal
# -------------------------------

def main():
    big_sep(57, "=")
    print(f"[INFO] Lendo catálogo 2MASS recortado: {INFILE}")
    # Muitas colunas têm tipos mistos, então deixamos o pandas escolher
    # e limpamos depois só as que importam.
    df = pd.read_csv(INFILE, sep="\t", comment="#", low_memory=False)
    print(f"[INFO] Linhas totais no arquivo: {len(df)}")

    # Garantir que RA/Dec são colunas numéricas em float
    for col in ["RAJ2000", "DEJ2000"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Eventual magnitude K
    if "Kmag" in df.columns:
        df["Kmag"] = pd.to_numeric(df["Kmag"], errors="coerce")
    else:
        df["Kmag"] = np.nan

    m_valid = df["RAJ2000"].notnull() & df["DEJ2000"].notnull()
    df = df[m_valid].copy()
    print(f"[INFO] Linhas com RA/Dec válidos após limpeza: {len(df)}")

    ra = df["RAJ2000"].values
    dec = df["DEJ2000"].values
    k_mag = df["Kmag"].values

    big_sep(57, "=")
    print("[INFO] Calculando distâncias angulares até o Cold Spot...")
    theta_deg = compute_theta_deg(ra, dec, RA_CS, DEC_CS)
    print(f"[INFO] theta_deg min/max: {theta_deg.min():.2f} – {theta_deg.max():.2f} deg")

    big_sep(57, "=")
    print("[RESULTADOS] Ajustes de modelo linear dens(r) = rho0 + slope * r")
    print(f"  Centro Cold Spot (galático): (l,b) = ({l_cs}, {b_cs}) deg")
    print(f"  Centro Cold Spot (ICRS):     (RA,Dec) = ({RA_CS:.4f}, {DEC_CS:.4f}) deg")
    print(f"  Radial bins: {RADIAL_BINS} deg")
    big_sep(57, "=")

    # Abrir arquivo de saída
    with open(OUTTXT, "w", encoding="utf-8") as f:
        f.write("Radial slope fits for 2MASS around CMB Cold Spot\n")
        f.write(f"Cold Spot (l,b) = ({l_cs}, {b_cs}) deg\n")
        f.write(f"Cold Spot (RA,Dec) = ({RA_CS:.6f}, {DEC_CS:.6f}) deg\n")
        f.write(f"Radial bins (deg): {RADIAL_BINS.tolist()}\n")
        f.write("\n")
        f.write("Model: dens(r) = rho0 + slope * r\n")
        f.write("Units: dens in sources/deg^2; r in deg\n\n")

        for label, (kmin, kmax) in MAG_BINS.items():
            big_sep(57, "-")
            print(f"[FAIXA] {label}")
            f.write(f"=== faixa: {label} ===\n")

            # Máscara de magnitude
            if (kmin is None) and (kmax is None):
                mask_mag = np.isfinite(k_mag)  # qualquer valor válido
                mag_info = "all K"
            else:
                mask_mag = np.isfinite(k_mag)
                mask_mag &= (k_mag >= kmin) & (k_mag < kmax)
                mag_info = f"{kmin} <= K < {kmax}"

            print(f"  Mag cut: {mag_info}")
            f.write(f"  Mag cut: {mag_info}\n")

            # Perfil radial
            (r_in, r_out, r_center,
             N, area, dens, sigma) = radial_profile(theta_deg, mask_mag, RADIAL_BINS)

            print("  Bin_r_in  Bin_r_out   N     area_deg2    dens[1/deg2]   sigma")
            f.write("  Bin_r_in  Bin_r_out   N     area_deg2    dens[1/deg2]   sigma\n")
            for i in range(len(r_in)):
                print(f"   {r_in[i]:4.1f}    {r_out[i]:4.1f}  {N[i]:7d}  {area[i]:9.2f}  {dens[i]:11.3f}  {sigma[i]:8.3f}")
                f.write(f"   {r_in[i]:4.1f}    {r_out[i]:4.1f}  {N[i]:7d}  {area[i]:9.2f}  {dens[i]:11.3f}  {sigma[i]:8.3f}\n")

            # Ajuste linear
            a, b, sigma_a, sigma_b, chi2_val, chi2_red, dof, p_value = weighted_linear_fit(
                r_center, dens, sigma
            )

            print("\n  [Ajuste dens(r) = rho0 + slope * r]")
            print(f"    rho0   = {a: .3f} ± {sigma_a: .3f} sources/deg²")
            print(f"    slope  = {b: .3f} ± {sigma_b: .3f} sources/deg²/deg")
            print(f"    chi2   = {chi2_val: .3f}")
            print(f"    dof    = {dof}")
            print(f"    chi2_red = {chi2_red: .3f}")
            print(f"    p-value  = {p_value: .3e}\n")

            f.write("\n  [Fit dens(r) = rho0 + slope * r]\n")
            f.write(f"    rho0   = {a: .3f} ± {sigma_a: .3f} sources/deg^2\n")
            f.write(f"    slope  = {b: .3f} ± {sigma_b: .3f} sources/deg^2/deg\n")
            f.write(f"    chi2   = {chi2_val: .3f}\n")
            f.write(f"    dof    = {dof}\n")
            f.write(f"    chi2_red = {chi2_red: .3f}\n")
            f.write(f"    p-value  = {p_value: .3e}\n\n")

        big_sep(57, "=")
        print(f"[OK] Resultados completos salvos em: {OUTTXT}")
        big_sep(57, "=")


if __name__ == "__main__":
    main()