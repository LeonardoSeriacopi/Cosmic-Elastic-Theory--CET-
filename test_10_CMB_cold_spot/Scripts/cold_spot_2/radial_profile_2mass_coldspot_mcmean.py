#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
radial_profile_2mass_coldspot_mcmean.py

Constrói:
  - perfil radial observado em torno do (segundo) CMB Cold Spot;
  - perfil radial MÉDIO em torno de muitos centros aleatórios
    (dentro da mesma área do recorte 2MASS).

Usa o catálogo: 2mass.tsv
Formato: TSV (tab-separated), com colunas RAJ2000, DEJ2000, Kmag.

Saídas:
  - radial_profile_coldspot_vs_random.txt   (tabela resumo)
"""

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u

# -----------------------------
# 1. Parâmetros principais
# -----------------------------

infile = "2mass.tsv"   # catálogo 2MASS recortado em torno do Cold Spot

# Centro do SEGUNDO Cold Spot:
#   galáctico: l = 208.4349 deg, b = -55.8545 deg
#   equatorial (ICRS, J2000): RA ~ 49.7550 deg, Dec ~ -18.3500 deg
RA_CS_DEG  = 49.7550
DEC_CS_DEG = -18.3500

# Bins radiais (em graus)
RADIAL_BINS = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0])  # 5 anéis

# Corte em magnitude K (mesmo usado nos outros testes)
KMIN = 8.0
KMAX = 16.0

# Número de centros aleatórios para o perfil médio
N_MC = 1000  # se ficar pesado, pode reduzir para 500


# -----------------------------
# 2. Funções auxiliares
# -----------------------------

def build_radial_profile(theta_deg, k_mag, radial_bins, kmin=None, kmax=None):
    """
    Constrói perfil radial (contagens, densidade, erro Poisson)
    usando distâncias angulares theta_deg (em graus) e magnitudes k_mag.

    Retorna:
      - r_in, r_out: limites internos/externos (arrays)
      - N: contagens por anel
      - area_deg2: área de cada anel em deg^2 (aprox. plana)
      - dens: densidades (N/area)
      - sigma: incerteza Poisson (sqrt(N)/area)
    """
    theta = np.array(theta_deg)
    kmag  = np.array(k_mag)

    mask_mag = np.isfinite(kmag)
    if kmin is not None:
        mask_mag &= (kmag >= kmin)
    if kmax is not None:
        mask_mag &= (kmag <  kmax)

    theta_sel = theta[mask_mag]

    H, edges = np.histogram(theta_sel, bins=radial_bins)
    N = H.astype(float)

    r_in  = edges[:-1]
    r_out = edges[1:]
    area_deg2 = np.pi * (r_out**2 - r_in**2)

    dens  = N / area_deg2
    sigma = np.sqrt(N) / area_deg2

    return r_in, r_out, N, area_deg2, dens, sigma


def weighted_linear_fit(r, dens, sigma):
    """
    Ajuste linear dens(r) = a + b*r com pesos 1/sigma^2.
    Usa apenas bins com sigma > 0.
    Retorna:
      a, b, sigma_a, sigma_b
    """
    r   = np.array(r, dtype=float)
    y   = np.array(dens, dtype=float)
    err = np.array(sigma, dtype=float)

    mask = (err > 0) & np.isfinite(y) & np.isfinite(r)
    r   = r[mask]
    y   = y[mask]
    err = err[mask]

    if len(r) < 2:
        return np.nan, np.nan, np.nan, np.nan

    w   = 1.0 / (err**2)
    Sw  = np.sum(w)
    Swr = np.sum(w * r)
    Swy = np.sum(w * y)
    Swr2 = np.sum(w * r * r)
    Swry = np.sum(w * r * y)

    Delta = Sw * Swr2 - Swr**2
    if Delta == 0:
        return np.nan, np.nan, np.nan, np.nan

    a = (Swr2 * Swy - Swr * Swry) / Delta
    b = (Sw  * Swry - Swr * Swy ) / Delta

    sigma_a = np.sqrt(Swr2 / Delta)
    sigma_b = np.sqrt(Sw   / Delta)

    return a, b, sigma_a, sigma_b


# -----------------------------
# 3. Programa principal
# -----------------------------

def main():
    print("=========================================================")
    print(f"[INFO] Lendo catálogo 2MASS recortado: {infile}")

    # Lê o TSV, ignorando linhas de comentário
    df = pd.read_csv(infile, sep="\t", comment="#")

    # Garante RA,Dec,Kmag numéricos
    for col in ["RAJ2000", "DEJ2000", "Kmag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    mask_valid = (
        df["RAJ2000"].notna() &
        df["DEJ2000"].notna() &
        df["Kmag"].notna()
    )

    df = df.loc[mask_valid].copy()
    print(f"[INFO] Linhas com RA/Dec e Kmag válidos: {len(df)}")

    ra   = df["RAJ2000"].values
    dec  = df["DEJ2000"].values
    kmag = df["Kmag"].values

    print("[INFO] RA min/max:  {:.3f} – {:.3f} deg".format(np.min(ra), np.max(ra)))
    print("[INFO] Dec min/max: {:.3f} – {:.3f} deg".format(np.min(dec), np.max(dec)))

    # SkyCoord de todas as fontes
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

    # -------------------------------------------------------
    # 3.1 Perfil radial observado em torno do Cold Spot 2
    # -------------------------------------------------------
    print("=========================================================")
    print("[INFO] Construindo perfil radial OBSERVADO (Cold Spot 2 real)...")

    cs_center = SkyCoord(ra=RA_CS_DEG * u.deg, dec=DEC_CS_DEG * u.deg, frame="icrs")
    theta_cs = coords.separation(cs_center).deg

    r_in, r_out, N_obs, area_deg2, dens_obs, sigma_obs = build_radial_profile(
        theta_cs, kmag, RADIAL_BINS, kmin=KMIN, kmax=KMAX
    )

    r_mid = 0.5 * (r_in + r_out)

    print("[INFO] Perfil radial observado:")
    print(" r_in  r_out   N      area_deg2   dens[1/deg2]   sigma")
    for i in range(len(r_in)):
        print(f" {r_in[i]:4.1f}  {r_out[i]:4.1f}  {int(N_obs[i]):7d}  "
              f"{area_deg2[i]:10.2f}  {dens_obs[i]:10.3f}  {sigma_obs[i]:8.3f}")

    # Ajuste linear no observado
    a_obs, b_obs, sa_obs, sb_obs = weighted_linear_fit(r_mid, dens_obs, sigma_obs)
    print("---------------------------------------------------------")
    print("[RESULT] Ajuste linear dens_obs(r) = a_obs + b_obs * r:")
    print(f"  a_obs = {a_obs:.3f} ± {sa_obs:.3f}  sources/deg^2")
    print(f"  b_obs = {b_obs:.3f} ± {sb_obs:.3f}  sources/deg^2/deg")

    # -------------------------------------------------------
    # 3.2 Monte Carlo: perfil médio de centros aleatórios
    # -------------------------------------------------------
    print("=========================================================")
    print(f"[INFO] Monte Carlo de perfis radiais com N_MC = {N_MC} centros aleatórios...")
    print("       (usando o mesmo recorte em RA/Dec do catálogo)")

    ra_min, ra_max   = np.min(ra), np.max(ra)
    dec_min, dec_max = np.min(dec), np.max(dec)

    nbins = len(RADIAL_BINS) - 1
    dens_mc = np.zeros((N_MC, nbins), dtype=float)

    rng = np.random.default_rng(seed=12345)

    for i in range(N_MC):
        ra_rand  = rng.uniform(ra_min, ra_max)
        dec_rand = rng.uniform(dec_min, dec_max)

        center_rand = SkyCoord(ra=ra_rand * u.deg, dec=dec_rand * u.deg, frame="icrs")
        theta_rand = coords.separation(center_rand).deg

        _, _, N_rand, _, dens_rand, _ = build_radial_profile(
            theta_rand, kmag, RADIAL_BINS, kmin=KMIN, kmax=KMAX
        )
        dens_mc[i, :] = dens_rand

        if (i + 1) % 100 == 0:
            print(f"[INFO] ... {i+1} / {N_MC} centros aleatórios processados.")

    dens_mc_mean = np.mean(dens_mc, axis=0)
    dens_mc_std  = np.std(dens_mc,  axis=0)

    print("=========================================================")
    print("[RESULT] Perfil radial MÉDIO dos centros aleatórios:")
    print(" r_mid   dens_mean_rand   sigma_rand")
    for i in range(nbins):
        print(f" {r_mid[i]:4.1f}   {dens_mc_mean[i]:12.3f}   {dens_mc_std[i]:10.3f}")

    a_rand, b_rand, sa_rand, sb_rand = weighted_linear_fit(
        r_mid, dens_mc_mean, dens_mc_std
    )
    print("---------------------------------------------------------")
    print("[RESULT] Ajuste linear do perfil MÉDIO randômico:")
    print(f"  a_rand = {a_rand:.3f} ± {sa_rand:.3f}  sources/deg^2")
    print(f"  b_rand = {b_rand:.3f} ± {sb_rand:.3f}  sources/deg^2/deg")

    # -------------------------------------------------------
    # 3.3 Salvar tudo em arquivo texto
    # -------------------------------------------------------
    outfile = "radial_profile_coldspot_vs_random.txt"
    with open(outfile, "w", encoding="utf-8") as f:
        f.write("# Radial density profile: 2nd CMB Cold Spot vs average random centers\n")
        f.write(f"# Input catalog: {infile}\n")
        f.write(f"# Cold Spot 2 (RA,Dec) = ({RA_CS_DEG:.4f}, {DEC_CS_DEG:.4f}) deg\n")
        f.write("# Radial bins [deg]: " + ", ".join([f"{x:.1f}" for x in RADIAL_BINS]) + "\n")
        f.write(f"# Kmag range: [{KMIN}, {KMAX}]\n")
        f.write(f"# N_MC random centers: {N_MC}\n")
        f.write("#\n")
        f.write("# Columns:\n")
        f.write("# r_in  r_out  r_mid  N_obs  area_deg2  dens_obs  sigma_obs  "
                "dens_rand_mean  dens_rand_std\n")
        for i in range(nbins):
            f.write(
                f"{r_in[i]:5.2f}  {r_out[i]:5.2f}  {r_mid[i]:5.2f}  "
                f"{int(N_obs[i]):7d}  {area_deg2[i]:9.3f}  "
                f"{dens_obs[i]:9.3f}  {sigma_obs[i]:9.3f}  "
                f"{dens_mc_mean[i]:9.3f}  {dens_mc_std[i]:9.3f}\n"
            )

        f.write("#\n")
        f.write("# Linear fit (observed 2nd Cold Spot): dens_obs(r) = a_obs + b_obs * r\n")
        f.write(f"# a_obs = {a_obs:.3f} ± {sa_obs:.3f}  sources/deg^2\n")
        f.write(f"# b_obs = {b_obs:.3f} ± {sb_obs:.3f}  sources/deg^2/deg\n")
        f.write("#\n")
        f.write("# Linear fit (mean random profile): dens_rand(r) = a_rand + b_rand * r\n")
        f.write(f"# a_rand = {a_rand:.3f} ± {sa_rand:.3f}  sources/deg^2\n")
        f.write(f"# b_rand = {b_rand:.3f} ± {sb_rand:.3f}  sources/deg^2/deg\n")

    print("=========================================================")
    print(f"[OK] Perfil radial observado x médio randômico salvo em: {outfile}")


if __name__ == "__main__":
    main()