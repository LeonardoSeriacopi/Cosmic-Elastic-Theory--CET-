# gagliano_env_metrics_sn2023zkd_sql.py
# Ambiente do host de SN 2023zkd (Gagliano et al. 2025) via SDSS (SQL, sem limite de 3')
# Saídas: neighbors_sn2023zkd.csv + métricas Σ5 e ρ_3D

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astroquery.sdss import SDSS

# ---------------------------
# Host (J2000) do artigo
# ---------------------------
RA_HMS  = "15h48m47.536s"
DEC_DMS = "+09d12m00.28s"
Z_HOST  = 0.0560

# Parâmetros de busca/métricas
RADIUS_ARCMIN = 30.0          # raio no céu p/ busca (sem limite via SQL)
DV_KMS        = 1000.0        # janela de velocidade ± (km/s)
CYL_RADIUS_MPC= 1.0           # raio físico do cilindro p/ ρ_3D

# ---------------------------
# Helpers
# ---------------------------
C_KMS = 299792.458

def redshift_window(z_host, dv_kms=DV_KMS):
    dz = (dv_kms / C_KMS) * (1.0 + z_host)
    return z_host - dz, z_host + dz

def dist_proj_kpc(sep_arcsec, z_ref):
    DA = cosmo.angular_diameter_distance(z_ref).to(u.kpc)  # kpc
    theta = (sep_arcsec * u.arcsec).to(u.rad)
    return theta.value * DA.value  # kpc

def cylinder_half_length_mpc(z_host, dv_kms=DV_KMS):
    Hz = cosmo.H(z_host).to(u.km/u.s/u.Mpc).value  # km/s/Mpc
    return dv_kms / Hz   # Mpc

# ---------------------------
# Consulta via SQL (SpecPhotoAll + fGetNearbyObjEq)
# ---------------------------
def fetch_sdss_sql(ra_deg, dec_deg, radius_arcmin, z_lo, z_hi, data_release=17):
    # Atenção: 'class' é palavra reservada em Python, mas coluna no SDSS; tratamos depois.
    sql = f"""
        SELECT s.ra, s.dec, s.z, s.class
        FROM SpecPhotoAll AS s
        JOIN dbo.fGetNearbyObjEq({ra_deg:.8f}, {dec_deg:.8f}, {radius_arcmin:.3f}) AS nb
            ON s.objID = nb.objID
        WHERE s.z BETWEEN {z_lo:.6f} AND {z_hi:.6f}
          AND s.z IS NOT NULL
    """
    res = SDSS.query_sql(sql, data_release=data_release)
    if res is None or len(res) == 0:
        return pd.DataFrame(columns=["ra","dec","z","class"])
    df = res.to_pandas()
    # Normaliza colunas
    for c in ["ra","dec","z","class"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["ra","dec","z","class"]].copy()

# ---------------------------
# Main
# ---------------------------
def main():
    host = SkyCoord(RA_HMS, DEC_DMS, frame="icrs")
    ra_deg, dec_deg = host.ra.deg, host.dec.deg
    zlo, zhi = redshift_window(Z_HOST, DV_KMS)

    df = fetch_sdss_sql(ra_deg, dec_deg, RADIUS_ARCMIN, zlo, zhi, data_release=17)
    if df.empty:
        print("[INFO] Nenhum espectro SDSS no cone/intervalo de z. "
              "Sugestão: tente aumentar o raio ou usar DESI (Legacy).")
        return

    # Filtro de classe GALAXY
    df["class"] = df["class"].astype(str).str.upper()
    df = df[df["class"] == "GALAXY"].copy()
    if df.empty:
        print("[INFO] Sem GALAXY com spec-z na janela. Tente relaxar filtros.")
        return

    # Distâncias projetadas
    coords = SkyCoord(ra=df["ra"].values * u.deg, dec=df["dec"].values * u.deg)
    sep_arcsec = coords.separation(host).arcsec
    df["sep_arcsec"] = sep_arcsec
    df["Rproj_kpc"]  = [dist_proj_kpc(s, Z_HOST) for s in sep_arcsec]

    # Ordena por Rproj p/ Σ5
    df = df.sort_values("Rproj_kpc").reset_index(drop=True)

    # Σ5 (5º vizinho)
    if len(df) >= 5:
        r5_kpc = df.loc[4, "Rproj_kpc"]
        area_mpc2 = np.pi * (r5_kpc/1000.0)**2
        sigma5 = 5.0 / area_mpc2
    else:
        r5_kpc = np.nan
        sigma5 = np.nan

    # ρ_3D em cilindro de R=1 Mpc, half-L via ±dv
    half_len_mpc = cylinder_half_length_mpc(Z_HOST, DV_KMS)
    in_cyl_proj = df["Rproj_kpc"].values <= (CYL_RADIUS_MPC * 1000.0)
    N_proj = int(np.count_nonzero(in_cyl_proj))
    vol_mpc3 = np.pi * CYL_RADIUS_MPC**2 * (2.0 * half_len_mpc)
    rho_3d = N_proj / vol_mpc3 if vol_mpc3 > 0 else np.nan

    # Salva tabela
    out_csv = "neighbors_sn2023zkd.csv"
    df.to_csv(out_csv, index=False)

    # Resumo
    print("=== Ambiente do host SN 2023zkd (SDSS SQL) ===")
    print(f"Host (RA,Dec,z): ({ra_deg:.6f}, {dec_deg:.6f}, {Z_HOST:.5f})")
    print(f"Busca: raio={RADIUS_ARCMIN:.1f}' | Δv=±{DV_KMS:.0f} km/s | z∈[{zlo:.5f},{zhi:.5f}]")
    print(f"GALAXY com spec-z no cone: {len(df)}")
    if np.isfinite(sigma5):
        print(f"Σ5 = {sigma5:.4f} 1/Mpc^2  | r5 = {r5_kpc/1000.0:.3f} Mpc")
    else:
        print("Σ5 = n/d (menos de 5 vizinhos espectroscópicos)")
    print(f"ρ_3D = {rho_3d:.4f} 1/Mpc^3 | cilindro R={CYL_RADIUS_MPC:.1f} Mpc, half-L={half_len_mpc:.3f} Mpc")
    print(f"Tabela salva: {out_csv}")

if __name__ == "__main__":
    main()