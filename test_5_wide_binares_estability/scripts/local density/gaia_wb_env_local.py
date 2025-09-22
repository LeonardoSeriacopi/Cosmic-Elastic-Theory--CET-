# gaia_wb_env_local.py
# Local environment metrics for a wide-binary field using Gaia DR3 (NO LOS).
# Outputs:
#   - gaia_neighbors_local.csv   (vizinhanca estelar com parallax/PM)
# Prints:
#   - Sigma5 (1/pc^2) and rho_3D (1/pc^3) in stellar counts

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import Angle
from astroquery.gaia import Gaia

# ---------------------------
# CONFIG (edit here)
# ---------------------------
# Barycenter (ou uma das estrelas) do par
RA_HMS  = "10h10m10.0s"
DEC_DMS = "+10d10m10.0s"

# Parallax alvo (mas) e janela relativa/absoluta
PARALLAX_TARGET_MAS = 10.0     # ex.: 10 mas ~ 100 pc
PARALLAX_TOL_MAS    = 1.0      # tolerância absoluta (±1 mas)
PM_TOL_MASYR        = 5.0      # filtro brando de proper motion (opcional)

# Raio de busca no céu
RADIUS_ARCMIN = 30.0

# Parâmetros do cilindro para rho_3D (em pc)
CYL_RADIUS_PC = 20.0           # raio físico projetado (pc)
DEPTH_PC      = 40.0           # profundidade (LOS) do cilindro (pc)

# ---------------------------
# Helpers
# ---------------------------
def hmsdms_to_deg(ra_hms, dec_dms):
    c = SkyCoord(ra_hms, dec_dms, frame="icrs")
    return c.ra.deg, c.dec.deg, c

def query_gaia_cone(ra_deg, dec_deg, radius_arcmin):
    # Gaia early DR3 main table: gaiaedr3.gaia_source (astroquery rota DR3)
    radius_deg = Angle(radius_arcmin, unit=u.arcmin).degree
    query = f"""
    SELECT
      source_id, ra, dec, parallax, pmra, pmdec, phot_g_mean_mag
    FROM gaiadr3.gaia_source
    WHERE CONTAINS(
        POINT('ICRS', ra, dec),
        CIRCLE('ICRS', {ra_deg}, {dec_deg}, {radius_deg})
    ) = 1
      AND parallax IS NOT NULL
    """
    job = Gaia.launch_job_async(query)
    return job.get_results().to_pandas()

def projected_distance_pc(sep_arcsec, distance_pc):
    # s = theta * d ; theta (rad) = arcsec * (pi/648000)
    return (sep_arcsec * np.pi / (180.0*3600.0)) * distance_pc

# ---------------------------
# Main
# ---------------------------
def main():
    ra_deg, dec_deg, center = hmsdms_to_deg(RA_HMS, DEC_DMS)
    df = query_gaia_cone(ra_deg, dec_deg, RADIUS_ARCMIN)
    if df.empty:
        print("[INFO] No Gaia sources in field (unexpected).")
        return

    # Filtros por paralaxe e PM (janela simples)
    par = df["parallax"].astype(float)
    pmra = df["pmra"].astype(float)
    pmdec= df["pmdec"].astype(float)

    sel = np.isfinite(par)
    sel &= (par >= PARALLAX_TARGET_MAS - PARALLAX_TOL_MAS) & (par <= PARALLAX_TARGET_MAS + PARALLAX_TOL_MAS)
    # filtro brando de PM (opcional)
    sel &= np.isfinite(pmra) & np.isfinite(pmdec)
    sel &= (np.abs(pmra) <= np.abs(pmra).median() + PM_TOL_MASYR) & (np.abs(pmdec) <= np.abs(pmdec).median() + PM_TOL_MASYR)

    df = df[sel].copy()
    if df.empty:
        print("[INFO] No stars after parallax/PM filter. Consider relaxing tolerances.")
        return

    # Distância média (pc) a partir da paralaxe alvo (aprox): d(pc)=1000/parallax(mas)
    d_pc = 1000.0 / PARALLAX_TARGET_MAS

    # Separações angulares ao centro e distância projetada
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg)
    sep_arcsec = coords.separation(center).arcsec
    df["sep_arcsec"] = sep_arcsec
    df["Rproj_pc"] = projected_distance_pc(sep_arcsec, d_pc)

    # Sigma5 (5º vizinho)
    df = df.sort_values("Rproj_pc").reset_index(drop=True)
    if len(df) >= 5:
        r5_pc = df.loc[4, "Rproj_pc"]
        area_pc2 = np.pi * (r5_pc**2)
        sigma5 = 5.0 / area_pc2  # 1/pc^2
    else:
        r5_pc = np.nan
        sigma5 = np.nan

    # rho_3D: dentro de cilindro R=CYL_RADIUS_PC e profundidade DEPTH_PC
    in_cyl = df["Rproj_pc"].values <= CYL_RADIUS_PC
    N_cyl  = int(np.count_nonzero(in_cyl))
    vol_pc3 = np.pi * (CYL_RADIUS_PC**2) * DEPTH_PC
    rho3d = (N_cyl / vol_pc3) if vol_pc3 > 0 else np.nan

    # Salvar vizinhança
    df.to_csv("gaia_neighbors_local.csv", index=False)

    print("=== Gaia local stellar environment ===")
    print(f"Center (RA,Dec) = ({ra_deg:.6f}, {dec_deg:.6f}) | parallax~ {PARALLAX_TARGET_MAS:.2f}±{PARALLAX_TOL_MAS:.2f} mas")
    print(f"Field radius = {RADIUS_ARCMIN:.1f} arcmin | stars kept = {len(df)}")
    if np.isfinite(sigma5):
        print(f"Sigma5 = {sigma5:.4e} 1/pc^2  | r5 = {r5_pc:.2f} pc")
    else:
        print("Sigma5 = n/d (fewer than 5 stars)")
    print(f"rho_3D = {rho3d:.4e} 1/pc^3 | cylinder R={CYL_RADIUS_PC:.1f} pc, depth={DEPTH_PC:.1f} pc")
    print("Saved: gaia_neighbors_local.csv")

if __name__ == "__main__":
    main()