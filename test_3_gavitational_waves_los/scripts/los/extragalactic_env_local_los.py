# extragalactic_env_local_los.py
# Local environment (Sigma5 / rho_3D) + LOS profile for an extragalactic host (SN/QSO/GW)
# Catalog: SDSS SpecPhoto (SQL via astroquery.sdss)
# Outputs:
#   - neighbors_local_sdss.csv
#   - los_profile_sdss.csv

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astroquery.sdss import SDSS

# ---------------------------
# CONFIG (edit here)
# ---------------------------
RA_HMS  = "15h48m47.536s"
DEC_DMS = "+09d12m00.28s"
Z_HOST  = 0.0560

# Local env
RADIUS_ARCMIN_LOCAL = 30.0
DV_KMS_LOCAL        = 1000.0

# LOS column
RADIUS_ARCMIN_LOS   = 6.0
NBINS_LOS           = 24
Z_MARGIN_TOP        = 0.002
DATA_RELEASE        = 17

# ---------------------------
# Helpers
# ---------------------------
C_KMS = 299792.458

def redshift_window(z_center, dv_kms):
    dz = (dv_kms / C_KMS) * (1.0 + z_center)
    return z_center - dz, z_center + dz

def cylinder_half_length_mpc(z_host, dv_kms):
    Hz = cosmo.H(z_host).to(u.km/u.s/u.Mpc).value
    return dv_kms / Hz

def comoving_area_mpc2(theta_arcmin, z_mid):
    theta = (theta_arcmin * u.arcmin).to(u.rad).value
    DM = cosmo.comoving_transverse_distance(z_mid).to(u.Mpc).value
    R = theta * DM
    return np.pi * R * R

def comoving_depth_mpc(z1, z2):
    chi1 = cosmo.comoving_distance(z1).to(u.Mpc).value
    chi2 = cosmo.comoving_distance(z2).to(u.Mpc).value
    return max(0.0, chi2 - chi1)

def fetch_sdss_sql_cone(ra_deg, dec_deg, radius_arcmin, z_lo=None, z_hi=None, dr=17):
    where_z = ""
    if (z_lo is not None) and (z_hi is not None):
        where_z = f"AND s.z BETWEEN {z_lo:.6f} AND {z_hi:.6f}"
    sql = f"""
        SELECT s.ra, s.dec, s.z, s.class
        FROM SpecPhotoAll AS s
        JOIN dbo.fGetNearbyObjEq({ra_deg:.8f}, {dec_deg:.8f}, {radius_arcmin:.3f}) AS nb
            ON s.objID = nb.objID
        WHERE s.z IS NOT NULL
          {where_z}
    """
    res = SDSS.query_sql(sql, data_release=dr)
    if res is None or len(res) == 0:
        return pd.DataFrame(columns=["ra","dec","z","class"])
    df = res.to_pandas()
    for c in ["ra","dec","z","class"]:
        if c not in df.columns:
            df[c] = np.nan
    return df[["ra","dec","z","class"]].copy()

# ---------------------------
# Local (Sigma5 / rho_3D)
# ---------------------------
def compute_local(host, z_host):
    ra_deg, dec_deg = host.ra.deg, host.dec.deg
    zlo, zhi = redshift_window(z_host, DV_KMS_LOCAL)
    df = fetch_sdss_sql_cone(ra_deg, dec_deg, RADIUS_ARCMIN_LOCAL, zlo, zhi, DATA_RELEASE)
    if df.empty:
        return df, np.nan, np.nan, np.nan

    df["class"] = df["class"].astype(str).upper()
    df = df[df["class"] == "GALAXY"].copy()
    if df.empty:
        return df, np.nan, np.nan, np.nan

    # separações e distância projetada (usar DA ~ comóvel/(1+z) não é necessário p/ ordenação)
    coords = SkyCoord(ra=df["ra"].values*u.deg, dec=df["dec"].values*u.deg)
    sep_arcsec = coords.separation(host).arcsec
    # converter para Mpc com DA(z_host)
    DA = cosmo.angular_diameter_distance(z_host).to(u.Mpc).value
    Rproj_mpc = (sep_arcsec * np.pi / (180.0*3600.0)) * DA
    df["sep_arcsec"] = sep_arcsec
    df["Rproj_Mpc"]  = Rproj_mpc

    df = df.sort_values("Rproj_Mpc").reset_index(drop=True)
    if len(df) >= 5:
        r5 = df.loc[4, "Rproj_Mpc"]
        area = np.pi * (r5**2)
        sigma5 = 5.0 / area  # 1/Mpc^2
    else:
        r5 = np.nan
        sigma5 = np.nan

    halfL = cylinder_half_length_mpc(z_host, DV_KMS_LOCAL)  # Mpc
    in_cyl = df["Rproj_Mpc"].values <= 1.0
    N_cyl  = int(np.count_nonzero(in_cyl))
    vol    = np.pi * (1.0**2) * (2.0 * halfL)
    rho3d  = (N_cyl/vol) if vol>0 else np.nan

    return df, sigma5, r5, rho3d

# ---------------------------
# LOS profile
# ---------------------------
def compute_los(host, z_host):
    ra_deg, dec_deg = host.ra.deg, host.dec.deg
    zmax = z_host + Z_MARGIN_TOP
    df_all = fetch_sdss_sql_cone(ra_deg, dec_deg, RADIUS_ARCMIN_LOS, 0.0, zmax, DATA_RELEASE)
    if df_all.empty:
        return pd.DataFrame(columns=["z_lo","z_hi","z_mid","N","area_Mpc2","depth_Mpc","rho_Mpc3"])

    df_all["class"] = df_all["class"].astype(str).str.upper()
    df_all = df_all[df_all["class"] == "GALAXY"].copy()
    if df_all.empty:
        return pd.DataFrame(columns=["z_lo","z_hi","z_mid","N","area_Mpc2","depth_Mpc","rho_Mpc3"])

    edges = np.linspace(0.0, zmax, NBINS_LOS+1)
    rows = []
    for i in range(NBINS_LOS):
        z_lo, z_hi = edges[i], edges[i+1]
        z_mid = 0.5*(z_lo+z_hi)
        sel = (df_all["z"] >= z_lo) & (df_all["z"] < z_hi)
        N = int(sel.sum())
        area  = comoving_area_mpc2(RADIUS_ARCMIN_LOS, z_mid)
        depth = comoving_depth_mpc(z_lo, z_hi)
        rho   = (N/(area*depth)) if (area>0 and depth>0) else np.nan
        rows.append({"z_lo":z_lo,"z_hi":z_hi,"z_mid":z_mid,
                     "N":N,"area_Mpc2":area,"depth_Mpc":depth,"rho_Mpc3":rho})
    prof = pd.DataFrame(rows)
    prof["rho_Mpc3_smooth"] = prof["rho_Mpc3"].rolling(3, center=True, min_periods=1).median()
    return prof

# ---------------------------
# Main
# ---------------------------
def main():
    host = SkyCoord(RA_HMS, DEC_DMS, frame="icrs")
    print("=== Extragalactic local+LOS (SDSS) ===")
    print(f"Host (RA,Dec,z): ({host.ra.deg:.6f}, {host.dec.deg:.6f}, {Z_HOST:.5f})")

    df_local, sigma5, r5, rho3d = compute_local(host, Z_HOST)
    if df_local.empty:
        print("[LOCAL] No GALAXY spec-z found in local cone.")
    else:
        df_local.to_csv("neighbors_local_sdss.csv", index=False)
        print(f"[LOCAL] N={len(df_local)} | Sigma5={sigma5 if np.isfinite(sigma5) else 'n/d'} 1/Mpc^2 | r5={r5 if np.isfinite(r5) else 'n/d'} Mpc | rho_3D={rho3d:.4f} 1/Mpc^3")
        print("Saved: neighbors_local_sdss.csv")

    prof = compute_los(host, Z_HOST)
    if prof.empty:
        print("[LOS] No LOS galaxies (spec-z). Consider using photo-z catalogs (DESI Legacy).")
    else:
        prof.to_csv("los_profile_sdss.csv", index=False)
        med = np.nanmedian(prof["rho_Mpc3"])
        p16, p84 = np.nanpercentile(prof["rho_Mpc3"], [16,84])
        print(f"[LOS] Bins={len(prof)} | median rho_LOS={med:.4e} 1/Mpc^3 (16–84%: {p16:.4e}–{p84:.4e})")
        print("Saved: los_profile_sdss.csv")

if __name__ == "__main__":
    main()