# compute_K_SN2023zkd.py
# K/Ksat for SN 2023zkd using (A) K(z) from Pantheon Excel and (B) local density proxy
# Prints Ktilde_z, Ktilde_rho and a blended Ktilde_final.

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u

# ---------------------------
# Fixed event info (Gagliano)
# ---------------------------
EVENT = "SN 2023zkd"
z_event = 0.0560

# Local environment already measured (CET units)
Sigma5 = 0.545   # Mpc^-2 (from your SDSS local cone)
r5_mpc = 1.709   # Mpc
rho3D  = 0.0221  # Mpc^-3 (cylinder R=1 Mpc, half-L = dv/H(z))

# ---------------------------
# A) K(z) from Pantheon Excel (if available)
# ---------------------------
# Put your Pantheon file here (same folder or change the path)
PANTHEON_XLSX = Path("pantheon_shoes_CET_density_analysis.xlsx")

# If your Excel has a specific sheet with K(z), set it here; else leave as None
SHEET_NAME = None   # e.g., "Kz_curve" if you have it

# Columns expected (any of these names will be searched, case-insensitive)
Z_COL_CANDIDATES = ["z", "zcmb", "z_cmb", "redshift"]
K_COL_CANDIDATES = ["Ktilde", "K_z", "K", "tau"]

def read_Kz_from_excel(xlsx_path: Path, sheet_name=None):
    if not xlsx_path.exists():
        return None
    # try to read all sheets if sheet_name is None
    if sheet_name is None:
        xls = pd.ExcelFile(xlsx_path)
        sheets = xls.sheet_names
        for sh in sheets:
            df = pd.read_excel(xlsx_path, sheet_name=sh)
            zcol = next((c for c in df.columns if str(c).strip().lower() in Z_COL_CANDIDATES), None)
            kcol = next((c for c in df.columns if str(c).strip().lower() in [s.lower() for s in K_COL_CANDIDATES]), None)
            if zcol and kcol:
                sub = df[[zcol, kcol]].dropna()
                sub.columns = ["z", "Ktilde"]
                if len(sub) >= 2:
                    return sub.sort_values("z").reset_index(drop=True)
        return None
    else:
        df = pd.read_excel(xlsx_path, sheet_name=sheet_name)
        zcol = next((c for c in df.columns if str(c).strip().lower() in Z_COL_CANDIDATES), None)
        kcol = next((c for c in df.columns if str(c).strip().lower() in [s.lower() for s in K_COL_CANDIDATES]), None)
        if zcol and kcol:
            sub = df[[zcol, kcol]].dropna()
            sub.columns = ["z", "Ktilde"]
            if len(sub) >= 2:
                return sub.sort_values("z").reset_index(drop=True)
    return None

def interp_Ktilde_at_z(table, z):
    if table is None or len(table) < 2:
        return np.nan
    zgrid = table["z"].values.astype(float)
    Kgrid = table["Ktilde"].values.astype(float)
    # clip to bounds to avoid NaN if z is outside range
    z = np.clip(z, zgrid.min(), zgrid.max())
    return float(np.interp(z, zgrid, Kgrid))

# ---------------------------
# B) K from local density (sigmoid on rho/rho_crit_env)
# ---------------------------
# Set your calibrated CET environmental critical density and slope here
rho_crit_env = 0.05   # Mpc^-3  <-- replace with your calibrated value
m_rho        = 8.0    # slope of the logistic around rho/rho_crit_env = 1

def Ktilde_from_rho(rho_local, rho_crit_env, m_rho):
    x = m_rho * ((rho_local / rho_crit_env) - 1.0)
    return 1.0 / (1.0 + np.exp(-x))  # logistic in [0,1]

# ---------------------------
# Optional: blend z-based and rho-based estimates
# ---------------------------
def blend_K(Kz, Krho, w_local=0.3):
    if not np.isfinite(Kz) and np.isfinite(Krho):
        return Krho
    if not np.isfinite(Krho) and np.isfinite(Kz):
        return Kz
    if not np.isfinite(Kz) and not np.isfinite(Krho):
        return np.nan
    return (1.0 - w_local) * Kz + w_local * Krho

# ---------------------------
# Main
# ---------------------------
def main():
    # A) K(z) from Pantheon Excel (if present)
    Kz_table = read_Kz_from_excel(PANTHEON_XLSX, sheet_name=SHEET_NAME)
    Ktilde_z = interp_Ktilde_at_z(Kz_table, z_event) if Kz_table is not None else np.nan

    # B) K from local density proxy (CET environmental)
    Ktilde_rho = Ktilde_from_rho(rho3D, rho_crit_env, m_rho)

    # Blend (optional)
    Ktilde_final = blend_K(Ktilde_z, Ktilde_rho, w_local=0.30)

    # Print summary
    print("=== CET K-estimate for SN 2023zkd ===")
    print(f"Event: {EVENT} | z = {z_event:.5f}")
    print(f"Local env: Sigma5 = {Sigma5:.3f} Mpc^-2 | r5 = {r5_mpc:.3f} Mpc | rho3D = {rho3D:.4f} Mpc^-3")
    print("\n[A] From Pantheon K(z):")
    if np.isfinite(Ktilde_z):
        print(f"   K/Ksat(z={z_event:.3f}) = {Ktilde_z:.4f}")
    else:
        print("   K/Ksat: not available (no K(z) found in Excel)")
    print("\n[B] From local density (rho/rho_crit_env → logistic):")
    print(f"   rho_crit_env = {rho_crit_env:.4f} Mpc^-3 | m_rho = {m_rho:.2f}")
    print(f"   K/Ksat(rho) = {Ktilde_rho:.4f}")
    print("\n[Blended] (optional)")
    print(f"   K/Ksat (final) = {Ktilde_final:.4f}")

    # Save a tiny json summary (optional)
    out = {
        "event": EVENT,
        "z": z_event,
        "Sigma5_Mpc^-2": Sigma5,
        "r5_Mpc": r5_mpc,
        "rho3D_Mpc^-3": rho3D,
        "Ktilde_z": None if not np.isfinite(Ktilde_z) else float(Ktilde_z),
        "Ktilde_rho": float(Ktilde_rho),
        "Ktilde_final": None if not np.isfinite(Ktilde_final) else float(Ktilde_final),
        "rho_crit_env": rho_crit_env,
        "m_rho": m_rho
    }
    try:
        import json
        with open("K_SN2023zkd_summary.json", "w") as f:
            json.dump(out, f, indent=2)
        print("\nSaved: K_SN2023zkd_summary.json")
    except Exception as e:
        pass

if __name__ == "__main__":
    main()