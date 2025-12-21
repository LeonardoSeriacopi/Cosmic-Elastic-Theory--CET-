#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distance normalization of the 3D LOS (JADES + CANDELS/GOODS-S)

Input:
  - jades_GOODSS_LOS3D.fits
    (must contain at least: z_best, LOS3D_count)

Output:
  - jades_GOODSS_LOS3D_distnorm.fits

New columns added:
  - LOS3D_Dc_full   : comoving distance up to z_best   [Mpc]
  - LOS3D_Dc_eff    : comoving distance up to z_eff    [Mpc], z_eff = min(z_best, z_max)
  - LOS3D_density   : LOS3D_count / Dc_eff             [1/Mpc]
  - LOS3D_scaled    : LOS3D_count * (Dc_full/Dc_eff)   (LOS extrapolated to z_best)
  - LOS3Ddens_class : classes 0/1/2 in terciles of LOS3D_density
"""

import numpy as np
from astropy.table import Table
from astropy.io import fits

# -----------------------------
# Simple flat ΛCDM cosmology
# -----------------------------

C_KM_S = 299792.458  # km/s
H0     = 70.0        # km/s/Mpc
OMEGA_M = 0.3
OMEGA_L = 1.0 - OMEGA_M

def E_z(z):
    """E(z) = H(z)/H0 for flat ΛCDM cosmology."""
    return np.sqrt(OMEGA_M * (1.0 + z)**3 + OMEGA_L)

def comoving_distance(z, nsteps=2000):
    """
    Comoving distance Dc(z) in Mpc via simple numerical integration.
    Dc(z) = (c/H0) ∫_0^z dz'/E(z')
    """
    z = float(z)
    if not np.isfinite(z) or z <= 0:
        return 0.0
    z_grid = np.linspace(0.0, z, nsteps)
    Ez = E_z(z_grid)
    # Trapezoidal integration in z
    integral = np.trapz(1.0 / Ez, z_grid)
    Dc = (C_KM_S / H0) * integral  # Mpc
    return Dc

# -----------------------------
# LOS / catalog parameters
# -----------------------------

JADES_FITS = "jades_GOODSS_LOS3D.fits"           # input
OUT_FITS   = "jades_GOODSS_LOS3D_distnorm.fits" # output

ZMAX_CATALOG = 3.0  # effective maximum depth of CANDELS tracers (z_cut)

# -----------------------------
# Load JADES catalog
# -----------------------------

print("=== Loading JADES catalog with LOS3D ===")
jades = Table.read(JADES_FITS, hdu=1)
print(f"Number of objects in catalog: {len(jades)}")

if "z_best" not in jades.colnames:
    raise SystemExit("[ERR] Catalog does not contain column 'z_best'.")
if "LOS3D_count" not in jades.colnames:
    raise SystemExit("[ERR] Catalog does not contain column 'LOS3D_count'.")

z_best = np.array(jades["z_best"], dtype=float)
los3d_count = np.array(jades["LOS3D_count"], dtype=float)

# -----------------------------
# Compute Dc_full and Dc_eff
# -----------------------------

print("=== Computing comoving distances (Dc_full, Dc_eff) ===")

Dc_full = np.zeros_like(z_best, dtype=float)
Dc_eff  = np.zeros_like(z_best, dtype=float)

for i, z in enumerate(z_best):
    if np.isfinite(z) and z > 0:
        z_eff = min(z, ZMAX_CATALOG)
        Dc_full[i] = comoving_distance(z)        # full photon path
        Dc_eff[i]  = comoving_distance(z_eff)    # effective path sampled by the catalog
    else:
        Dc_full[i] = 0.0
        Dc_eff[i]  = 0.0

# -----------------------------
# Build normalized LOS metrics
# -----------------------------

print("=== Building normalized LOS metrics ===")

LOS3D_density = np.full_like(los3d_count, np.nan, dtype=float)
LOS3D_scaled  = np.full_like(los3d_count, np.nan, dtype=float)

valid_path = (Dc_eff > 0) & np.isfinite(los3d_count)

LOS3D_density[valid_path] = los3d_count[valid_path] / Dc_eff[valid_path]
LOS3D_scaled[valid_path]  = los3d_count[valid_path] * (Dc_full[valid_path] / Dc_eff[valid_path])

# -----------------------------
# Classification into terciles (0/1/2) using LOS3D_density
# -----------------------------

print("=== Classifying into 3 regimes using LOS3D_density ===")

LOS3Ddens_class = np.full(len(jades), -1, dtype=int)

mask_class = valid_path & np.isfinite(LOS3D_density) & (LOS3D_density > 0)

if np.count_nonzero(mask_class) > 0:
    vals = LOS3D_density[mask_class]
    q1, q2 = np.quantile(vals, [1.0/3.0, 2.0/3.0])
    print(f"LOS3D_density terciles: Q1={q1:.4g}, Q2={q2:.4g}")

    # class 0: low LOS
    LOS3Ddens_class[(LOS3D_density <= q1) & mask_class] = 0
    # class 1: intermediate
    LOS3Ddens_class[(LOS3D_density > q1) & (LOS3D_density <= q2) & mask_class] = 1
    # class 2: high LOS
    LOS3Ddens_class[(LOS3D_density > q2) & mask_class] = 2

    for c in [-1, 0, 1, 2]:
        n_c = np.count_nonzero(LOS3Ddens_class == c)
        print(f"class {c}: {n_c} objects")
else:
    print("[WARN] No valid objects to classify LOS3D_density.")

# -----------------------------
# Append columns and save
# -----------------------------

print("=== Adding columns to catalog ===")

jades["LOS3D_Dc_full"]   = Dc_full
jades["LOS3D_Dc_eff"]    = Dc_eff
jades["LOS3D_density"]   = LOS3D_density
jades["LOS3D_scaled"]    = LOS3D_scaled
jades["LOS3Ddens_class"] = LOS3Ddens_class

print(f"=== Saving catalog to: {OUT_FITS} ===")
jades.write(OUT_FITS, overwrite=True)
print("Done.")