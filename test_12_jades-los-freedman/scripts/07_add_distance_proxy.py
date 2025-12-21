#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D LOS (cone + polygon) for GOODS-S

Uses:
 - JADES base catalog with RA_TARG, Dec_TARG, z_best
 - CANDELS 3D tracers: RA, DEC, z
 - GOODS-S mask defined as 2D polygons

Method: identical to the CEERS implementation used in Paper I.
"""

import numpy as np
import math
import json
import pandas as pd
from astropy.table import Table

C_KM_S = 299792.458

# ------------------------
# CONFIGURATION
# ------------------------
JADES_FITS   = "jades_GOODSS_LOS3D_clean.fits"
CANDELS_FITS = "candels_GOODSS_3D.fits"
MASK_CSV    = "goods_s_clean.csv"

ALPHA = 1.0
EXCLUDE_DV_KMS = 1500.0
SHELL_DZ = 0.0

OUT_FITS = "jades_GOODSS_LOS3D_cone.fits"

# ------------------------
# AUXILIARY FUNCTIONS
# ------------------------

def point_in_polygon(ra, dec, poly):
    """Ray-casting point-in-polygon test."""
    inside = False
    n = len(poly)
    x1, y1 = poly[0]
    for i in range(1, n+1):
        x2, y2 = poly[i % n]
        if (dec > min(y1, y2)) and (dec <= max(y1, y2)) and (ra <= max(x1, x2)):
            if y1 != y2:
                xinters = (dec - y1) * (x2 - x1) / (y2 - y1) + x1
            else:
                xinters = ra
            if ra <= xinters:
                inside = not inside
        x1, y1 = x2, y2
    return inside


def load_polygons(path):
    """Load GOODS-S polygons from CSV file."""
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    pts = []
    k = 1
    while f"ra{k}" in cols and f"dec{k}" in cols:
        pts.append((f"ra{k}", f"dec{k}"))
        k += 1

    polys = []
    for _, row in df.iterrows():
        poly = []
        for ra_c, dec_c in pts:
            poly.append((float(row[ra_c]), float(row[dec_c])))
        if poly[0] != poly[-1]:
            poly.append(poly[0])
        polys.append(poly)
    return polys


def approx_dv(zb, z):
    """Approximate line-of-sight velocity separation in km/s."""
    return C_KM_S * np.abs(zb - z) / (1 + max(zb, 1e-9))


# ------------------------
# CORE LOS COMPUTATION
# ------------------------

def compute_LOS(ra_b, dec_b, z_b, ext_ra, ext_dec, ext_z, polygons):
    """Compute 3D LOS counts and weighted sum for a single JADES target."""
    if not np.isfinite(z_b) or z_b <= 0:
        return 0, 0.0

    # Identify which polygon the target belongs to
    poly_target = None
    for poly in polygons:
        if point_in_polygon(ra_b, dec_b, poly):
            poly_target = poly
            break
    if poly_target is None:
        return 0, 0.0

    # Directional cut: 0 <= z <= z_b
    mask = (ext_z >= 0) & (ext_z <= z_b)

    # Tracers inside the same polygon
    mask_poly = []
    for i in range(mask.size):
        if mask[i] and point_in_polygon(ext_ra[i], ext_dec[i], poly_target):
            mask_poly.append(True)
        else:
            mask_poly.append(False)
    mask &= np.array(mask_poly)

    if not np.any(mask):
        return 0, 0.0

    z_use = ext_z[mask]

    # Exclude companions with |dv| < 1500 km/s
    dv = approx_dv(z_b, z_use)
    z_use = z_use[dv >= EXCLUDE_DV_KMS]

    if z_use.size == 0:
        return 0, 0.0

    dz = z_b - z_use
    w = 1.0 / np.power(1 + dz, ALPHA)

    return int(z_use.size), float(np.sum(w))


# ------------------------
# MAIN
# ------------------------

def main():
    print("=== Loading JADES catalog ===")
    j = Table.read(JADES_FITS, hdu=1)
    print("Number of JADES objects:", len(j))

    print("=== Loading CANDELS 3D tracers ===")
    c = Table.read(CANDELS_FITS, hdu=1)
    ext_ra  = c["RA"].data
    ext_dec = c["DEC"].data
    ext_z   = c["z"].data
    print("Number of tracers:", len(c))

    print("=== Loading GOODS-S mask ===")
    polys = load_polygons(MASK_CSV)
    print("Number of polygons:", len(polys))

    los_count = []
    los_sumw  = []

    print("=== Computing CEERS-style 3D LOS ===")
    for i, row in enumerate(j):
        ra_b  = float(row["RA_TARG"])
        dec_b = float(row["Dec_TARG"])
        z_b   = float(row["z_best"])

        n, w = compute_LOS(ra_b, dec_b, z_b, ext_ra, ext_dec, ext_z, polys)
        los_count.append(n)
        los_sumw.append(w)

        if (i + 1) % 50 == 0:
            print(f"[{i+1}/{len(j)}]  count={n}")

    j["LOS3D_cone_count"] = los_count
    j["LOS3D_cone_sumw"]  = los_sumw

    # Classification into three regimes
    arr = np.array(los_count)
    valid = arr[arr > 0]
    if valid.size > 3:
        q1, q2 = np.quantile(valid, [1/3, 2/3])
        cl = np.full(len(arr), -1)
        cl[(arr > 0) & (arr <= q1)] = 0
        cl[(arr > q1) & (arr <= q2)] = 1
        cl[(arr > q2)] = 2
        j["LOS3D_cone_class"] = cl

        print("Q1, Q2:", q1, q2)
        for c0 in [-1, 0, 1, 2]:
            print("class", c0, ":", np.sum(cl == c0))
    else:
        j["LOS3D_cone_class"] = -1

    print("=== Saving output ===")
    j.write(OUT_FITS, overwrite=True)
    print("Saved:", OUT_FITS)


if __name__ == "__main__":
    main()