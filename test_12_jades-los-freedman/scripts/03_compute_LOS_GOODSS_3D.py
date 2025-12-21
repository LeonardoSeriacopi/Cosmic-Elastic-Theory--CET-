# los_goodsS_candels_3D.py

import numpy as np
import csv
from astropy.io import fits
from astropy.table import Table
from matplotlib.path import Path

# ==========================
# INPUT FILES
# ==========================
jades_file   = "jades_GOODSS_LOS.fits"        # current JADES catalog
mask_file    = "goods_s_clean.csv"            # GOODS-S mask (ra1..ra4, dec1..dec4, label)
candels_file = "candels_GOODSS_3D.fits"       # produced by the previous script

# Redshift limits for LOS neighbor counting
z_min_los = 0.0
z_max_los = 3.0   # e.g., count foreground structures up to z ~ 3

print("=== Loading JADES catalog ===")
with fits.open(jades_file) as hdul:
    jades_tab = Table(hdul[1].data)

N_jades = len(jades_tab)
print(f"Total JADES objects: {N_jades}")

# JADES sky coordinates
ra_j  = np.array(jades_tab["RA_TARG"])
dec_j = np.array(jades_tab["Dec_TARG"])

print("\n=== Loading GOODS-S mask ===")
polygons = []
labels   = []

with open(mask_file, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ra1  = float(row["ra1"])
        dec1 = float(row["dec1"])
        ra2  = float(row["ra2"])
        dec2 = float(row["dec2"])
        ra3  = float(row["ra3"])
        dec3 = float(row["dec3"])
        ra4  = float(row["ra4"])
        dec4 = float(row["dec4"])
        label = row.get("label", "")

        verts = [
            (ra1, dec1),
            (ra2, dec2),
            (ra3, dec3),
            (ra4, dec4),
        ]
        polygons.append(verts)
        labels.append(label)

n_poly = len(polygons)
print(f"Number of polygons in GOODS-S mask: {n_poly}")

print("\n=== Loading CANDELS GOODS-S 3D catalog ===")
with fits.open(candels_file) as hdul:
    candels_tab = Table(hdul[1].data) if len(hdul) > 1 else Table(hdul[0].data)

ra_c = np.array(candels_tab["RA"])
dec_c = np.array(candels_tab["DEC"])
z_c  = np.array(candels_tab["z"])

# Select CANDELS objects within the LOS redshift range
mask_z = (z_c > z_min_los) & (z_c < z_max_los)
ra_c  = ra_c[mask_z]
dec_c = dec_c[mask_z]
z_c   = z_c[mask_z]

N_candels = len(z_c)
print(f"Total CANDELS objects with {z_min_los} < z < {z_max_los}: {N_candels}")

# ==========================
# PRE-ALLOCATE OUTPUT ARRAYS
# ==========================
LOS3D_count = np.zeros(N_jades, dtype=float) - 999.0  # default outside mask
LOS3D_class = np.zeros(N_jades, dtype=int)   - 1      # -1 = outside mask
LOS3D_field = np.zeros(N_jades, dtype=int)   - 1      # polygon index or -1

# Track how many JADES objects fall inside at least one polygon
mask_inside_any = np.zeros(N_jades, dtype=bool)

print("\n=== Computing 3D LOS per GOODS-S polygon ===")

points_c = np.vstack([ra_c, dec_c]).T
points_j = np.vstack([ra_j, dec_j]).T

for ip, (verts, lab) in enumerate(zip(polygons, labels)):
    path = Path(verts)

    # CANDELS objects inside polygon (2D sky projection)
    inside_c = path.contains_points(points_c)
    n_c_poly = inside_c.sum()

    # JADES objects inside polygon
    inside_j = path.contains_points(points_j)
    idx_j = np.where(inside_j)[0]
    n_j_poly = len(idx_j)

    if n_j_poly > 0:
        print(
            f"Polygon {ip:02d} ({lab}): "
            f"{n_j_poly} JADES, {n_c_poly} CANDELS (LOS)"
        )

        # 3D LOS density = number of CANDELS objects within z range inside polygon
        count_value = float(n_c_poly)

        LOS3D_count[idx_j] = count_value
        LOS3D_field[idx_j] = ip
        mask_inside_any[idx_j] = True

# Footprint statistics
N_inside = mask_inside_any.sum()
frac_inside = 100.0 * N_inside / N_jades

print("\n=== GOODS-S footprint statistics (3D) ===")
print(f"Total JADES objects.............: {N_jades}")
print(f"JADES inside GOODS-S mask.......: {N_inside}")
print(f"Fraction (%)...................: {frac_inside:.2f}%")

# ==========================
# CLASSIFICATION INTO 3 REGIMES
# ==========================
print("\n=== Classifying LOS 3D into three regimes ===")

# Consider only JADES objects inside the mask
valid_counts = LOS3D_count[mask_inside_any]

if len(valid_counts) > 0:
    # Quantiles for tercile classification (low / intermediate / high density)
    q1, q2 = np.quantile(valid_counts, [1/3, 2/3])
    print(
        f"LOS3D_count quantiles (GOODS-S JADES): "
        f"Q1={q1:.2f}, Q2={q2:.2f}"
    )

    # Assign classes:
    # 0 = low density (<= Q1)
    # 1 = intermediate density (Q1 < count <= Q2)
    # 2 = high density (> Q2)
    for i in range(N_jades):
        if not mask_inside_any[i]:
            continue
        c = LOS3D_count[i]
        if c <= q1:
            LOS3D_class[i] = 0
        elif c <= q2:
            LOS3D_class[i] = 1
        else:
            LOS3D_class[i] = 2
else:
    print(
        "No JADES objects inside the mask with valid LOS3D values; "
        "classes remain set to -1."
    )

# ==========================
# ADD COLUMNS TO JADES CATALOG
# ==========================
print("\n=== Adding LOS3D columns to JADES catalog ===")

# Remove old columns if they already exist
for colname in ["LOS3D_count", "LOS3D_class", "LOS3D_field"]:
    if colname in jades_tab.colnames:
        jades_tab.remove_column(colname)

jades_tab["LOS3D_count"] = LOS3D_count
jades_tab["LOS3D_class"] = LOS3D_class
jades_tab["LOS3D_field"] = LOS3D_field

out_file = "jades_GOODSS_LOS