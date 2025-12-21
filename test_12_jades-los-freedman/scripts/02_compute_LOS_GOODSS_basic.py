import numpy as np
from astropy.table import Table
from astropy.io import fits
from matplotlib.path import Path
import csv

# ==========================
# INPUT FILES
# ==========================
jades_file = "jades_preprocessed.fits"

candels_s_file = "hlsp_candels_hst_wfc3_goodss-tot-multiband_f160w_v1_cat.fits"
mask_s_csv = "goods_s_clean.csv"

out_file = "jades_GOODSS_LOS.fits"

# ==========================
# AUXILIARY FUNCTIONS
# ==========================

def load_mask_csv(mask_file):
    """
    Load GOODS-S sky masks from a CSV file.
    Each mask is defined by four (RA, Dec) vertices.
    """
    polys = []
    with open(mask_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            ra = [float(row[f"ra{i}"]) for i in range(1, 5)]
            dec = [float(row[f"dec{i}"]) for i in range(1, 5)]
            polys.append({
                "label": row.get("label", ""),
                "ra": ra,
                "dec": dec
            })
    return polys


def points_in_polygon(ra_pts, dec_pts, ra_poly, dec_poly):
    """
    Point-in-polygon test using matplotlib Path.
    """
    verts = np.column_stack([ra_poly, dec_poly])
    path = Path(verts, closed=True)
    pts = np.column_stack([ra_pts, dec_pts])
    return path.contains_points(pts)


def compute_env_per_polygon(candels_tab, mask_polys):
    """
    Count how many CANDELS galaxies fall inside each GOODS-S polygon.
    """
    ra_c = np.array(candels_tab["RA"])
    dec_c = np.array(candels_tab["DEC"])

    env_dict = {}

    print(f"Number of CANDELS objects in GOODS-S field: {len(ra_c)}")

    for i, poly in enumerate(mask_polys):
        label = poly["label"] or f"poly_{i}"
        inside = points_in_polygon(ra_c, dec_c, poly["ra"], poly["dec"])
        count = np.sum(inside)
        env_dict[label] = count
        print(f"  Polygon {i:02d} ({label}): {count} CANDELS galaxies")

    return env_dict


def assign_env_to_jades(jades_tab, mask_polys, env_dict):
    """
    Assign CANDELS-based LOS environment counts to JADES objects
    that fall inside the GOODS-S masks.
    """
    ra_j = np.array(jades_tab["RA_TARG"])
    dec_j = np.array(jades_tab["Dec_TARG"])

    env = np.full(len(jades_tab), np.nan)

    print("\nAssigning LOS environments to JADES GOODS-S sources...")

    for i, poly in enumerate(mask_polys):
        label = poly["label"] or f"poly_{i}"
        inside = points_in_polygon(ra_j, dec_j, poly["ra"], poly["dec"])
        n_in = np.sum(inside)

        if n_in > 0:
            val = env_dict[label]
            env[inside] = val
            print(f"  Polygon {i:02d} ({label}): {n_in} JADES → env={val}")

    return env


# ==========================
# PIPELINE
# ==========================

print("=== Loading JADES catalog ===")
jades = Table.read(jades_file)
print(f"Total JADES objects: {len(jades)}")

# Select GOODS-S only (Dec < 0)
mask_S = jades["Dec_TARG"] < 0
jades_s = jades[mask_S]
print(f"JADES objects in GOODS-S: {np.sum(mask_S)}")

# --------------------------
# LOAD MASKS AND CANDELS
# --------------------------

mask_polys_s = load_mask_csv(mask_s_csv)
candels_s = Table.read(candels_s_file)

# --------------------------
# COMPUTE ENVIRONMENT PER POLYGON
# --------------------------

env_s = compute_env_per_polygon(candels_s, mask_polys_s)

# --------------------------
# ASSIGN ENVIRONMENT TO JADES
# --------------------------

env_count_s = assign_env_to_jades(jades_s, mask_polys_s, env_s)

# Global arrays (full JADES length)
LOS_env_count = np.full(len(jades), np.nan)
LOS_env_field = np.full(len(jades), -1, dtype=int)

LOS_env_count[mask_S] = env_count_s
LOS_env_field[mask_S] = 1  # 1 = GOODS-S field

# --------------------------
# ENVIRONMENT CLASSES
# --------------------------

vals = env_count_s[np.isfinite(env_count_s)]

if len(vals) > 0:
    q1 = np.quantile(vals, 1 / 3)
    q2 = np.quantile(vals, 2 / 3)
else:
    q1 = q2 = np.nan

print("\nLOS_env_count quantiles (GOODS-S):")
print(f"  Q1 = {q1}")
print(f"  Q2 = {q2}")

LOS_env_class = np.full(len(jades), -1, dtype=int)

if np.isfinite(q1) and np.isfinite(q2):
    LOS_env_class[(LOS_env_count <= q1) & (LOS_env_field == 1)] = 0
    LOS_env_class[(LOS_env_count > q1) & (LOS_env_count <= q2) & (LOS_env_field == 1)] = 1
    LOS_env_class[(LOS_env_count > q2) & (LOS_env_field == 1)] = 2

# --------------------------
# SAVE FINAL CATALOG
# --------------------------

print("\n=== Saving final catalog ===")

jades["LOS_env_count"] = LOS_env_count
jades["LOS_env_class"] = LOS_env_class
jades["LOS_env_field"] = LOS_env_field

jades.write(out_file, overwrite=True)

print(f"\nCatalog saved to: {out_file}")
print("Done! GOODS-S only.\n")