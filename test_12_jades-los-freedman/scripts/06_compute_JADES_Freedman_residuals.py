# jades_prep_residuals_GOODSS.py
from astropy.io import fits
from astropy.table import Table
import numpy as np

# === 1. Read JADES + LOS (GOODS-S) catalog ===
jades_file = "jades_GOODSS_LOS.fits"
with fits.open(jades_file) as hdul:
    tab = Table(hdul[1].data)

print(f"Total rows in catalog: {len(tab)}")
print("Available columns:")
print(tab.colnames)

# === 2. Define exact FITS column names ===
col_z    = "z_best"
col_envc = "LOS_env_count"
col_envk = "LOS_env_class"

# === 3. Basic filter: valid redshift ===
mask_good = np.isfinite(tab[col_z]) & (tab[col_z] > 0)
tab = tab[mask_good]
print(f"After filtering valid z_best: {len(tab)} objects")

# === 4. Create log10(LOS_env_count) ===
env_raw = np.array(tab[col_envc])
env_raw_safe = np.where(env_raw > 0, env_raw, 1.0)
tab["log_LOS_env_count"] = np.log10(env_raw_safe)

# === 5. Class distribution ===
unique_classes, counts = np.unique(tab[col_envk], return_counts=True)
print("\nDistribution of LOS_env_class:")
for c, n in zip(unique_classes, counts):
    print(f"  class {c}: {n} objects")

# === 6. Save filtered working catalog ===
out_file = "jades_GOODSS_env_work.fits"
tab.write(out_file, overwrite=True)
print(f"\nWorking catalog saved to: {out_file}")