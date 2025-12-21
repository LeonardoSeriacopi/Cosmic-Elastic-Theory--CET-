from astropy.io import fits
from astropy.table import Table
import numpy as np

infile  = "jades_GOODSS_LOS3D_distnorm.fits"
outfile = "jades_GOODSS_LOS3D_distnorm_clean.fits"

# --- Load input catalog ---
with fits.open(infile) as hdul:
    tab = Table(hdul[1].data)

print(f"Original catalog: {len(tab)} objects")

# --- Basic quality filters ---
z = np.array(tab["z_best"])
cls = np.array(tab["LOS_env_class"])

mask = (
    np.isfinite(z) &
    (z > 0) &
    (cls >= 0)      # remove invalid environment class (-1)
)

tab_clean = tab[mask]
print(f"After filtering (z > 0 and LOS_env_class >= 0): {len(tab_clean)} objects")

# --- Safe computation of log10(LOS_env_count) ---
env_cnt = np.array(tab_clean["LOS_env_count"])
env_cnt_safe = np.where(env_cnt > 0, env_cnt, 1.0)
tab_clean["log_LOS_env_count"] = np.log10(env_cnt_safe)

# --- Quick statistics ---
u_cls, n_cls = np.unique(tab_clean["LOS_env_class"], return_counts=True)
print("\nEnvironment class distribution (clean sample):")
for c, n in zip(u_cls, n_cls):
    print(f"  class {c}: {n} objects")

# --- Save final working catalog ---
tab_clean.write(outfile, overwrite=True)
print(f"\nFinal working catalog saved to: {outfile}")