import numpy as np
from astropy.table import Table

# Name of the CANDELS GOODS-S redshift text catalog
txt_file = "hlsp_candels_hst_wfc3_goodss_multi_v2_redshift-cat.txt"

print(f"Reading text catalog: {txt_file}")

# Read file while ignoring comment lines, as a structured array
data = np.genfromtxt(
    txt_file,
    dtype=None,
    encoding=None,
    comments="#",
    invalid_raise=False
)

print("Number of rows read:", data.shape[0])
print("Number of detected columns:", len(data[0]))

# In structured arrays, columns are accessed as fields: f0, f1, f2, ...
# Zero-based indices according to the catalog documentation:
# 0: filename
# 1: ID
# 2: RA
# 3: DEC
# 4: z_best
# 5: z_best_type
# 6: z_spec
# 7: z_spec_ref
# 8: z_grism

filename_col = data["f0"]
ID_col       = data["f1"].astype(int)
RA_col       = data["f2"].astype(float)
DEC_col      = data["f3"].astype(float)
z_best       = data["f4"].astype(float)
z_spec       = data["f6"].astype(float)
z_grism      = data["f8"].astype(float)

# Build final redshift with priority: z_spec > z_grism > z_best
z_final = np.where(
    z_spec > 0, z_spec,
    np.where(z_grism > 0, z_grism, z_best)
)

print("\nExamples (RA, DEC, z_best, z_spec, z_grism, z_final):")
for i in range(5):
    print(RA_col[i], DEC_col[i], z_best[i], z_spec[i], z_grism[i], z_final[i])

# Build ASTROPY table
tab = Table(
    [ID_col, RA_col, DEC_col, z_final],
    names=["ID", "RA", "DEC", "z"]
)

out_fits = "candels_GOODSS_3D.fits"
tab.write(out_fits, overwrite=True)
print(f"\nFITS file saved to: {out_fits}")