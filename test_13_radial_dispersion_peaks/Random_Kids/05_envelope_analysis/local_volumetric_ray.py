import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
import astropy.units as u

# ==============================
# CONFIG
# ==============================
INPUT = "anchor_out_merged_with_proxies.csv"
OUTPUT = "cet_environment_classification.csv"

r_mode = 0.3125        # Mpc (mode found in data)
R_CET = 2 * r_mode     # Mpc
DZ_TOL = 0.03          # redshift tolerance

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv(INPUT)

coords = SkyCoord(
    ra=df["RA"].values * u.deg,
    dec=df["DEC"].values * u.deg
)

z = df["Z_center"].values

# ==============================
# COMPUTE ANGULAR SCALE
# ==============================
# Angular diameter distance per object
DA = cosmo.angular_diameter_distance(z).to(u.Mpc).value

# Angular radius corresponding to CET scale
theta_cet = (R_CET / DA) * u.rad

# ==============================
# ENVIRONMENT CLASSIFICATION
# ==============================
env_flag = []

for i in range(len(df)):
    sep = coords[i].separation(coords)
    dz = np.abs(z - z[i])

    neighbors = (
        (sep < theta_cet[i]) &
        (dz < DZ_TOL) &
        (dz > 0)
    )

    if neighbors.sum() > 0:
        env_flag.append("clustered")
    else:
        env_flag.append("isolated")

df["environment"] = env_flag

df.to_csv(OUTPUT, index=False)

print("Saved:", OUTPUT)
print(df["environment"].value_counts(normalize=True))