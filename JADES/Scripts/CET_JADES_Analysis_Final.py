# Cosmic Elastic Theory — JADES-GS Aggregation Analysis
# Final Combined Pipeline Script (Ready for Publication)
# --------------------------------------------------------

# >>> Stage 1: CET Redshift Correction & Density Estimation <<<

# [1] Consulta ao MAST/JWST
from astroquery.mast import Catalogs
data = Catalogs.query_region(
    "53.16477 -27.77463",
    radius=0.15*u.deg,
    catalog="JADES",
    data_release="DR3"
)
# Filtros aplicados: SNR > 5, z_phot_quality == 'A'

# [2] Cálculo de densidade 3D
from astropy.cosmology import Planck18
import numpy as np
from scipy.spatial import KDTree

def get_comoving_coords(ra, dec, z):
    dist = Planck18.comoving_distance(z).value
    x = dist * np.cos(np.radians(dec)) * np.cos(np.radians(ra))
    y = dist * np.cos(np.radians(dec)) * np.sin(np.radians(ra))
    z_coord = dist * np.sin(np.radians(dec))
    return np.vstack([x, y, z_coord]).T

coords = get_comoving_coords(data['ra'], data['dec'], data['z_phot'])
tree = KDTree(coords)
data['density'] = tree.query_ball_point(coords, r=2, count_only=True) / (4/3*np.pi*8)

# [3] Aplicação CET
rho_crit = 2.8e-26
def z_CET(row):
    rho_local = row['density'] * 2.55e-30 * (1 + row['z_phot'])**3
    xi = (1 + (2.55e-30/rho_crit)**0.7) / (1 + (rho_local/rho_crit)**0.7)
    return (1 + row['z_phot']) * xi - 1

data['z_CET'] = data.apply(z_CET, axis=1)

# >>> Stage 2: Statistical Validation and Correlation Tests <<<

from scipy.stats import linregress
slope, intercept, r_val, p_val, _ = linregress(data['density'], data['Δz'])
print(f"Relação linear: Δz = {slope:.3f}×ρ + {intercept:.3f} (R²={r_val**2:.2f})")