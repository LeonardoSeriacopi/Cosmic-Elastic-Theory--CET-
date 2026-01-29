# peak_dispersion_profile.py
import numpy as np
import pandas as pd
from astropy.cosmology import Planck18 as cosmo
from astropy.coordinates import SkyCoord
import astropy.units as u

CENTERS = "kids_10k_centers.csv"
KIDS_FITS = "kids_dr4.fits"
OUTPUT = "kids_dispersion_peaks.csv"

R_MAX = 5.0   # Mpc
NBINS = 25

centers = pd.read_csv(CENTERS)

print("Abrindo KiDS completo...")
from astropy.io import fits
with fits.open(KIDS_FITS, memmap=True) as hdul:
    kids = hdul[1].data
    kids_coord = SkyCoord(
        kids["RAJ2000"] * u.deg,
        kids["DECJ2000"] * u.deg
    )
    e_mod = np.sqrt(kids["e1"]**2 + kids["e2"]**2)

results = []

for i, row in centers.iterrows():
    z = row["Z"]
    if z <= 0:
        continue

    center = SkyCoord(row["RA"] * u.deg, row["DEC"] * u.deg)
    d_ang = center.separation(kids_coord).to(u.rad).value

    DA = cosmo.angular_diameter_distance(z).value  # Mpc
    r = d_ang * DA

    mask = r < R_MAX
    if mask.sum() < 50:
        continue

    r_sel = r[mask]
    e_sel = e_mod[mask]

    bins = np.linspace(0, R_MAX, NBINS)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])

    prof = []
    for j in range(len(bins)-1):
        m = (r_sel >= bins[j]) & (r_sel < bins[j+1])
        prof.append(np.mean(e_sel[m]) if m.sum() > 5 else np.nan)

    prof = np.array(prof)
    if np.all(np.isnan(prof)):
        continue

    peak_idx = np.nanargmax(prof)
    r_peak = bin_centers[peak_idx]

    results.append({
        "ID": row["ID"],
        "Z": z,
        "r_peak_Mpc": r_peak,
    })

    if i % 100 == 0:
        print(f"{i}/{len(centers)}")

pd.DataFrame(results).to_csv(OUTPUT, index=False)
print(f"Salvo: {OUTPUT}")