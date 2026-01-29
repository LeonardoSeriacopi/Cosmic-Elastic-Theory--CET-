# sample_kids_10k.py
import numpy as np
from astropy.io import fits
import pandas as pd

KIDS_FITS = "kids_dr4.fits"
N_SAMPLE = 10_000
OUTPUT = "kids_10k_centers.csv"

print("Abrindo KiDS (memmap)...")
with fits.open(KIDS_FITS, memmap=True) as hdul:
    data = hdul[1].data
    n_total = len(data)

    print(f"Total de objetos: {n_total}")
    idx = np.random.choice(n_total, size=N_SAMPLE, replace=False)

    sample = pd.DataFrame({
        "ID": data["ID"][idx],
        "RA": data["RAJ2000"][idx],
        "DEC": data["DECJ2000"][idx],
        "Z": data["Z_B"][idx],
        "e1": data["e1"][idx],
        "e2": data["e2"][idx],
        "weight": data["weight"][idx],
    })

sample.to_csv(OUTPUT, index=False)
print(f"Salvo: {OUTPUT}")