import pandas as pd
import numpy as np

df = pd.read_csv("anchor_out_merged_with_proxies.csv")

df["ellipticity"] = np.sqrt(df["e1"]**2 + df["e2"]**2)

# Split by ellipticity
low = df[df["ellipticity"] < df["ellipticity"].median()]
high = df[df["ellipticity"] >= df["ellipticity"].median()]

def modal(r):
    hist, bins = np.histogram(r, bins=20)
    return 0.5 * (bins[np.argmax(hist)] + bins[np.argmax(hist)+1])

print("Low ellipticity mode :", modal(low["r_peak_Mpc"]))
print("High ellipticity mode:", modal(high["r_peak_Mpc"]))