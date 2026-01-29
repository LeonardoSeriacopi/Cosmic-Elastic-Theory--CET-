import pandas as pd
import numpy as np

df = pd.read_csv("anchor_out_merged_with_proxies.csv")
r = df["r_peak_Mpc"].dropna().values

modes = []

for _ in range(1000):
    sample = np.random.choice(r, size=len(r), replace=True)
    hist, bins = np.histogram(sample, bins=25)
    mode = 0.5 * (bins[np.argmax(hist)] + bins[np.argmax(hist)+1])
    modes.append(mode)

print(f"Bootstrap mode = {np.mean(modes):.3f} ± {np.std(modes):.3f} Mpc")