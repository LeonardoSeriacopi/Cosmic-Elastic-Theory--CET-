import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

df = pd.read_csv("anchor_out_merged_with_proxies.csv")

# Define mass proxy bins (quantiles)
df["Z_bin"] = pd.qcut(df["Z_center"], q=4)

print("Modal r_peak per Z bin:\n")

for zbin, sub in df.groupby("Z_bin"):
    r = sub["r_peak_Mpc"].dropna().values
    if len(r) < 10:
        continue

    kde = gaussian_kde(r)
    x = np.linspace(r.min(), r.max(), 400)
    y = kde(x)
    mode = x[np.argmax(y)]

    print(f"{zbin}: mode = {mode:.3f} Mpc (N={len(r)})")