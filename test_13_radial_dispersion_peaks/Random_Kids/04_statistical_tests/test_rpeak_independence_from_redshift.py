import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

df = pd.read_csv("anchor_out_merged_with_proxies.csv")

x = df["Z_center"].values
y = df["r_peak_Mpc"].values

mask = np.isfinite(x) & np.isfinite(y)
x = x[mask]
y = y[mask]

# Spearman
rho, pval = spearmanr(x, y)

# OLS (apenas como guia visual)
coef = np.polyfit(x, y, 1)
xx = np.linspace(x.min(), x.max(), 200)
yy = coef[0] * xx + coef[1]

plt.figure(figsize=(6.8, 4.2))
plt.scatter(x, y, s=14, alpha=0.25, edgecolor="none", label="Data")

plt.plot(
    xx, yy,
    color="#6a0dad",  
    linewidth=2.5,
    label=f"OLS slope = {coef[0]:.3f} Mpc per Z"
)

plt.xlabel("Mass proxy (Z)")
plt.ylabel("r_peak [Mpc]")
plt.title(f"r_peak vs Z (Spearman ρ = {rho:.3f}, p = {pval:.1e})")
plt.legend()
plt.tight_layout()
plt.savefig("fig_rpeak_vs_z_spearman.png", dpi=220)
plt.close()