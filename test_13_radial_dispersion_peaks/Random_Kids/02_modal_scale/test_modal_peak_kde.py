import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("anchor_out_merged_with_proxies.csv")

r = df["r_peak_Mpc"].dropna().values

# KDE
kde = gaussian_kde(r, bw_method="scott")
x = np.linspace(r.min(), r.max(), 500)
y = kde(x)

# Mode
r_mode = x[np.argmax(y)]

print(f"Global modal r_peak = {r_mode:.3f} Mpc")

# Plot
plt.figure(figsize=(6,4))
plt.hist(r, bins=25, density=True, alpha=0.4, label="Data")
plt.plot(x, y, lw=2, label="KDE")
plt.axvline(r_mode, ls="--", label=f"Mode = {r_mode:.2f} Mpc")

plt.xlabel("r_peak [Mpc]")
plt.ylabel("Probability density")
plt.title("Global dispersion peak distribution")
plt.legend()
plt.tight_layout()
plt.savefig("fig_modal_peak_kde.png", dpi=200)
plt.close()