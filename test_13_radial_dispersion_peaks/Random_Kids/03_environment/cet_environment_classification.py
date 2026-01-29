import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

df = pd.read_csv("cet_environment_classification.csv")

def mode_kde(x):
    kde = gaussian_kde(x)
    xx = np.linspace(x.min(), x.max(), 500)
    return xx[np.argmax(kde(xx))]

isolated = df[df["environment"] == "isolated"]["r_peak_Mpc"].values
clustered = df[df["environment"] == "clustered"]["r_peak_Mpc"].values

mode_iso = mode_kde(isolated)
mode_clu = mode_kde(clustered)

print("Mode (isolated):", mode_iso)
print("Mode (clustered):", mode_clu)

# ==============================
# PLOT
# ==============================
plt.figure(figsize=(6,4))

plt.hist(isolated, bins=20, density=True, alpha=0.6, label="Isolated")
plt.hist(clustered, bins=20, density=True, alpha=0.6, label="Clustered")

plt.axvline(mode_iso, color="blue", lw=2, linestyle="--")
plt.axvline(mode_clu, color="red", lw=2, linestyle="--")

plt.xlabel("Dispersion peak radius r_peak [Mpc]")
plt.ylabel("Density")
plt.title("Local volumetric overflow scale by environment")
plt.legend()

plt.tight_layout()
plt.savefig("fig_environmental_modes.png", dpi=200)
plt.close()