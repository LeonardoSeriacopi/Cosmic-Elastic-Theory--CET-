import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("anchor_out_merged_with_proxies.csv")

x = df["Z_center"].values
y = df["r_peak_Mpc"].values

# Saturating model
def saturating(x, a, b):
    return a * np.tanh(b * x)

# Fit
popt, _ = curve_fit(saturating, x, y, maxfev=5000)

# Plot
xx = np.linspace(x.min(), x.max(), 300)
yy = saturating(xx, *popt)

plt.figure(figsize=(6,4))

# Data: light gray points
plt.scatter(
    x, y,
    s=25,
    alpha=0.6,
    color="gray",
    edgecolor="none",
    label="Data"
)

# Fit: red line
plt.plot(
    xx, yy,
    lw=2.5,
    color="crimson",
    label="Saturating fit"
)

plt.xlabel("Mass proxy (Z)")
plt.ylabel("r_peak [Mpc]")
plt.title("Saturation of the dispersion peak scale")
plt.legend()
plt.tight_layout()
plt.savefig("fig_saturation_test.png", dpi=200)
plt.close()

print("Saturation parameters:", popt)