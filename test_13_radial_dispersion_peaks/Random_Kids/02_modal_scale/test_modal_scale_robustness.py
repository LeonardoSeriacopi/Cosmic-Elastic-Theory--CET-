import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def kde_gaussian(xgrid, data, bw):
    # KDE Gaussiana simples, sem libs externas
    data = data[:, None]
    xg = xgrid[None, :]
    z = (xg - data) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=0) / (len(data) * bw * np.sqrt(2*np.pi))
    return dens

df = pd.read_csv("anchor_out_merged_with_proxies.csv")

r = df["r_peak_Mpc"].dropna().values
r = r[(r > 0) & np.isfinite(r)]

# grid para KDE
xgrid = np.linspace(np.percentile(r, 0.5), np.percentile(r, 99.5), 800)

# bandwidth (Silverman)
sigma = np.std(r, ddof=1)
bw = 1.06 * sigma * (len(r) ** (-1/5))
bw = max(bw, 1e-4)

# moda do sample completo
dens = kde_gaussian(xgrid, r, bw)
mode_full = xgrid[np.argmax(dens)]

# bootstrap
B = 2000
modes = np.empty(B)
rng = np.random.default_rng(123)

for i in range(B):
    sample = rng.choice(r, size=len(r), replace=True)
    dens_b = kde_gaussian(xgrid, sample, bw)
    modes[i] = xgrid[np.argmax(dens_b)]

ci16, ci50, ci84 = np.percentile(modes, [16, 50, 84])

# plot
plt.figure(figsize=(7,4))
plt.hist(modes, bins=40, alpha=0.7, edgecolor="black")
plt.axvline(mode_full, linewidth=2, label=f"Mode (full) = {mode_full:.3f} Mpc")
plt.axvline(ci16, linestyle="--", linewidth=2, label=f"16th = {ci16:.3f}")
plt.axvline(ci50, linestyle="--", linewidth=2, label=f"50th = {ci50:.3f}")
plt.axvline(ci84, linestyle="--", linewidth=2, label=f"84th = {ci84:.3f}")

plt.xlabel("Bootstrap mode of r_peak [Mpc]")
plt.ylabel("Count")
plt.title("Robustness of the modal dispersion-peak scale (bootstrap KDE)")
plt.legend()
plt.tight_layout()
plt.savefig("fig_bootstrap_mode_kde.png", dpi=220)
plt.close()

print("Bandwidth:", bw)
print("Mode(full):", mode_full)
print("Bootstrap percentiles (16,50,84):", (ci16, ci50, ci84))