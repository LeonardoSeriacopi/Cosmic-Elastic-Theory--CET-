import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- helper KDE mode ---
def kde_gaussian(xgrid, data, bw):
    data = data[:, None]
    xg = xgrid[None, :]
    z = (xg - data) / bw
    dens = np.exp(-0.5 * z**2).sum(axis=0) / (len(data) * bw * np.sqrt(2*np.pi))
    return dens

df = pd.read_csv("anchor_out_merged_with_proxies.csv")
r = df["r_peak_Mpc"].values
mask = np.isfinite(r) & (r > 0)
r = r[mask]

xgrid = np.linspace(np.percentile(r, 0.5), np.percentile(r, 99.5), 800)
sigma = np.std(r, ddof=1)
bw = 1.06 * sigma * (len(r) ** (-1/5))
bw = max(bw, 1e-4)

dens = kde_gaussian(xgrid, r, bw)
mode = xgrid[np.argmax(dens)]

k = 2.0
threshold = k * mode

df2 = df.copy()
df2["is_cluster_like"] = df2["r_peak_Mpc"] > threshold

n_all = len(df2)
n_cl = int(df2["is_cluster_like"].sum())
n_gal = n_all - n_cl

# plot distribution + threshold
r_all = df2["r_peak_Mpc"].dropna().values
r_gal = df2.loc[~df2["is_cluster_like"], "r_peak_Mpc"].dropna().values

plt.figure(figsize=(7,4))
plt.hist(r_all, bins=40, alpha=0.5, edgecolor="black", label="All")
plt.hist(r_gal, bins=40, alpha=0.7, edgecolor="black", label="After cluster-like removal")
plt.axvline(mode, linewidth=2, label=f"Mode = {mode:.3f} Mpc")
plt.axvline(threshold, linestyle="--", linewidth=2, label=f"Cluster threshold = {threshold:.3f} Mpc (k={k})")
plt.xlabel("r_peak [Mpc]")
plt.ylabel("Count")
plt.title("Effect of removing cluster-like objects (CET thresholding)")
plt.legend()
plt.tight_layout()
plt.savefig("fig_cluster_removal_effect.png", dpi=220)
plt.close()

df2.to_csv("anchor_out_with_cluster_flag.csv", index=False)

print("Mode:", mode)
print("Threshold:", threshold)
print("All:", n_all, "Cluster-like:", n_cl, "Non-cluster:", n_gal)
print("Saved: anchor_out_with_cluster_flag.csv")