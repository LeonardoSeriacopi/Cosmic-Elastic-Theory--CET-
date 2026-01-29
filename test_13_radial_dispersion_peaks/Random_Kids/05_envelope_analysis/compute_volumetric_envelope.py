import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("anchor_out_merged_with_proxies.csv")
x = df["Z_center"].values
y = df["r_peak_Mpc"].values

mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
x = x[mask]; y = y[mask]

# bins em Z
nbins = 8
bins = np.quantile(x, np.linspace(0, 1, nbins+1))

xmid = []
q50 = []; q84 = []; q95 = []; q99 = []
counts = []

for i in range(nbins):
    lo, hi = bins[i], bins[i+1]
    m = (x >= lo) & (x <= hi if i == nbins-1 else x < hi)
    yy = y[m]
    if len(yy) < 10:
        continue
    xm = np.median(x[m])
    xmid.append(xm)
    counts.append(len(yy))
    q50.append(np.quantile(yy, 0.50))
    q84.append(np.quantile(yy, 0.84))
    q95.append(np.quantile(yy, 0.95))
    q99.append(np.quantile(yy, 0.99))

xmid = np.array(xmid)
q50 = np.array(q50); q84=np.array(q84); q95=np.array(q95); q99=np.array(q99)

plt.figure(figsize=(6.8,4.2))
plt.scatter(x, y, s=14, alpha=0.25, edgecolor="none", label="Data")
plt.plot(xmid, q50, linewidth=2, label="Median (50%)")
plt.plot(xmid, q84, linewidth=2, label="84% envelope")
plt.plot(xmid, q95, linewidth=2, label="95% envelope")
plt.plot(xmid, q99, linewidth=2, label="99% envelope")
plt.xlabel("Mass proxy (Z)")
plt.ylabel("r_peak [Mpc]")
plt.title("Envelope analysis: r_peak scale vs mass proxy (no parametric fit)")
plt.legend()
plt.tight_layout()
plt.savefig("fig_envelope_vs_z.png", dpi=220)
plt.close()

print("Bin counts:", counts)
print("Saved: fig_envelope_vs_z.png")