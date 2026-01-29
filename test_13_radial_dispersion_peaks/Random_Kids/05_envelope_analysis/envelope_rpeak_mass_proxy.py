import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# carregar dados
df = pd.read_csv("anchor_out_merged_with_proxies.csv")

x = df["Z_center"].values
y = df["r_peak_Mpc"].values

# máscara de valores válidos
mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
x = x[mask]
y = y[mask]

# bins em quantis de Z
nbins = 8
bins = np.quantile(x, np.linspace(0, 1, nbins + 1))

xmid = []
q50, q84, q95, q99 = [], [], [], []
counts = []

for i in range(nbins):
    lo, hi = bins[i], bins[i + 1]

    if i == nbins - 1:
        m = (x >= lo) & (x <= hi)
    else:
        m = (x >= lo) & (x < hi)

    yy = y[m]
    if len(yy) < 10:
        continue

    xmid.append(np.median(x[m]))
    counts.append(len(yy))

    q50.append(np.quantile(yy, 0.50))
    q84.append(np.quantile(yy, 0.84))
    q95.append(np.quantile(yy, 0.95))
    q99.append(np.quantile(yy, 0.99))

xmid = np.array(xmid)
q50 = np.array(q50)
q84 = np.array(q84)
q95 = np.array(q95)
q99 = np.array(q99)

# plot (intencionalmente sem graça)
plt.figure(figsize=(6.2, 4.0))

plt.scatter(x, y, s=10, color="0.7", alpha=0.35, label="Data")

plt.plot(xmid, q50, color="black", lw=1.8, ls="--", label="Median (50%)")
plt.plot(xmid, q84, color="black", lw=1.2, label="84% envelope")
plt.plot(xmid, q95, color="black", lw=1.0, alpha=0.8, label="95% envelope")
plt.plot(xmid, q99, color="black", lw=0.8, alpha=0.6, label="99% envelope")

plt.xlabel("Mass proxy (Z)")
plt.ylabel(r"$r_{\mathrm{peak}}\ \mathrm{[Mpc]}$")
plt.title("Envelope analysis: $r_{\\mathrm{peak}}$ vs mass proxy (non-parametric)")

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("fig_envelope_vs_z.png", dpi=220)
plt.close()

print("Bin counts:", counts)
print("Saved fig_envelope_vs_z.png")