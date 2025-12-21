import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("z_over_dproxy_by_bin.csv")

# Plot with error bars
plt.figure(figsize=(7,5))
plt.errorbar(
    df["z_mean"],
    df["z_over_Dproxy"],
    yerr=df["SEM"],
    fmt='o',
    capsize=3
)

plt.xlabel(r"$\langle z \rangle$")
plt.ylabel(r"$\langle z / D_{\mathrm{proxy}} \rangle$")
plt.title("Model-independent redshift scaling (JADES)")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()