import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================================
# CONFIG
# ============================================================

INPUT_FILE = "consolidated_relic_sample.csv"
FIG_DIR    = "figures_results"

os.makedirs(FIG_DIR, exist_ok=True)

sns.set_style("whitegrid")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13
})

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)
df.columns = [c.strip() for c in df.columns]

print(f"[OK] Loaded {len(df)} relic galaxies")
print(df[["ID","dist_pico_mpc","limite_transbordo_mpc","Idade Estrutural (logρ)"]])

# ============================================================
# FIGURE 1 — Peak Distance vs Structural Age
# ============================================================

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=df,
    x="Idade Estrutural (logρ)",
    y="dist_pico_mpc",
    s=90,
    color="black"
)

plt.xlabel("Structural Age (log ρ)")
plt.ylabel("Peak Distance [Mpc]")
plt.title("Peak Distance vs Structural Age")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig1_peak_vs_structural_age.png", dpi=300)
plt.close()

# ============================================================
# FIGURE 2 — Peak Distance vs Redshift (Control Test)
# ============================================================

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=df,
    x="Redshift (z)",
    y="dist_pico_mpc",
    s=90,
    color="black"
)

plt.xlabel("Redshift (z)")
plt.ylabel("Peak Distance [Mpc]")
plt.title("Peak Distance vs Redshift")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig2_peak_vs_redshift.png", dpi=300)
plt.close()

# ============================================================
# FIGURE 3 — Peak Distance vs Transition Boundary
# ============================================================

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=df,
    x="dist_pico_mpc",
    y="limite_transbordo_mpc",
    s=90,
    color="black"
)

max_val = np.nanmax(df[["dist_pico_mpc","limite_transbordo_mpc"]].values)
plt.plot([0, max_val], [0, max_val], ls="--", color="gray", lw=1)

plt.xlabel("Peak Distance [Mpc]")
plt.ylabel("Transition Boundary [Mpc]")
plt.title("Peak Distance vs Transition Boundary")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig3_peak_vs_boundary.png", dpi=300)
plt.close()

# ============================================================
# FIGURE 4 — Dissipation Gap vs Peak Distance
# ============================================================

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=df,
    x="dist_pico_mpc",
    y="gap_pico_transbordo",
    s=90,
    color="black"
)

plt.xlabel("Peak Distance [Mpc]")
plt.ylabel("Boundary − Peak [Mpc]")
plt.title("Dissipation Gap vs Peak Distance")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig4_gap_vs_peak.png", dpi=300)
plt.close()

# ============================================================
# FIGURE 5 — Peak Distance vs Background Dispersion
# ============================================================

plt.figure(figsize=(7,6))
sns.scatterplot(
    data=df,
    x="mean_dispersion_rms",
    y="dist_pico_mpc",
    s=90,
    color="black"
)

plt.xlabel("Mean Background Dispersion (RMS)")
plt.ylabel("Peak Distance [Mpc]")
plt.title("Peak Distance vs Background Dispersion")

plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig5_peak_vs_background.png", dpi=300)
plt.close()

# ============================================================
# FIGURE 6 — Correlation Matrix
# ============================================================

corr_cols = [
    "dist_pico_mpc",
    "limite_transbordo_mpc",
    "gap_pico_transbordo",
    "Idade Estrutural (logρ)",
    "logM∗",
    "Re (kpc)",
    "σ (km/s)",
    "Redshift (z)",
    "mean_dispersion_rms"
]

corr = df[corr_cols].corr()

plt.figure(figsize=(10,8))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar_kws={"label": "Correlation"}
)

plt.title("Correlation Matrix — Relic Galaxy Sample")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/fig6_correlation_matrix.png", dpi=300)
plt.close()

print("[DONE] All Paper 5 consolidated figures generated.")