# plot_rpeak_regime_occupancy.py
# ============================================================
# CET — r_peak Regime Occupancy
# Full CET mode | No statistics | No model fitting
# Input: occupancy_physical.csv
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIG
# ============================================================
INPUT_FILE = "occupancy_physical.csv"
OUT_FIG = "fig_rpeak_regime_occupancy.png"

REGIMES = [
    ("frac_r_lt_0p3", "< 0.3 Mpc"),
    ("frac_0p3_1p0", "0.3 – 1.0 Mpc"),
    ("frac_1p0_2p5", "1.0 – 2.5 Mpc"),
    ("frac_gt_2p5", "> 2.5 Mpc"),
]

COLORS = {
    "relic": "#1f77b4",   # blue
    "random": "#7f7f7f",  # gray
}

# ============================================================
# LOAD DATA
# ============================================================
df = pd.read_csv(INPUT_FILE)

required_cols = ["sample"] + [r[0] for r in REGIMES]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

# Index by sample
df = df.set_index("sample")

# ============================================================
# PREP DATA
# ============================================================
labels = [r[1] for r in REGIMES]
x = np.arange(len(REGIMES))
width = 0.38

relic_vals = [df.loc["relic", r[0]] for r in REGIMES]
random_vals = [df.loc["random", r[0]] for r in REGIMES]

# ============================================================
# PLOT
# ============================================================
plt.figure(figsize=(9, 5.5))

plt.bar(
    x - width/2,
    relic_vals,
    width,
    label="Relics",
    color=COLORS["relic"],
    alpha=0.9
)

plt.bar(
    x + width/2,
    random_vals,
    width,
    label="Random (KiDS)",
    color=COLORS["random"],
    alpha=0.8
)

# Axes & labels
plt.xticks(x, labels)
plt.ylabel("Fraction of objects")
plt.xlabel("r_peak regime")
plt.ylim(0, 1.0)

plt.title("Occupancy of r_peak regimes")

plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=220)
plt.close()

print("[OK] Saved:", OUT_FIG)