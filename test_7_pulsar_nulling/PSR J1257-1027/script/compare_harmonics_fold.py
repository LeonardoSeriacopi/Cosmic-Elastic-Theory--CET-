import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# CET — Harmonic vs Orbital Folding Comparison (Vertical Layout)
# ============================================================

# Input file from blind validation
df = pd.read_csv("phase_intensity_blind.csv")

# Periods (from blind + validation results)
period_short = 0.01053  # 15.2 min
period_long = 0.042     # 1.01 h (approx, adjust if known better)

# Normalize time in days
t = df["MJD"].values
y = df["intensity"].values

# ============================================================
# Folding helper function
# ============================================================
def compute_folded_mean(time, intensity, period_days, bins=50):
    """Compute mean intensity per phase bin."""
    phase = ((time / period_days) % 1)
    phase_bins = np.linspace(0, 1, bins + 1)
    mean_intensity = np.array([
        intensity[(phase >= phase_bins[i]) & (phase < phase_bins[i + 1])].mean()
        for i in range(bins)
    ])
    return phase_bins[:-1], mean_intensity

# Compute short and long foldings
phase_short, mean_intensity_short = compute_folded_mean(t, y, period_short)
phase_long, mean_intensity_long = compute_folded_mean(t, y, period_long)

# ============================================================
# Create output directory
# ============================================================
output_dir = "results_CET_blind_validation"
os.makedirs(output_dir, exist_ok=True)

# ============================================================
# Plotting: vertical panels + overlay
# ============================================================
fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=False)

# Panel 1 — Short folding
axes[0].plot(phase_short, mean_intensity_short, color="steelblue")
axes[0].set_title("Short Folding (15.2 min) — Harmonic Modulation")
axes[0].set_ylabel("Mean Intensity")

# Panel 2 — Long folding
axes[1].plot(phase_long, mean_intensity_long, color="firebrick")
axes[1].set_title("Long Folding (1.01 h) — Orbital Envelope")
axes[1].set_ylabel("Mean Intensity")

# ============================================================
# Overlay normalized comparison (Panel 3)
# ============================================================

# Normalize intensities 0–1
norm_short = (mean_intensity_short - np.nanmin(mean_intensity_short)) / (
    np.nanmax(mean_intensity_short) - np.nanmin(mean_intensity_short)
)
norm_long = (mean_intensity_long - np.nanmin(mean_intensity_long)) / (
    np.nanmax(mean_intensity_long) - np.nanmin(mean_intensity_long)
)

# Normalize phase to 0–1 for overlay
phase_short_norm = np.linspace(0, 1, len(norm_short))
phase_long_norm = np.linspace(0, 1, len(norm_long))

axes[2].plot(phase_short_norm, norm_short, color="blue", label="15.2 min (normalized)")
axes[2].plot(phase_long_norm, norm_long, color="red", label="1.01 h (normalized)")
axes[2].set_title("CET — Hierarchical Coherence (Overlay of Foldings)")
axes[2].set_xlabel("Orbital Phase (folded)")
axes[2].set_ylabel("Normalized Intensity")
axes[2].legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "compare_folding_vertical_fixed.png"), dpi=300)
plt.show()

print(f"✅ Saved: {os.path.join(output_dir, 'compare_folding_vertical_fixed.png')}")