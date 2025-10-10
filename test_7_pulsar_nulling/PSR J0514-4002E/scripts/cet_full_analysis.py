import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
from scipy.signal import find_peaks, savgol_filter
from statsmodels.stats.multitest import multipletests

# =========================================================
# Functions
# =========================================================

def kepler_phase_from_time(t, pb_days, t0_mjd, ecc=0.0, omega_deg=0.0):
    """Compute orbital phase [0,1) from time (MJD) with eccentricity correction."""
    t = np.asarray(t, dtype=float)
    M = 2.0*np.pi * ((t - t0_mjd)/pb_days - np.floor((t - t0_mjd)/pb_days))
    E = M.copy()
    for _ in range(30):
        dE = -(E - ecc*np.sin(E) - M) / (1 - ecc*np.cos(E))
        E += dE
        if np.max(np.abs(dE)) < 1e-12:
            break
    nu = 2.0*np.arctan2(np.sqrt(1+ecc)*np.sin(E/2.0), np.sqrt(1-ecc)*np.cos(E/2.0))
    phase = ((nu + np.deg2rad(omega_deg)) % (2*np.pi)) / (2*np.pi)
    return phase

def circular_window_variance(phase, y, centers, half_width=0.05, min_points=10):
    var = np.full(len(centers), np.nan)
    mean = np.full(len(centers), np.nan)
    count = np.zeros(len(centers), dtype=int)
    for i, c in enumerate(centers):
        d = np.abs(((phase - c + 0.5) % 1.0) - 0.5)
        mask = d <= half_width
        if np.sum(mask) >= min_points:
            yy = y[mask]
            mean[i] = np.nanmean(yy)
            var[i]  = np.nanvar(yy, ddof=1)
            count[i]= np.sum(mask)
    return mean, var, count

# =========================================================
# Parameters (edit here)
# =========================================================

input_csv = "catalogo_observacoes.csv"          # catalog of observations
subint_folder = "."                             # folder with *_subint.csv files
atnf_csv = "atnf_filtered_pilot-1.csv"          # orbital parameters
output_folder = "results_CET"
bins = 128
half_width = 0.05
min_points = 10
prominence = 0.5
smooth_curve = True

os.makedirs(output_folder, exist_ok=True)

# =========================================================
# Load orbital parameters
# =========================================================

atnf = pd.read_csv(atnf_csv)
PB_days = float(atnf["PB"].values[0])
T0 = float(atnf["T0"].values[0])
ECC = float(atnf["ECC"].values[0]) if "ECC" in atnf.columns else 0.0
print(f"Orbital params: PB={PB_days} days, T0={T0}, ECC={ECC}")

# =========================================================
# Combine all subints into one dataset
# =========================================================

catalog = pd.read_csv(input_csv)
all_data = []

for idx, row in catalog.iterrows():
    filename = row["filename"]
    mjd_start = row["MJD_start"]
    subint_file = os.path.join(subint_folder, filename.replace(".fits", "_subint.csv"))
    if not os.path.exists(subint_file):
        continue
    subint = pd.read_csv(subint_file)
    if "OFFS_SUB_s" not in subint.columns or "INTENSIDADE_MEDIA" not in subint.columns:
        continue
    times_mjd = mjd_start + subint["OFFS_SUB_s"] / 86400.0
    phases = kepler_phase_from_time(times_mjd, PB_days, T0, ecc=ECC)
    df_temp = pd.DataFrame({
        "phase": phases,
        "intensity": subint["INTENSIDADE_MEDIA"],
        "file": filename
    })
    all_data.append(df_temp)

if len(all_data) == 0:
    raise RuntimeError("No valid subint data found!")

data = pd.concat(all_data, ignore_index=True)
data.to_csv(os.path.join(output_folder, "phase_intensity.csv"), index=False)

# =========================================================
# Spearman and Kendall correlation
# =========================================================

rho, p_rho = spearmanr(data["phase"], data["intensity"], nan_policy="omit")
tau, p_tau = kendalltau(data["phase"], data["intensity"])
print(f"Spearman rho={rho:.3f}, p={p_rho:.2e}")
print(f"Kendall tau={tau:.3f}, p={p_tau:.2e}")

# =========================================================
# Variance-by-phase analysis
# =========================================================

centers = np.linspace(0, 1, bins, endpoint=False)
mean_curve, var_curve, counts = circular_window_variance(
    data["phase"].to_numpy(), data["intensity"].to_numpy(),
    centers, half_width=half_width, min_points=min_points
)

vis_curve = var_curve.copy()
if smooth_curve:
    win = max(5, int(0.1*bins) | 1)
    try:
        vis_curve = savgol_filter(np.nan_to_num(var_curve, nan=np.nanmedian(var_curve)),
                                  window_length=win, polyorder=2, mode='interp')
    except Exception:
        pass

peaks, props = find_peaks(np.nan_to_num(var_curve, nan=-np.inf), prominence=prominence)
peaks_df = pd.DataFrame({
    "phase_center": [centers[i] for i in peaks],
    "variance": [var_curve[i] for i in peaks],
    "prominence": [props["prominences"][k] for k in range(len(peaks))] if len(peaks)>0 else []
})
peaks_df.to_csv(os.path.join(output_folder, "variance_peaks.csv"), index=False)

variance_df = pd.DataFrame({
    "phase_center": centers,
    "mean_intensity": mean_curve,
    "variance": var_curve,
    "count": counts
})
variance_df.to_csv(os.path.join(output_folder, "variance_by_phase.csv"), index=False)

# =========================================================
# Plot 1 — Intensity vs Orbital Phase
# =========================================================

plt.figure(figsize=(9,4))
plt.scatter(data["phase"], data["intensity"], s=10, alpha=0.6)
plt.xlabel("Orbital Phase")
plt.ylabel("Mean Intensity")
plt.title("Pulsar Emission vs Orbital Phase")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "intensity_vs_phase.png"), dpi=150)
plt.close()

# =========================================================
# Plot 2 — Variance vs Phase
# =========================================================

plt.figure(figsize=(9,4))
plt.plot(centers, vis_curve, lw=2, label="Variance (windowed)")
for j, i in enumerate(peaks):
    plt.plot(centers[i], var_curve[i], "o", color="C3", label="Peak" if j==0 else None)
plt.xlabel("Orbital Phase")
plt.ylabel("Variance of Intensity")
plt.title("CET — Variance Peaks (Transition Regimes)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_folder, "variance_vs_phase.png"), dpi=150)
plt.close()

print("✅ CET full analysis done. Results saved in:", output_folder)