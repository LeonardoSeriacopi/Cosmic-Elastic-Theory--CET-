import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

# --------------------------------------------------
# Main Configuration
# --------------------------------------------------

DELTAD_FILE = "DeltaD_binned.csv"
SAVE_FIGS = True
FIG_PREFIX = "CET_Universal_Dissipation_Curve_Final_Normalized"

# CRITICAL smoothing parameter for high-order derivatives
SMOOTHING_FACTOR = 30 

# --------------------------------------------------
# CET Formalism - Conceptual K Curve
# --------------------------------------------------

K_SAT = 1.5
Z_DECAY = 0.45 

def K_conceptual(z, K_sat, z_decay):
    """Conceptual function for K(z) evolution in the transition zone (Decaying)."""
    return K_sat * np.exp(-z / z_decay)

# --------------------------------------------------
# Data Reading and Derivatives Calculation
# --------------------------------------------------

print("=" * 70)
print(f"[INFO] Reading binned data file: {DELTAD_FILE}")

df = pd.read_csv(DELTAD_FILE)
z = df['z_bin_center'].values
DeltaD = df['DeltaD_mean'].values
DeltaD_err = df['DeltaD_error'].values

# Interpolation Spline with extreme smoothing
print(f"[INFO] Applying spline smoothing with s={SMOOTHING_FACTOR}")
spline = UnivariateSpline(z, DeltaD, w=1/DeltaD_err, s=SMOOTHING_FACTOR) 

# Calculate Derivatives on a high-resolution grid for smooth plots
z_smooth = np.linspace(z.min(), z.max(), 500) 
DeltaD_smooth = spline(z_smooth)
dDeltaD_dz_smooth = spline.derivative(n=1)(z_smooth)
d2DeltaD_dz2_smooth = spline.derivative(n=2)(z_smooth)

# Kinematic Acceleration at original bin centers (Binned Normal)
d2DeltaD_dz2_binned = spline.derivative(n=2)(z)
# Estimate error for visualization purposes (scaled D_err)
d2DeltaD_dz2_err_binned = DeltaD_err * 10 

# Conceptual K Curve
K_z_smooth = K_conceptual(z_smooth, K_SAT, Z_DECAY)

# --------------------------------------------------
# Normalization for Kinematic Acceleration (Panel 4)
# --------------------------------------------------

# Normalize everything by the maximum absolute value of the smooth fit
accel_norm_factor = np.abs(d2DeltaD_dz2_smooth).max() 
d2DeltaD_dz2_smooth_norm = d2DeltaD_dz2_smooth / accel_norm_factor
d2DeltaD_dz2_binned_norm = d2DeltaD_dz2_binned / accel_norm_factor
d2DeltaD_dz2_err_binned_norm = d2DeltaD_dz2_err_binned / accel_norm_factor

# --------------------------------------------------
# Plots - Universal Dissipation Curve (English Labels, Normalized Acceleration)
# --------------------------------------------------

plt.figure(figsize=(8, 10))

# Panel 1: Delta D(z) - Universal Dissipation (Saturation Curve)
ax1 = plt.subplot(4, 1, 1)
ax1.errorbar(z, DeltaD, yerr=DeltaD_err, fmt="o", ms=4, capsize=3, color='#004D99', label=r"$\Delta D(z)$ Data (SDSS+DESI)")
ax1.plot(z_smooth, DeltaD_smooth, color='#CC0000', linewidth=2, label="CET Fit (Saturation)")
ax1.set_ylabel(r"Accumulated Dissipation $\Delta D$ [Mpc]")
ax1.set_title(r"CET: Universal Dissipation Curve and Coupling Polarity ($\mathbf{K}$)", fontsize=14)
ax1.grid(True, which="major", ls=":", alpha=0.7)
ax1.legend(loc='lower right', fontsize=8)

# Panel 2: K(z) - Coupling Polarity
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
ax2.plot(z_smooth, K_z_smooth, color='#009900', linewidth=3, label=r"Elastic Coupling $K(z)$ (Conceptual)")
ax2.axhline(K_SAT, color='gray', linestyle='--', alpha=0.7, label=r"$K_{sat} = 1.5$ (Confining Regime)")
ax2.axhline(0.0, color='black', linestyle=':', alpha=0.5, label=r"$K = 0$ (Free Dissipative Regime)")

ax2.set_ylabel(r"Coupling $K$ (Polarity)")
ax2.grid(True, which="major", ls=":", alpha=0.7)
ax2.legend(loc='upper right', fontsize=8)
plt.setp(ax2.get_xticklabels(), visible=False)

# Panel 3: dΔD/dz - Effective Dissipation Rate (Gamma_diss)
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
ax3.plot(z_smooth, dDeltaD_dz_smooth, color='#330066', linewidth=2)
ax3.text(0.60, 0.75, r"$\frac{d(\Delta D)}{dz} \propto \Gamma_{diss}$ (Free Flow Rate)", 
         transform=ax3.transAxes, fontsize=10, color='#330066', fontweight='bold')
ax3.set_ylabel(r"Dissipation Rate $d(\Delta D)/dz$")
ax3.grid(True, which="major", ls=":", alpha=0.7)
plt.setp(ax3.get_xticklabels(), visible=False)

# Panel 4: Normalized Kinematic Acceleration (a_kin) - Binned Data + Smooth Fit
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
# 1. Plot normalized Binned Data points for comparison
ax4.errorbar(z, d2DeltaD_dz2_binned_norm, yerr=d2DeltaD_dz2_err_binned_norm, 
             fmt="o", ms=4, capsize=3, color='#990000', alpha=0.4, label="Binned Data (Normalized)") 
# 2. Plot normalized Smooth Fit (theoretical curve)
ax4.plot(z_smooth, d2DeltaD_dz2_smooth_norm, color='#990000', linewidth=2, label="CET Smooth Fit")

# *** AJUSTE DA POSIÇÃO DO TEXTO AQUI: 0.8 -> 0.2 ***
ax4.text(0.55, 0.2, r"$\frac{d^2(\Delta D)}{dz^2} \propto a_{kin}$ (Decaying Acceleration)", 
         transform=ax4.transAxes, fontsize=10, color='#990000', fontweight='bold')
ax4.set_ylabel(r"Norm. Kinematic Accel.")
ax4.set_xlabel(r"Redshift $z$")
ax4.grid(True, which="major", ls=":", alpha=0.7)
ax4.axhline(0.0, color='black', linestyle=':', alpha=0.7)
ax4.legend(loc='upper right', fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.98])

if SAVE_FIGS:
    plot_filename = f"{FIG_PREFIX}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"\n[SUCCESS] Plot saved as: {plot_filename}")

# Imprimindo o código para o usuário salvar
print("\n[PYTHON SCRIPT END]")