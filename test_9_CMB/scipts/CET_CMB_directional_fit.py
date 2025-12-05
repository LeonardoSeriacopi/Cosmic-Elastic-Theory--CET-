# ============================================================
# CET_CMB_directional_fit.py
# Análise hemisférica do mapa Planck SMICA para o modelo CET
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.optimize import curve_fit

# === Função CET ===
def cet_model(nu, A, gamma):
    return A * np.exp(-gamma * nu)

# === Ler mapa Planck SMICA ===
filename = "COM_CMB_IQU-smica_2048_R3.00_full.fits"
hdul = fits.open(filename)
I_map = hdul[1].data.field(0)  # componente I (intensidade)
hdul.close()

# === Parâmetros ===
nside = 2048
npix = len(I_map)
print(f"Mapa carregado com {npix} pixels")

# === Coordenadas (sem healpy, cálculo manual) ===
theta = np.arccos(1 - 2*np.arange(npix)/npix)
phi = 2 * np.pi * (np.arange(npix) / npix)

# === Separar hemisférios (latitude galáctica aproximada) ===
north_mask = theta < np.pi/2
south_mask = ~north_mask

I_north = I_map[north_mask]
I_south = I_map[south_mask]

# === Frequências médias correspondentes às bandas principais (GHz) ===
nu = np.array([30, 44, 70, 100, 143, 217, 353, 545, 857])

# === Intensidades médias (μK) por banda (simulação espectral) ===
# (Você pode substituir por dados reais do Planck band-averaged)
I_band_north = np.array([np.mean(I_north)] * len(nu))
I_band_south = np.array([np.mean(I_south)] * len(nu))

# === Normalização ΔI/I ===
deltaI_north = (I_band_north - np.mean(I_band_north)) / np.mean(I_band_north)
deltaI_south = (I_band_south - np.mean(I_band_south)) / np.mean(I_band_south)

# === Ajustes CET ===
popt_north, _ = curve_fit(cet_model, nu, deltaI_north, p0=[1e-5, 1e-3], maxfev=10000)
popt_south, _ = curve_fit(cet_model, nu, deltaI_south, p0=[1e-5, 1e-3], maxfev=10000)

A_N, gamma_N = popt_north
A_S, gamma_S = popt_south
delta_gamma = gamma_N - gamma_S

print(f"\nNORTH:  A={A_N:.3e}, γ={gamma_N:.3e}")
print(f"SOUTH:  A={A_S:.3e}, γ={gamma_S:.3e}")
print(f"Δγ (N-S) = {delta_gamma:.3e}")

# === Plot ===
plt.figure(figsize=(7,5))
plt.plot(nu, cet_model(nu, *popt_north), 'b-', lw=2, label=f"North fit (γ={gamma_N:.2e})")
plt.plot(nu, cet_model(nu, *popt_south), 'r--', lw=2, label=f"South fit (γ={gamma_S:.2e})")
plt.xlabel("Frequency (GHz)")
plt.ylabel("ΔI/I (normalized)")
plt.title("CET Dissipation Coefficient by Hemisphere (Planck SMICA)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("CET_CMB_directional_fit.png", dpi=300)
plt.show()