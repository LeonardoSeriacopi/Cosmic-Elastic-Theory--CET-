#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
make_coldspot_density_map.py

Gera um mapa 2D de densidade das fontes 2MASS ao redor do
Cold Spot principal (CS1), marcando também o segundo Cold Spot (CS2).

Saída:
    - fig_coldspot_density_map.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# -------------------------------------------------
# 1. Parâmetros dos Cold Spots e seleção
# -------------------------------------------------

# Cold Spot principal (CS1)
RA_CS1  = 48.2999   # deg
DEC_CS1 = -20.4373  # deg

# Segundo Cold Spot interno (CS2)
RA_CS2  = 49.3442   # deg
DEC_CS2 = -19.7679  # deg

R_MAX_DEG = 10.0    # raio máximo para incluir fontes
KMIN = 8.0
KMAX = 16.0

INFILE = "2mass.tsv"
OUTFIG = "fig_coldspot_density_map.png"

# -------------------------------------------------
# 2. Função para separação angular em graus
# -------------------------------------------------

def angsep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
    Distância angular em graus entre dois pontos na esfera.
    (RA,Dec) em graus. Usa fórmula do cosseno esférico.
    """
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)

    cos_theta = (
        np.sin(dec1)*np.sin(dec2) +
        np.cos(dec1)*np.cos(dec2)*np.cos(ra1 - ra2)
    )
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# -------------------------------------------------
# 3. Carregar catálogo 2MASS e filtrar
# -------------------------------------------------

print("=========================================================")
print(f"[INFO] Lendo catálogo 2MASS: {INFILE}")
df = pd.read_csv(INFILE, sep="\t", comment="#", low_memory=False)

# Garantir colunas numéricas
for col in ["RAJ2000", "DEJ2000", "Kmag"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

mask_valid = (
    df["RAJ2000"].notna() &
    df["DEJ2000"].notna() &
    df["Kmag"].notna()
)
df = df.loc[mask_valid].copy()
print(f"[INFO] Fontes com RA/Dec/Kmag válidos: {len(df)}")

# Seleção em magnitude
mask_mag = (df["Kmag"] >= KMIN) & (df["Kmag"] <= KMAX)
df = df.loc[mask_mag].copy()
print(f"[INFO] Fontes após corte {KMIN} <= K <= {KMAX}: {len(df)}")

ra  = df["RAJ2000"].values
dec = df["DEJ2000"].values

# -------------------------------------------------
# 4. Selecionar região R <= 10° em torno de CS1
# -------------------------------------------------

theta_cs1 = angsep_deg(ra, dec, RA_CS1, DEC_CS1)
mask_r = (theta_cs1 <= R_MAX_DEG)
ra_sel  = ra[mask_r]
dec_sel = dec[mask_r]
print(f"[INFO] Fontes dentro de R <= {R_MAX_DEG}° do CS1: {len(ra_sel)}")

# -------------------------------------------------
# 5. Converter para coordenadas locais (offsets)
# -------------------------------------------------

ra0_rad  = np.radians(RA_CS1)
dec0_rad = np.radians(DEC_CS1)

ra_rad   = np.radians(ra_sel)
dec_rad  = np.radians(dec_sel)

# Aproximação de pequeno ângulo (boa para ~10°)
dx_deg = (ra_rad - ra0_rad) * np.cos(dec0_rad) * (180.0/np.pi)
dy_deg = (dec_rad - dec0_rad) * (180.0/np.pi)

# Posição relativa do CS2 no mesmo sistema
ra2_rad  = np.radians(RA_CS2)
dec2_rad = np.radians(DEC_CS2)
dx_cs2 = (ra2_rad - ra0_rad) * np.cos(dec0_rad) * (180.0/np.pi)
dy_cs2 = (dec2_rad - dec0_rad) * (180.0/np.pi)

print(f"[INFO] Offset CS2 relativo ao CS1: Δx = {dx_cs2:.3f}°, Δy = {dy_cs2:.3f}°")

# -------------------------------------------------
# 6. Construir mapa de densidade 2D (histograma + smoothing)
# -------------------------------------------------

extent = R_MAX_DEG   # cobrimos -R_MAX .. +R_MAX em x,y
grid_size = 300

xbins = np.linspace(-extent, extent, grid_size+1)
ybins = np.linspace(-extent, extent, grid_size+1)

H, xedges, yedges = np.histogram2d(dy_deg, dx_deg, bins=[ybins, xbins])
# Nota: usamos (y,x) na chamada para casar com imshow (linha = y, coluna = x)

# Suavização gaussiana para um visual mais suave
H_smooth = gaussian_filter(H, sigma=3.0)

# -------------------------------------------------
# 7. Plotar o mapa
# -------------------------------------------------

plt.figure(figsize=(7, 6))

# imshow: extent = [xmin, xmax, ymin, ymax]
plt.imshow(
    H_smooth,
    extent=[-extent, extent, -extent, extent],
    origin="lower",
    cmap="inferno",
    aspect="equal"
)

cbar = plt.colorbar()
cbar.set_label("Relative 2MASS density (arbitrary units)")

# Marcar centro do CS1
plt.scatter(0.0, 0.0, marker="x", s=100, c="cyan", label="Main Cold Spot (CS1)")

# Marcar CS2
plt.scatter(dx_cs2, dy_cs2, marker="o", s=70, facecolors="none",
            edgecolors="lime", linewidths=2, label="Secondary Cold Spot (CS2)")

# Desenhar anéis (0–2, 2–4, ..., 8–10 deg)
for r in [2, 4, 6, 8, 10]:
    circ = plt.Circle((0.0, 0.0), r, color="white", linestyle="--",
                      linewidth=0.8, fill=False, alpha=0.7)
    plt.gca().add_patch(circ)

plt.xlim(-extent, extent)
plt.ylim(-extent, extent)

plt.xlabel(r"$\Delta\alpha \cos\delta_1\ \mathrm{[deg]}$")
plt.ylabel(r"$\Delta\delta\ \mathrm{[deg]}$")
plt.title("2MASS Galaxy Density Around the Main CMB Cold Spot")

plt.legend(loc="upper right", frameon=True, fontsize=9)
plt.grid(alpha=0.15, linestyle=":")

plt.tight_layout()
plt.savefig(OUTFIG, dpi=300)
plt.close()

print(f"[OK] Mapa de densidade salvo em: {OUTFIG}")