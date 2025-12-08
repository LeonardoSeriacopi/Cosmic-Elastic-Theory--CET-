#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analyze_2mass_coldspot.py

Usa o catálogo 2MASS recortado (2mass.tsv) para medir a densidade
de fontes em torno do CMB Cold Spot:
- disco central (R <= 5 deg)
- anel de controle (5 < R <= 10 deg)

Não precisa de healpy nem HEALPix, só RA/Dec.
"""

import numpy as np
import pandas as pd

# ---------------------------
# Configurações principais
# ---------------------------

# Nome do arquivo 2MASS recortado que você baixou do VizieR
TWO_MASS_FILE = "2mass.tsv"   # altere se o nome for diferente

# Centro do Cold Spot em coordenadas equatoriais (ICRS/J2000)
# Esses valores vêm da conversão anterior (l=209°, b=-57°)
RA_CS_DEG  = 48.2999
DEC_CS_DEG = -20.4373

# Raio do disco e do anel (em graus)
R_DISK_DEG = 5.0
R_RING_DEG = 10.0  # anel entre 5 e 10 graus

# Colunas de RA, Dec e magnitude no catálogo 2MASS
RA_COL   = "RAJ2000"
DEC_COL  = "DEJ2000"
KMAG_COL = "Kmag"   # se quiser trocar para Jmag ou Hmag é só mudar aqui

# ---------------------------
# Funções auxiliares
# ---------------------------

def angsep_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg):
    """
    Distância angular em graus entre (ra1, dec1) e (ra2, dec2) usando
    a fórmula do cosseno esférico.
    """
    ra1 = np.radians(ra1_deg)
    dec1 = np.radians(dec1_deg)
    ra2 = np.radians(ra2_deg)
    dec2 = np.radians(dec2_deg)

    cos_d = (np.sin(dec1) * np.sin(dec2) +
             np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    # Segurança numérica
    cos_d = np.clip(cos_d, -1.0, 1.0)
    d_rad = np.arccos(cos_d)
    return np.degrees(d_rad)

# ---------------------------
# Leitura do catálogo
# ---------------------------

print("=" * 80)
print(f"[INFO] Lendo catálogo 2MASS recortado: {TWO_MASS_FILE}")

# O arquivo vem em formato tab-separated values, com possíveis linhas de comentário
df = pd.read_csv(
    TWO_MASS_FILE,
    sep="\t",
    comment="#",
    low_memory=False
)

print(f"[INFO] Linhas totais no arquivo: {len(df)}")
print(f"[INFO] Colunas disponíveis: {list(df.columns)}")

if RA_COL not in df.columns or DEC_COL not in df.columns:
    raise RuntimeError(
        f"[ERRO] Não encontrei colunas '{RA_COL}'/'{DEC_COL}' no arquivo. "
        f"Colunas disponíveis: {list(df.columns)}"
    )

# Garante que RA/Dec sejam numéricos
df[RA_COL]  = pd.to_numeric(df[RA_COL], errors="coerce")
df[DEC_COL] = pd.to_numeric(df[DEC_COL], errors="coerce")

mask_valid = df[RA_COL].notna() & df[DEC_COL].notna()
df = df.loc[mask_valid].copy()

print(f"[INFO] Linhas com RA/Dec válidos: {len(df)}")

# ---------------------------
# Distância angular até o Cold Spot
# ---------------------------

print("=" * 80)
print("[INFO] Calculando distância angular até o centro do Cold Spot...")

theta_deg = angsep_deg(df[RA_COL].values,
                       df[DEC_COL].values,
                       RA_CS_DEG,
                       DEC_CS_DEG)

df["theta_deg"] = theta_deg

print(f"[INFO] theta_deg min/max = {theta_deg.min():5.2f} – {theta_deg.max():5.2f} deg")

# ---------------------------
# Seleção de disco e anel
# ---------------------------

disk_mask = df["theta_deg"] <= R_DISK_DEG
ring_mask = (df["theta_deg"] > R_DISK_DEG) & (df["theta_deg"] <= R_RING_DEG)

df_disk = df.loc[disk_mask].copy()
df_ring = df.loc[ring_mask].copy()

n_disk = len(df_disk)
n_ring = len(df_ring)

print("=" * 80)
print("[RESULTADO] Contagens 2MASS em torno do Cold Spot")
print(f"  Fontes no DISCO  (R <= {R_DISK_DEG:.1f}°):       {n_disk}")
print(f"  Fontes no ANEL   ({R_DISK_DEG:.1f}° < R <= {R_RING_DEG:.1f}°): {n_ring}")

# ---------------------------
# Densidades angulares
# ---------------------------

area_disk = np.pi * (R_DISK_DEG ** 2)
area_ring = np.pi * (R_RING_DEG ** 2 - R_DISK_DEG ** 2)

dens_disk = n_disk / area_disk if area_disk > 0 else np.nan
dens_ring = n_ring / area_ring if area_ring > 0 else np.nan

print()
print("[RESULTADO] Densidades angulares")
print(f"  Área do DISCO = {area_disk:7.2f} deg^2")
print(f"  Área do ANEL  = {area_ring:7.2f} deg^2")
print(f"  Densidade DISCO = {dens_disk:8.3f} fontes/deg^2")
print(f"  Densidade ANEL  = {dens_ring:8.3f} fontes/deg^2")

# ---------------------------
# Histograma em magnitude K (opcional)
# ---------------------------

if KMAG_COL in df.columns:
    print()
    print(f"[INFO] Calculando histogramas em {KMAG_COL} (DISCO vs ANEL)...")

    df_disk[KMAG_COL] = pd.to_numeric(df_disk[KMAG_COL], errors="coerce")
    df_ring[KMAG_COL] = pd.to_numeric(df_ring[KMAG_COL], errors="coerce")

    bins = np.arange(8.0, 16.1, 0.5)  # você pode mudar isso
    hist_disk, _ = np.histogram(df_disk[KMAG_COL].dropna(), bins=bins)
    hist_ring, _ = np.histogram(df_ring[KMAG_COL].dropna(), bins=bins)

    print("\n[RESULTADO] Histograma grosso em Kmag")
    print("  Bin(Kmag)    N_disk    N_ring")
    for i in range(len(bins) - 1):
        b0, b1 = bins[i], bins[i+1]
        print(f" {b0:4.1f}–{b1:4.1f}   {hist_disk[i]:7d}  {hist_ring[i]:7d}")

    # Salva histogramas em arquivo texto
    hist_out = "2mass_coldspot_hist_Kmag.txt"
    with open(hist_out, "w", encoding="utf-8") as f:
        f.write("# Kmag_bin_low  Kmag_bin_high  N_disk  N_ring\n")
        for i in range(len(bins) - 1):
            f.write(f"{bins[i]:.2f} {bins[i+1]:.2f} {hist_disk[i]} {hist_ring[i]}\n")
    print(f"\n[INFO] Histogramas salvos em: {hist_out}")

else:
    print(f"[AVISO] Coluna {KMAG_COL} não encontrada; pulando histograma de magnitude.")

# ---------------------------
# Salvando catálogos recortados
# ---------------------------

df_disk.to_csv("2mass_in_coldspot_disk.csv", index=False)
df_ring.to_csv("2mass_in_coldspot_ring.csv", index=False)

print()
print("[INFO] Catálogos recortados salvos como:")
print("       - 2mass_in_coldspot_disk.csv")
print("       - 2mass_in_coldspot_ring.csv")
print("\n[OK] Finalizado.")