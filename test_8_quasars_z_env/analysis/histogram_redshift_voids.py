#!/usr/bin/env python3
# histogram_redshift_voids_updated.py
"""
Gera histograma normalizado (density=True) comparando redshifts de quasares
dentro e fora de voids – baseado em dr16q_merged_voidflag.csv.

Saída: histogram_redshift_voids.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   # só para theme, não para plot

# ----------------------------------------------------------------------
# 1. Arquivo de entrada -------------------------------------------------
CSV_IN = "dr16q_merged_voidflag.csv"     # <<-- arquivo alvo atualizado
PNG_OUT = "histogram_redshift_voids.png"

# 2. Lê catálogo e garante tipos numéricos ------------------------------
df = pd.read_csv(CSV_IN)

# Colunas-chave
z_col = "Z_QSO"          # redshift do quasar
flag_col = "inside_void" # True / False

# Converte para numérico e Boolean
df[z_col] = pd.to_numeric(df[z_col], errors="coerce")
df[flag_col] = df[flag_col].astype(bool)
df = df.dropna(subset=[z_col])           # remove linhas sem z

inside = df[df[flag_col]][z_col]
outside = df[~df[flag_col]][z_col]

print(f"N inside = {len(inside):,d}, N outside = {len(outside):,d}")

# ----------------------------------------------------------------------
# 3. Plot ---------------------------------------------------------------
sns.set_style("whitegrid")
plt.figure(figsize=(10, 6))

bins = 40
plt.hist(
    inside,
    bins=bins,
    alpha=0.6,
    density=True,
    label=f"Dentro de voids (N={len(inside):,d})",
    color="#F4C430",
)
plt.hist(
    outside,
    bins=bins,
    alpha=0.6,
    density=True,
    label=f"Fora de voids (N={len(outside):,d})",
    color="#E8745D",
)

plt.xlabel("Redshift (Z)", fontsize=13)
plt.ylabel("Densidade normalizada", fontsize=13)
plt.title("Distribuição de redshift – Quasares dentro vs. fora de voids", fontsize=15)
plt.legend()
plt.tight_layout()
plt.savefig(PNG_OUT, dpi=300)
plt.show()

print(f"\nFigura salva em: {PNG_OUT}")
