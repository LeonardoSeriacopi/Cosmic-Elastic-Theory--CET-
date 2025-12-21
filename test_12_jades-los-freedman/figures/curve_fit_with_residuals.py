#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera figura 2x1 para o teste JADES:
- Painel superior: Δμ(z) com barras de erro + ajuste CET-like
- Painel inferior: resíduos Δμ_obs - Δμ_model

Entrada: jades_dissipacao_curve_binned.csv
(espera colunas: z_bin_center, mean_delta_mu, std_delta_mu, N_bin)
"""

import numpy as np
import matplotlib.pyplot as plt
from math import isfinite

# -----------------------------
# Configurações básicas
# -----------------------------

CSV_FILE = "jades_dissipation_curve_binned.csv"
OUT_PNG  = "curve_fit_with_residuals.png"

# fixa tau como no ajuste anterior
TAU_FIXO = 3.0

# mínimo de objetos por bin e sigma>0
MIN_N_BIN   = 5
MIN_STD_MU  = 0.0   # já garantimos >0 checando diretamente


# -----------------------------
# Modelo CET-like (exp saturante)
# -----------------------------

def cet_model(z, mu_inf, A, tau=TAU_FIXO):
    z0 = z.min()
    return mu_inf + A * np.exp(-(z - z0) / tau)


def chi2(mu_obs, mu_err, mu_model):
    return np.sum(((mu_obs - mu_model) / mu_err) ** 2)


# -----------------------------
# Leitura do CSV
# -----------------------------

data = np.genfromtxt(CSV_FILE, delimiter=",", names=True)

z_all     = data["z_bin_center"]
mu_all    = data["mean_delta_mu"]
sigma_all = data["std_delta_mu"]
N_all     = data["N_bin"]

# máscara: bins com N >= MIN_N_BIN e sigma>0
mask = (N_all >= MIN_N_BIN) & (sigma_all > 0.0)

z   = z_all[mask]
mu  = mu_all[mask]
err = sigma_all[mask]

print(f"N total de bins: {len(z_all)}")
print(f"N bins usados   : {len(z)}")
print("Faixa de z usada:", z.min(), z.max())

# -----------------------------
# Ajuste do modelo CET-like
# (tau fixo, ajusta mu_inf e A por chi2 scan simples)
# -----------------------------

# chutes iniciais grosseiros
mu_inf_guess = mu.min() - 0.001
A_guess      = mu.max() - mu_inf_guess

mu_inf_vals = np.linspace(mu_inf_guess - 0.002, mu_inf_guess + 0.002, 81)
A_vals      = np.linspace(A_guess * 0.5, A_guess * 1.5, 81)

best_chi2 = np.inf
best_mu_inf = None
best_A = None

for mu_inf_try in mu_inf_vals:
    for A_try in A_vals:
        mu_model_try = cet_model(z, mu_inf_try, A_try, TAU_FIXO)
        # evita casos malucos
        if not np.all(np.isfinite(mu_model_try)):
            continue
        c2 = chi2(mu, err, mu_model_try)
        if c2 < best_chi2:
            best_chi2 = c2
            best_mu_inf = mu_inf_try
            best_A = A_try

dof = len(z) - 2  # dois parâmetros livres: mu_inf e A
chi2_red = best_chi2 / dof if dof > 0 else np.nan

print("\n=== Parâmetros do ajuste CET-like (τ fixo) ===")
print(f"mu_inf = {best_mu_inf:.8f}")
print(f"A      = {best_A:.8f}")
print(f"tau    = {TAU_FIXO:.3f} (fixo)")
print(f"chi2    = {best_chi2:.3f}")
print(f"dof     = {dof}")
print(f"chi2red = {chi2_red:.3f}")

# modelo final (para curva suave)
z_plot = np.linspace(z.min(), z.max(), 400)
mu_model_plot = cet_model(z_plot, best_mu_inf, best_A, TAU_FIXO)
mu_model_bins = cet_model(z, best_mu_inf, best_A, TAU_FIXO)

residuos = mu - mu_model_bins

# -----------------------------
# Figura 2x1
# -----------------------------

fig, (ax1, ax2) = plt.subplots(
    2, 1, sharex=True, figsize=(7, 6),
    gridspec_kw={"height_ratios": [3, 1]}
)

# Painel superior: Δμ com CET
ax1.errorbar(
    z, mu, yerr=err,
    fmt="o", markersize=4, capsize=3, label=r"JADES (LOS/Freedman bins)"
)
ax1.plot(
    z_plot, mu_model_plot,
    "-", label=r"Modelo CET-like (exp. saturante)"
)

ax1.set_ylabel(r"$\langle \Delta\mu \rangle$")
ax1.legend(loc="best", fontsize=9)
ax1.grid(True, alpha=0.3)

# Painel inferior: resíduos
ax2.axhline(0.0, linestyle="--", linewidth=1)
ax2.errorbar(
    z, residuos, yerr=err,
    fmt="o", markersize=4, capsize=3
)
ax2.set_xlabel(r"$z$")
ax2.set_ylabel(r"$\Delta\mu - \Delta\mu_{\rm CET}$")
ax2.grid(True, alpha=0.3)

fig.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"\nFigura salva em: {OUT_PNG}")
plt.close(fig)