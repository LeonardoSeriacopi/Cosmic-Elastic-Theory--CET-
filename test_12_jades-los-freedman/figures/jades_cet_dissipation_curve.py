#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gera a figura principal do teste JADES:
Δμ (LOS–Freedman) vs redshift, com ajuste CET tipo exponencial saturante
e bandas de confiança (1σ, 2σ) usando regressão linear com τ fixo.

Entrada esperada:
  jades_dissipacao_curve_binned.csv
com colunas:
  z_bin_center, mean_delta_mu, std_delta_mu, N_bin

Saída:
  jades_cet_dissipation_curve.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. Carregar tabela binned
# -------------------------------------------------------
CSV_FILE = "jades_dissipation_curve_binned.csv"
df = pd.read_csv(CSV_FILE)

# Renomear colunas se necessário (ajuste se os nomes forem diferentes)
col_z   = "z_bin_center"
col_mu  = "mean_delta_mu"
col_sig = "std_delta_mu"
col_N   = "N_bin"

z_all   = df[col_z].to_numpy(dtype=float)
mu_all  = df[col_mu].to_numpy(dtype=float)
sig_all = df[col_sig].to_numpy(dtype=float)
N_all   = df[col_N].to_numpy(dtype=float)

# -------------------------------------------------------
# 2. Selecionar bins "robustos":
#    N_bin >= 5 e sigma > 0
# -------------------------------------------------------
mask = (N_all >= 5) & (sig_all > 0)
z    = z_all[mask]
mu   = mu_all[mask]
sig  = sig_all[mask]

print("N total de bins:", z_all.size)
print("N bins usados   :", z.size)
print("Faixa de z usada:", z.min(), z.max())

# -------------------------------------------------------
# 3. Ajuste CET-like: exponencial saturante com τ fixo
#
#    μ(z) = μ_inf + A * exp( - (z - z0)/τ )
#
#    Este modelo é linear em μ_inf e A para τ fixo.
# -------------------------------------------------------
tau = 3.0           # valor fixo (pode mudar se quiser testar)
z0  = z.min()       # âncora em z mínimo da amostra

# Construir matriz de projeto X:
#   μ = μ_inf * 1 + A * exp(...)

exp_term = np.exp(-(z - z0)/tau)
X = np.vstack([
    np.ones_like(z),   # coluna 1 -> μ_inf
    exp_term           # coluna 2 -> A
]).T  # shape (N,2)

# Matriz de pesos (erro em μ)
w = 1.0 / (sig**2)
W = np.diag(w)

# Resolver (X^T W X) p = X^T W y
XTW = X.T @ W
XTWX = XTW @ X
XTWy = XTW @ mu

cov_params = np.linalg.inv(XTWX)
params = cov_params @ XTWy

mu_inf, A = params
print("\n=== Parâmetros do ajuste CET-like (τ fixo) ===")
print(f"mu_inf = {mu_inf:.8f}")
print(f"A      = {A:.8f}")
print(f"tau    = {tau:.3f} (fixo)")

# Qui-quadrado do modelo
mu_model = X @ params
chi2 = np.sum(((mu - mu_model)/sig)**2)
dof  = z.size - 2  # 2 parâmetros livres (mu_inf, A)
chi2_red = chi2/dof
print(f"chi2    = {chi2:.3f}")
print(f"dof     = {dof}")
print(f"chi2red = {chi2_red:.3f}")

# -------------------------------------------------------
# 4. Gerar curva suave e bandas de confiança
# -------------------------------------------------------
z_grid = np.linspace(z.min() - 0.1, z.max() + 0.1, 400)
exp_grid = np.exp(-(z_grid - z0)/tau)

# Modelo na grade
mu_grid = mu_inf + A * exp_grid

# Incerteza do modelo (propagação linear)
# grad = [∂μ/∂μ_inf, ∂μ/∂A] = [1, exp_grid]
# σ_model^2 = grad^T Cov grad
sigma_model = []
for eg in exp_grid:
    grad = np.array([1.0, eg])
    var = grad @ cov_params @ grad
    sigma_model.append(np.sqrt(var))
sigma_model = np.array(sigma_model)

mu_grid_1p = mu_grid + sigma_model
mu_grid_1m = mu_grid - sigma_model
mu_grid_2p = mu_grid + 2*sigma_model
mu_grid_2m = mu_grid - 2*sigma_model

# -------------------------------------------------------
# 5. Plotar figura
# -------------------------------------------------------
plt.figure(figsize=(7,5))

ax = plt.gca()

# Faixa vertical da janela JADES (opcional)
zmin_jades = 3.0
zmax_jades = 13.0
ax.axvspan(zmin_jades, zmax_jades, color="lightgrey", alpha=0.2,
           label="JADES window")

# Bandas 2σ e 1σ
ax.fill_between(z_grid, mu_grid_2m, mu_grid_2p,
                color="grey", alpha=0.2, label=r"$2\sigma$ band")
ax.fill_between(z_grid, mu_grid_1m, mu_grid_1p,
                color="grey", alpha=0.4, label=r"$1\sigma$ band")

# Curva CET
ax.plot(z_grid, mu_grid, color="red", linewidth=2.0,
        label="CET dissipation model")

# Pontos (bins JADES)
ax.errorbar(z, mu, yerr=sig, xerr=None,
            fmt="o", markersize=5, capsize=3,
            color="navy", ecolor="navy", elinewidth=1,
            label="JADES (LOS–Freedman bins)")

# Linha horizontal em mu_inf (saturação assintótica)
ax.axhline(mu_inf, color="black", linestyle="--",
           linewidth=1.2, label=r"$\mu_\infty$")

# Rótulos e layout
ax.set_xlabel("Redshift $z$")
ax.set_ylabel(r"$\Delta\mu$ (LOS--Freedman residual)")
ax.set_xlim(z_grid.min(), z_grid.max())
# Ajuste leve nos limites de y
ymin = min(mu_grid_2m.min(), mu.min() - sig.max())
ymax = max(mu_grid_2p.max(), mu.max() + sig.max())
ax.set_ylim(ymin - 0.0005, ymax + 0.0005)

ax.legend(loc="best", fontsize=9)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("jades_dissipation_curve.png", dpi=300)
plt.close()

print("\nFigura salva em: jades_cet_dissipation_curve.png")