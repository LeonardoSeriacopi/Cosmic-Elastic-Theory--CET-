# ============================================================
# CET_fit_CMB.py
# Ajuste do modelo de dissipação CET aos resíduos espectrais do CMB
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# === 1. Carregar dados (tolerante a nomes de colunas diferentes) ===
data = pd.read_csv("cmb_residuals.csv")
print("Columns found:", data.columns.tolist())

# Identifica automaticamente as colunas principais
col_freq = [c for c in data.columns if "Freq" in c or "GHz" in c][0]
col_dI   = [c for c in data.columns if "Delta" in c or "I" in c][0]
col_sig  = [c for c in data.columns if "Sig" in c or "err" in c][0]

nu = data[col_freq].values          # frequência em GHz
deltaI = data[col_dI].values        # ΔI/I
sigma = data[col_sig].values        # incertezas

# === 2. Modelo CET: dissipação exponencial ===
def cet_model(nu, A, gamma):
    """Modelo CET: ΔI/I = A * exp(-γ * ν)"""
    return A * np.exp(-gamma * nu)

# === 3. Ajuste não-linear ===
p0 = [3e-4, 1e-3]  # chute inicial
popt, pcov = curve_fit(cet_model, nu, deltaI, sigma=sigma, p0=p0, absolute_sigma=True, maxfev=20000)
A_fit, gamma_fit = popt
err_A, err_gamma = np.sqrt(np.diag(pcov))

print(f"\nBest-fit parameters:")
print(f"  A = {A_fit:.3e} ± {err_A:.3e}")
print(f"  γ = {gamma_fit:.3e} ± {err_gamma:.3e}  [GHz⁻¹]")

# === 4. Curva ajustada ===
nu_fit = np.linspace(min(nu), max(nu), 500)
deltaI_fit = cet_model(nu_fit, *popt)

# === 5. Plot ===
plt.figure(figsize=(8,5))
plt.errorbar(nu, deltaI, yerr=sigma, fmt='o', color='royalblue', label='Observations (COBE + Planck + SPT)')
plt.plot(nu_fit, deltaI_fit, 'r-', lw=2, label=f'CET fit (γ = {gamma_fit:.2e} GHz⁻¹)')
plt.xlabel('Frequency ν (GHz)')
plt.ylabel('Residual ΔI/I')
plt.title('CMB Spectral Residuals vs CET Dissipation Model')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("CET_CMB_residual_fit.png", dpi=300)
plt.show()