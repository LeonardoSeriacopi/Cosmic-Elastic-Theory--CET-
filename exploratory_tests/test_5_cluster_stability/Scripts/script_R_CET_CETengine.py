
import pandas as pd
import numpy as np
from scipy.constants import c, G

# Constantes cosmológicas
H0_kms_Mpc = 70
h = H0_kms_Mpc / 100
H0_s = H0_kms_Mpc * 1000 / 3.08568e22  # H0 em s⁻¹
theta = 0.0003  # rad (1 arcmin)

# Relação massa-riqueza calibrada (DES Y3)
def richness_to_mass(n_gal, A=1.5, alpha=1.3):
    return A * 1e14 * (n_gal / 40.0) ** alpha  # M_sun

# Cálculo revisado de R_CET
def calculate_R_CET_corrected(row, P0, V0):
    # Massa recalculada (em kg)
    M_halo = richness_to_mass(row['n_galaxies'])
    mass_kg = M_halo * 1.989e30

    # Distância real
    d_real = (c / H0_s) * np.log(1 + row['z'])

    # Volume real com novo θ
    r_physical = theta * d_real
    V_real = (4/3) * np.pi * r_physical**3

    # Densidade e pressão CET
    rho_CET = mass_kg / V_real
    P_CET = rho_CET * c**2

    # Pressão crítica CET
    P_crit_CET = (3 * H0_s**2 * c**2 / (8 * np.pi * G)) * (1 + row['z'])**1.5

    # R-value (normalizado)
    R_raw = (P_CET - P_crit_CET) * V_real
    return R_raw / (P0 * V0)

# Leitura dos dados
df = pd.read_csv("Simulated_Cluster_Density_Table.csv")

# Massa recalculada para todos os clusters
df['M_halo'] = df['n_galaxies'].apply(richness_to_mass)
df['mass_kg'] = df['M_halo'] * 1.989e30

# Distância real e volume real com novo θ
df['d_real'] = (c / H0_s) * np.log(1 + df['z'])
df['r_physical'] = theta * df['d_real']
df['V_real'] = (4/3) * np.pi * df['r_physical']**3

# Pressão CET
df['rho_CET'] = df['mass_kg'] / df['V_real']
df['P_CET'] = df['rho_CET'] * c**2

# Pressão crítica CET
df['P_crit_CET'] = (3 * H0_s**2 * c**2 / (8 * np.pi * G)) * (1 + df['z'])**1.5

# Normalização
P0 = df['P_CET'].median()
V0 = df['V_real'].median()

# Cálculo final de R_CET
df['R_CET'] = df.apply(lambda row: calculate_R_CET_corrected(row, P0, V0), axis=1)
df['Estabilidade'] = df['R_CET'].apply(lambda r: '✅ (R>1)' if r > 1 else '❌ (R<=1)')

# Salvar resultado
df.to_excel("Resultado_CET_Corrigido.xlsx", index=False)
