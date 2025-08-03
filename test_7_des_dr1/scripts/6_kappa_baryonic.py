import pandas as pd
import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G, c

# --- CORREÇÃO APLICADA AQUI ---
# Garantir que as constantes G e c tenham as unidades corretas do Astropy
# (É mais seguro importar diretamente de astropy.constants para evitar erros de digitação)

# Carregar os dados
df = pd.read_csv("PSF_com_massa_barionica.csv")

# Filtrar apenas galáxias
df = df[df["class_star_i"] < 0.9].copy() # Usar .copy() para evitar SettingWithCopyWarning

# Verificar existência das colunas necessárias
assert "M_barionica_kg" in df.columns, "Coluna 'M_barionica_kg' não encontrada"
assert "mean_z_x" in df.columns, "Coluna 'mean_z_x' não encontrada"

# Extração de massa e redshift
M = df["M_barionica_kg"].values * u.kg
z_l = df["mean_z_x"].values
z_s = 1.0  # Redshift de fonte fixo

# Filtro de validade para redshifts (a lente deve estar na frente da fonte)
valid_z = z_l < z_s
df_valid = df[valid_z].copy()
z_l_valid = df_valid["mean_z_x"].values
M_valid = df_valid["M_barionica_kg"].values * u.kg

# Cálculo das distâncias angulares apenas para os redshifts válidos
D_l = cosmo.angular_diameter_distance(z_l_valid)
D_s = cosmo.angular_diameter_distance(z_s)
D_ls = cosmo.angular_diameter_distance_z1z2(z_l_valid, z_s)

# Área assumida com raio físico de 50 kpc
r_phys = (50 * u.kpc).to(u.m)
area = np.pi * r_phys**2

# Calcular Σ_barionica para as galáxias válidas
Sigma_barionica_valid = (M_valid / area).to(u.kg / u.m**2)
df_valid["Sigma_barionica"] = Sigma_barionica_valid.value

# Calcular Sigma_crit para as galáxias válidas
# A fórmula já resulta em kg/m^2 automaticamente com as constantes do Astropy
sigma_crit_calc = (c**2 / (4 * np.pi * G)) * (D_s / (D_l * D_ls))
df_valid["Sigma_crit"] = sigma_crit_calc.to(u.kg / u.m**2).value

# Calcular kappa_barionica para as galáxias válidas
df_valid["kappa_barionica"] = df_valid["Sigma_barionica"] / df_valid["Sigma_crit"]

# Salvar resultado (apenas com as galáxias válidas)
df_valid.to_csv("PSF_com_kappa_barionica.csv", index=False)

# Feedback do processo
print("✅ Cálculo do kappa bariônico concluído com sucesso!")
print("\nEstatísticas descritivas para as galáxias com z_l < z_s:")
print(df_valid[["M_barionica_kg", "Sigma_barionica", "Sigma_crit", "kappa_barionica"]].describe())