import pandas as pd
import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from astropy.constants import G

# 1. Carregar dados
df = pd.read_csv("PSF_final_limpado.csv")

# 2. Garantir que as colunas necessárias existem
for col in ["M_barionica_kg", "kron_radius", "mean_z_x"]:
    assert col in df.columns, f"Coluna '{col}' não encontrada no DataFrame"

# 3. Constantes físicas
pixel_scale_arcsec = 0.263  # escala de pixel DES em arcsec/pixel
arcsec_to_rad = (1/3600) * (np.pi / 180)  # arcsec para radianos

# 4. Calcular o raio físico (R) da lente para cada objeto
z = df["mean_z_x"].values
# Adicionar um pequeno valor a z=0 para evitar erros em D_ang, se houver z=0
z[z == 0] = 1e-9 
D_ang = cosmo.angular_diameter_distance(z).to(u.m).value  # em metros
r_pixel = df["kron_radius"].values  # raio em pixels
r_rad = r_pixel * pixel_scale_arcsec * arcsec_to_rad  # raio em radianos
R_fisico = D_ang * r_rad  # raio físico em metros

# 5. Massa
M = df["M_barionica_kg"].values  # massa em kg

# 6. Calcular pressão dinâmica (em SI: N/m^2 = kg/(m·s²))
# --- CORREÇÃO APLICADA AQUI: Usar ** para exponenciação ---
P_dyn = (G.value * M**2) / (4 * np.pi * R_fisico**4)

# 7. Adicionar coluna ao DataFrame
df["pressure_dyn"] = P_dyn

# 8. Salvar novo arquivo
df.to_csv("PSF_with_KappaObs_and_Pressao.csv", index=False)

print("✅ Pressão dinâmica calculada e salva como 'pressure_dyn'.")
print("\nEstatísticas da nova coluna de pressão:")
print(df["pressure_dyn"].describe())
