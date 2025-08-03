import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u

# Carrega o catálogo com redshifts e magnitudes
df = pd.read_csv("PSF_with_Geff_and_photoz.csv")

# Filtrar objetos com class_star_i < 0.9 para manter apenas galáxias
df = df[df["class_star_i"] < 0.9].copy()

# Cosmologia padrão
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# Usar z_mc como o redshift principal
df["z"] = df["z_mc_y"]

# Calcular distância de luminosidade (em Mpc)
df["DL_Mpc"] = cosmo.luminosity_distance(df["z"]).to(u.Mpc).value

# Aplicar K-correção aproximada na banda i
df["Kcorr_i"] = 2.5 * np.log10(1 + df["z"])

# Calcular magnitude absoluta na banda i
df["M_i"] = df["mag_auto_i"] - 5 * np.log10(df["DL_Mpc"] * 1e6 / 10) - df["Kcorr_i"]

# Luminosidade solar na banda i
M_sun_i = 4.58
df["L_i"] = 10 ** ((M_sun_i - df["M_i"]) / 2.5)  # Em unidades de L_sun

# Modulação da razão massa-luminosidade via bulge_fraction
df["ML_ratio"] = 0.5 + 1.5 * df["bulge_fraction"]

# Estimativa da massa bariônica (em M_sun e em kg)
df["M_barionica_Msun"] = df["ML_ratio"] * df["L_i"]
df["M_barionica_kg"] = df["M_barionica_Msun"] * 1.989e30

# Salva o resultado
df.to_csv("PSF_com_massa_barionica.csv", index=False)

print("✅ Massa bariônica calculada e salva em 'PSF_com_massa_barionica.csv'.")