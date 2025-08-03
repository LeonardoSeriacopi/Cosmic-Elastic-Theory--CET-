import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.constants import G, M_sun
from astropy.coordinates import Distance

# 1. Carregar dados
df = pd.read_csv("PSF_with_KappaObs.csv")

# 2. Cosmologia
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

# 3. Constantes
ML_ratio = 1.5  # razão massa-luminosidade em M_sun / L_sun
L_sun_r = 4.65  # magn. abs. do Sol em banda r
G_value = G.value  # m^3 / kg / s^2

# 4. Função para calcular pressão
def calc_pressure(row):
    try:
        # Distância de luminosidade (Mpc)
        z = row["mean_z"]
        if z <= 0:
            return np.nan
        
        d_ang = cosmo.angular_diameter_distance(z).to(u.kpc).value  # kpc
        mag_r = row["mag_auto_r"]
        kron_arcsec = row["kron_radius"]

        # Luminosidade relativa ao Sol
        M_abs = mag_r - 5 * np.log10(cosmo.luminosity_distance(z).to(u.pc).value / 10)
        L_rel = 10 ** (-0.4 * (M_abs - L_sun_r))

        # Massa bariônica (em kg)
        M = L_rel * ML_ratio * M_sun.value

        # Raio físico (kron_radius em arcsec → rad → kpc)
        R_rad = (kron_arcsec * u.arcsec).to(u.radian).value
        R_kpc = d_ang * R_rad  # em kpc
        R_m = R_kpc * 3.085677581e19  # em metros

        # Pressão dinâmica: P = G M² / (4π R⁴)
        P = G_value * M**2 / (4 * np.pi * R_m**4)
        return P
    except:
        return np.nan

# 5. Calcular coluna 'pressure_dyn'
df["pressure_dyn"] = df.apply(calc_pressure, axis=1)

# 6. Exportar
df.to_csv("PSF_with_Pressure.csv", index=False)
print("✅ Pressão dinâmica calculada e salva em 'PSF_with_Pressure.csv'")


