
import pandas as pd
import numpy as np

# Constantes
M_solar_AB = 4.83  # Magnitude solar no sistema AB
M_solar_kg = 1.989e30  # Massa do Sol em kg
radius_kpc = 50
radius_m = radius_kpc * 3.086e19

def calcular_densidade_local(df, mag_col='mb_corr'):
    df = df.copy()
    df['stellar_mass_kg'] = 10 ** (0.4 * (M_solar_AB - df[mag_col])) * M_solar_kg
    volume_m3 = (4/3) * np.pi * (radius_m ** 3)
    df['density_local'] = df['stellar_mass_kg'] / volume_m3
    return df

# Exemplo de uso
if __name__ == "__main__":
    df = pd.read_csv("CEERS_catalog_filtrado.csv")
    df_result = calcular_densidade_local(df)
    df_result.to_csv("CEERS_com_densidade.csv", index=False)
