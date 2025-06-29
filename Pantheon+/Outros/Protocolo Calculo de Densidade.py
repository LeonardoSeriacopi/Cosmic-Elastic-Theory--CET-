import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# Cosmologia
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

def calcular_densidade_segura(row, df):
    z = row['zCMB']
    
    # Distância comovível (em Mpc)
    d = cosmo.comoving_distance(z).value if z > 0.1 else (3e5 * z) / 70
    
    # Volume adaptativo (raio = 1 Mpc para z < 0.05, 0.5 Mpc para z > 0.05)
    raio = 0.5 if z > 0.05 else 1.0
    volume = (4/3) * np.pi * (raio)**3
    
    # Massa estelar aproximada da galáxia hospedeira (se MU existir)
    if 'MU' in row and not np.isnan(row['MU']):
        massa = 10**(0.4 * (4.77 - row['MU'])) * 1e10  # Em massas solares
    else:
        # Interpolação: média de galáxias com z ± 0.01
        z_prox = df[(df['zCMB'] >= z - 0.01) & (df['zCMB'] <= z + 0.01)]
        massa = z_prox['MU'].apply(lambda x: 10**(0.4*(4.77 - x))).mean() * 1e10
    
    densidade = massa / volume
    return densidade

# Aplicar
df = pd.read_excel("Pantheon_SH0ES_selecionados.xlsx")
df['densidade'] = df.apply(lambda x: calcular_densidade_segura(x, df), axis=1)
df.to_excel("densidade_calculada_melhorada.xlsx", index=False)