import pandas as pd
import numpy as np
from astropy.cosmology import FlatLambdaCDM

# Cosmologia Flat ΛCDM
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def calcular_densidade_segura(row, df):
    """
    Calcula a densidade local baseada em massa estelar adaptativa.
    """
    z = row['zCMB']
    # Distância comovível (em Mpc)
    d = cosmo.comoving_distance(z).value if z > 0.1 else (3e5 * z) / 70
    # Volume adaptativo: raio depende de z
    raio = 0.5 if z > 0.05 else 1.0
    volume = (4/3) * np.pi * raio**3
    # Massa estelar: se MU disponível, caso contrário média em z±0.01
    if 'MU' in row and not np.isnan(row['MU']):
        massa = 10**(0.4 * (4.77 - row['MU'])) * 1e10
    else:
        vizinhos = df[(df['zCMB'] >= z - 0.01) & (df['zCMB'] <= z + 0.01)]
        massa = vizinhos['MU'].apply(lambda x: 10**(0.4 * (4.77 - x))).mean() * 1e10
    return massa / volume


def compute_z_cet(z_obs, densidade_local, k=1.0, rho0=1.0):
    """
    Calcula o redshift previsto pela CET (z_cet) com base no redshift observado e na densidade local.
    Substitua o corpo desta função pela fórmula real.
    """
    # Exemplo genérico:
    # return (1 + z_obs) * (1 + k * (densidade_local / rho0)) - 1
    raise NotImplementedError("Implementar a fórmula de z_cet aqui")


if __name__ == "__main__":
    # Leitura dos dados brutos
    df = pd.read_csv("data/raw/pantheon_shoes_raw.csv")

    # Cálculo da densidade local
    df['densidade_local'] = df.apply(lambda row: calcular_densidade_segura(row, df), axis=1)

    # Cálculo do z_cet
    df['z_cet'] = df.apply(lambda row: compute_z_cet(row['zCMB'], row['densidade_local']), axis=1)

    # Salvando os resultados processados
    df.to_csv("data/processed/pantheon_shoes_with_density_and_zcet.csv", index=False)
    print("Processamento concluído: outputs em data/processed/pantheon_shoes_with_density_and_zcet.csv")
