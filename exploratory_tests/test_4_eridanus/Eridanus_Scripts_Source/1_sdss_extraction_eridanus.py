import pandas as pd
import numpy as np
from astropy.cosmology import Planck18 as cosmo
from astropy.table import Table

# Carregar o CSV original
df = pd.read_csv("data_eridanus_SDSS.csv")

# Calcular a distância comóvel
df['dist_comov'] = cosmo.comoving_distance(df['z']).value  # em Mpc

# Converter coordenadas esféricas para cartesianas usando distância comóvel
df['x'] = df['dist_comov'] * np.cos(np.radians(df['dec'])) * np.cos(np.radians(df['ra']))
df['y'] = df['dist_comov'] * np.cos(np.radians(df['dec'])) * np.sin(np.radians(df['ra']))
df['z_coord'] = df['dist_comov'] * np.sin(np.radians(df['dec']))

# Salvar como tabela astropy (opcional)
table = Table.from_pandas(df)
table.write("processed_eridanus_comoving.csv", format='csv', overwrite=True)

print("✅ Processing complete. Output saved to 'processed_eridanus_comoving.csv'.")
