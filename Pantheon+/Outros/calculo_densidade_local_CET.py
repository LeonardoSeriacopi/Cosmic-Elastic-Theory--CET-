# Código em execução neste momento
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.stats import gaussian_kde

# 1. Baixar dados Pantheon+SH0ES
url = "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat"
df = pd.read_csv(url, delim_whitespace=True)

# 2. Calcular coordenadas cartesianas (exemplo para primeira entrada)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
df['r'] = cosmo.comoving_distance(df['zHD']).value  # em Mpc
# [Conversão RA/DEC/r para x,y,z - código completo em execução]

# 3. Calcular densidade via KDE 3D
positions = np.vstack([df['x'], df['y'], df['z']])
kde = gaussian_kde(positions, bw_method='scott')
df['rho_kde'] = kde(positions)
df['densidade'] = df['rho_kde'] / 9e-27  # Normalização física

# 4. Análise CET (in progress)