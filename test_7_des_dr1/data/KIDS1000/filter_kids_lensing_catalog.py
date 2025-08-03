import pandas as pd

# Carrega o catálogo original (ajuste o caminho conforme necessário)
df = pd.read_csv("kids_lensing_sample_200000.csv")

# Colunas que queremos manter (com nomes seguros)
columns_to_keep = [
    # Identificação
    'SeqNr', 'ID', 'SLID', 'SID',

    # Posição e coordenadas
    'RAJ2000', 'DECJ2000', 'X_WORLD', 'Y_WORLD',

    # Fotometria
    'MAG_AUTO', 'MAGERR_AUTO',
    'MAG_ISO', 'MAGERR_ISO',
    'MAG_ISOCOR', 'MAGERR_ISOCOR',
    'FLUX_AUTO', 'FLUXERR_AUTO',
    'FLUX_ISO', 'FLUXERR_ISO',
    'FLUX_ISOCOR', 'FLUXERR_ISOCOR',

    # Extinção
    'EXTINCTION_u', 'EXTINCTION_g', 'EXTINCTION_r',
    'EXTINCTION_i', 'EXTINCTION_Z',

    # Forma
    'A_IMAGE', 'B_IMAGE', 'THETA_J2000',
    'FWHM_IMAGE', 'S_ELLIPTICITY', 'S_ELONGATION',

    # Classificação e flags úteis
    'CLASS_STAR', 'fitclass', 'star_galaxy_f_probability',

    # Redshift
    'Z_B', 'Z_B_MIN', 'Z_B_MAX',

    # Lensing
    'e1', 'e2', 'weight'
]

# Filtra
filtered_df = df[columns_to_keep]

# Exporta
filtered_df.to_csv("kids_lensing_filtered.csv", index=False)
print(f"Arquivo filtrado salvo com {filtered_df.shape[0]} linhas e {filtered_df.shape[1]} colunas.")