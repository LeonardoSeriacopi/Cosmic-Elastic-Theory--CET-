#!/usr/bin/env python
# eridanus_raw_data_processing.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
import astropy.units as u

# 1. Parâmetros do Eridanus Supervoid
ERIDANUS_CENTER = SkyCoord(ra=65.0*u.degree, dec=-10.0*u.degree, frame='icrs')
RADIUS = 15 * u.degree  # Área de 15 graus de raio
Z_RANGE = [0.001, 0.015]  # Intervalo de redshift

# 2. Buscar dados brutos em catálogos públicos
def fetch_eridanus_data():
    print("🔍 Buscando dados brutos do Eridanus Supervoid...")
    
    # Configurar busca no Vizier
    Vizier.ROW_LIMIT = -1  # Sem limite de linhas
    catalogs = [
        "J/ApJS/240/32",      # DES DR2 Galaxy Catalog
        "I/350/gaiaedr3",     # Gaia EDR3 (movimentos próprios)
        "II/363/unwise"       # UnWISE (infravermelho)
    ]
    
    # Fazer consulta
    result = Vizier.query_region(
        ERIDANUS_CENTER,
        radius=RADIUS,
        catalog=catalogs
    )
    
    # Combinar resultados
    df = pd.DataFrame()
    for table in result:
        if len(table) > 0:
            temp_df = table.to_pandas()
            if 'z' not in temp_df.columns and 'Phot_z' in temp_df.columns:
                temp_df.rename(columns={'Phot_z': 'z'}, inplace=True)
            df = pd.concat([df, temp_df], ignore_index=True)
    
    # Filtrar por redshift
    df = df[(df['z'] >= Z_RANGE[0]) & (df['z'] <= Z_RANGE[1])]
    
    print(f"✅ {len(df)} objetos encontrados na região do Eridanus")
    return df.drop_duplicates(subset=['RAJ2000', 'DEJ2000'])

# 3. Processamento CET dos dados brutos
def process_with_cet(df):
    print("\n⚙️ Processando com pipeline CET...")
    
    # a) Converter para coordenadas cartesianas
    df['x'] = df['z'] * np.cos(np.radians(df['DEJ2000'])) * np.cos(np.radians(df['RAJ2000']))
    df['y'] = df['z'] * np.cos(np.radians(df['DEJ2000'])) * np.sin(np.radians(df['RAJ2000']))
    df['z_coord'] = df['z'] * np.sin(np.radians(df['DEJ2000']))
    
    # b) Calcular densidade local (método CET)
    from scipy.spatial import KDTree
    coords = df[['x','y','z_coord']].values
    tree = KDTree(coords)
    
    # Raio de 5 Mpc para cálculo de densidade
    df['density'] = tree.query_ball_point(coords, r=5.0, return_length=True)[0] - 1
    
    # c) Correção CET de redshift
    ρ_crit = 9.0e-27  # kg/m³
    n = 0.7
    df['z_cet'] = df['z'] * (1 + (df['density'].median()/ρ_crit)**n) / (1 + (df['density']/ρ_crit)**n)
    
    print("✅ Processamento CET completo")
    return df

# 4. Visualização profissional
def create_publication_plots(df):
    print("\n🎨 Gerando figuras para publicação...")
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Figura 1: Distribuição espacial
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    sc = ax[0].scatter(df['RAJ2000'], df['DEJ2000'], c=df['z'], cmap='viridis', s=10, alpha=0.7)
    ax[0].set_title('Distribuição Espacial do Eridanus Supervoid')
    ax[0].set_xlabel('RA (deg)')
    ax[0].set_ylabel('Dec (deg)')
    plt.colorbar(sc, ax=ax[0], label='Redshift (z)')
    
    # Figura 2: Perfil de densidade
    radial_dist = np.sqrt((df['RAJ2000'] - 65.0)**2 + (df['DEJ2000'] + 10.0)**2)
    ax[1].scatter(radial_dist, df['density'], s=8, alpha=0.6, color='darkred')
    ax[1].set_yscale('log')
    ax[1].set_title('Perfil Radial de Densidade')
    ax[1].set_xlabel('Distância do Centro (deg)')
    ax[1].set_ylabel('Densidade Relativa')
    
    plt.tight_layout()
    plt.savefig('eridanus_spatial_density.png', dpi=300)
    
    # Figura 3: Comparação de redshifts
    plt.figure(figsize=(10, 8))
    plt.hist(df['z'], bins=40, alpha=0.7, label='z observado', color='skyblue')
    plt.hist(df['z_cet'], bins=40, alpha=0.7, label='z CET', color='salmon')
    plt.xlabel('Redshift')
    plt.ylabel('Contagem')
    plt.title('Distribuição de Redshifts no Eridanus')
    plt.legend()
    plt.savefig('eridanus_redshift_comparison.png', dpi=300)
    
    print("✅ Figuras salvas como PNG de alta resolução")

# 5. Salvar dados processados
def save_processed_data(df):
    output_file = 'eridanus_processed_data.fits'
    from astropy.table import Table
    t = Table.from_pandas(df)
    t.write(output_file, format='fits', overwrite=True)
    print(f"\n💾 Dados processados salvos em {output_file}")

if __name__ == "__main__":
    # Executar pipeline completo
    raw_df = fetch_eridanus_data()
    processed_df = process_with_cet(raw_df)
    create_publication_plots(processed_df)
    save_processed_data(processed_df)
    
    print("\n" + "="*50)
    print("PROCESSAMENTO COMPLETO! Dados do Eridanus prontos para análise")
    print("="*50)