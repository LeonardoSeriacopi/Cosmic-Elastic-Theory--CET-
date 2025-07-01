import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.table import Table

def verify_sample_quality(file_path):
    """Verifica qualidade da amostra CEERS"""
    cat = Table.read(file_path)
    df = cat.to_pandas()
    
    print("\n[VERIFICAÇÃO DE AMOSTRA]")
    print("=" * 50)
    print(f"Total de galáxias no catálogo: {len(df)}")
    print(f"Faixa de redshift: {df['z_phot'].min():.2f} - {df['z_phot'].max():.2f}")
    print(f"Faixa de RA: {df['ra'].min():.2f} - {df['ra'].max():.2f}")
    print(f"Faixa de Dec: {df['dec'].min():.2f} - {df['dec'].max():.2f}")
    
    # Verificação de valores ausentes
    print("\n[VALORES AUSENTES]")
    print(df.isnull().sum())
    
    # Verificação de distribuição de redshift
    plt.figure(figsize=(10, 6))
    plt.hist(df['z_phot'], bins=50, alpha=0.7, color='skyblue')
    plt.axvline(8.7, color='red', linestyle='--', label='Protocluster z=8.7')
    plt.xlabel('Redshift Fotométrico')
    plt.ylabel('Número de Galáxias')
    plt.title('Distribuição de Redshift no CEERS')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('ceers_redshift_distribution.png', dpi=150, bbox_inches='tight')
    
    # Verificação de distribuição espacial
    plt.figure(figsize=(10, 8))
    plt.scatter(df['ra'], df['dec'], s=1, alpha=0.5)
    plt.xlabel('RA [deg]')
    plt.ylabel('Dec [deg]')
    plt.title('Distribuição Espacial CEERS')
    plt.grid(alpha=0.3)
    plt.savefig('ceers_spatial_distribution.png', dpi=150, bbox_inches='tight')
    
    # Verificação de qualidade fotométrica
    if 'phot_quality' in df.columns:
        plt.figure(figsize=(10, 6))
        quality_counts = df['phot_quality'].value_counts()
        plt.bar(quality_counts.index, quality_counts.values, color='purple')
        plt.xlabel('Qualidade Fotométrica')
        plt.ylabel('Contagem')
        plt.title('Distribuição de Qualidade Fotométrica')
        plt.grid(alpha=0.3)
        plt.savefig('ceers_photometric_quality.png', dpi=150, bbox_inches='tight')
    
    print("\nVerificação concluída! Gráficos salvos.")

if __name__ == "__main__":
    verify_sample_quality('CEERS_DR1_catalog.fits')