import pandas as pd
import numpy as np

# --- 1. Carregar os Dados ---
# Carrega o arquivo que já contém a coluna 'Geff_G0_ratio' e 'kappa_barionica'
try:
    df = pd.read_csv("PSF_with_Geff.csv")
except FileNotFoundError:
    print("Erro: O arquivo 'PSF_with_Geff.csv' não foi encontrado.")
    print("Por favor, execute o script 'calculate_Geff.py' primeiro.")
    exit()

# --- 2. Verificar Colunas Necessárias ---
required_cols = ['kappa_barionica', 'Geff_G0_ratio']
for col in required_cols:
    if col not in df.columns:
        print(f"Erro: A coluna necessária '{col}' não foi encontrada no arquivo.")
        exit()

# --- 3. Calcular o Kappa CET Final ---
# Esta é a previsão teórica completa da TEC:
# a convergência bariônica, modulada pelo fator de acoplamento gravitacional.
df['kappa_cet'] = df['kappa_barionica'] * df['Geff_G0_ratio']

# --- 4. Salvar e Apresentar os Resultados ---
# Salva o DataFrame final com todas as colunas calculadas
df.to_csv("PSF_resultados_finais_CET.csv", index=False)

print("✅ Cálculo final do kappa_cet concluído com sucesso!")
print("\nO resultado foi salvo no arquivo 'PSF_resultados_finais_CET.csv'.")

print("\nEstatísticas descritivas para a sua previsão final 'kappa_cet':")
print(df["kappa_cet"].describe())