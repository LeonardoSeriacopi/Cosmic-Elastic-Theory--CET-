import pandas as pd
import numpy as np

# --- Parâmetros da Teoria Elástica Causal (TEC) ---
# Baseado no seu "Paper IV" e discussões anteriores
beta = 0.33
P_crit = 2.3e-15  # Pa (Pressão Crítica)
P0 = 5.3e-15      # Pa (Escala de Pressão da Transição)

# --- 1. Carregar os Dados ---
# Carrega o arquivo que já contém a coluna 'pressure_dyn'
try:
    df = pd.read_csv("PSF_with_KappaObs_and_Pressao.csv")
except FileNotFoundError:
    print("Erro: O arquivo 'PSF_with_KappaObs_and_Pressao.csv' não foi encontrado.")
    print("Por favor, execute o script 'calculate_P.py' primeiro.")
    exit()

# Verificar se a coluna de pressão existe
if "pressure_dyn" not in df.columns:
    print("Erro: A coluna 'pressure_dyn' não foi encontrada no arquivo.")
    exit()
    
# Remover linhas com valores nulos ou infinitos na pressão para evitar erros
df.dropna(subset=['pressure_dyn'], inplace=True)
df = df[np.isfinite(df['pressure_dyn'])]

# --- 2. Calcular a Razão Geff/G0 ---
# Aplicamos a fórmula de saturação com a função tangente hiperbólica.
# Esta fórmula garante que a correção se comporte bem em todos os regimes de pressão.

P = df['pressure_dyn'].values
df['Geff_G0_ratio'] = 1 + beta * np.tanh((P - P_crit) / P0)

# --- 3. Salvar e Apresentar os Resultados ---
# Salva o DataFrame com a nova coluna
df.to_csv("PSF_with_Geff.csv", index=False)

print("✅ Razão Geff/G0 calculada com sucesso!")
print("\nO resultado foi salvo no arquivo 'PSF_with_Geff.csv'.")

print("\nEstatísticas descritivas para a nova coluna 'Geff_G0_ratio':")
print(df["Geff_G0_ratio"].describe())
