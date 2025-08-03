import pandas as pd
import numpy as np

# --- Configurações ---
INPUT_FILE = "PSF_resultados_finais_CET.csv"
OUTPUT_FILE = "PSF_final_for_analysis.csv"

print(f"--- Iniciando cálculo de kappa_DM e F_DM_CET ---")
# CORREÇÃO: Usar a variável INPUT_FILE que contém o nome do arquivo como texto
print(f"Lendo dados de: {INPUT_FILE}")

# --- Carregar Dados ---
try:
    # CORREÇÃO: Usar a variável INPUT_FILE para ler o arquivo
    df = pd.read_csv(INPUT_FILE)
except FileNotFoundError:
    print(f"ERRO: Arquivo '{INPUT_FILE}' não encontrado.")
    print("Certifique-se de que o nome do arquivo está correto e no mesmo diretório.")
    exit()

# --- Verificar Colunas Necessárias ---
required_cols = ['kappa_obs', 'kappa_barionica', 'kappa_cet']
for col in required_cols:
    if col not in df.columns:
        print(f"ERRO: A coluna necessária '{col}' não foi encontrada no arquivo.")
        exit()

# --- Passo 1: Calcular kappa_DM (a anomalia independente da teoria) ---
# Esta é a "quantidade faltante" de lente que o modelo padrão atribui à matéria escura.
df['kappa_DM'] = df['kappa_obs'] - df['kappa_barionica']

# --- Passo 2: Calcular a Métrica-Chave F_DM_CET ---
# Mede a fração do efeito "matéria escura" que a TEC NÃO explica.
# (kappa_cet - kappa_barionica) é o "efeito extra" previsto pela TEC.
# kappa_DM é o "efeito extra" que precisamos explicar.

# Evitar divisão por zero em casos onde kappa_DM é muito pequeno (ruído)
# Usamos um valor pequeno para regularização
epsilon = 1e-9
kappa_DM_reg = df['kappa_DM'].copy()
kappa_DM_reg[np.abs(kappa_DM_reg) < epsilon] = epsilon

df['F_DM_CET'] = 1 - (df['kappa_cet'] - df['kappa_barionica']) / kappa_DM_reg

# --- Passo 3: Salvar e Apresentar Resultados ---
df.to_csv(OUTPUT_FILE, index=False)
# CORREÇÃO: Usar a variável OUTPUT_FILE
print(f"\n✅ Cálculo concluído! Dados salvos em '{OUTPUT_FILE}'.")

print("\nEstatísticas descritivas para as novas colunas:")
print(df[['kappa_DM', 'F_DM_CET']].describe())