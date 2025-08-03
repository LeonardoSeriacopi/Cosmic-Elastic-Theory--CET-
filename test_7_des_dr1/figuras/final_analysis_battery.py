import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Settings ---
INPUT_FILE = "PSF_final_for_analysis.csv" # <<< NOME DO ARQUIVO ATUALIZADO AQUI
P_crit = 2.3e-15  # Pa

print(f"--- Iniciando a Bateria de Testes para o arquivo '{INPUT_FILE}' ---")

# --- Load and Clean Data ---
try:
    df = pd.read_csv(INPUT_FILE)
    key_columns = ['kappa_obs', 'kappa_barionica', 'pressure_dyn', 'Geff_G0_ratio', 'kappa_cet']
    # Adicionar as novas colunas para análise, se existirem
    if 'kappa_DM' in df.columns:
        key_columns.append('kappa_DM')
    if 'F_DM_CET' in df.columns:
        key_columns.append('F_DM_CET')
        
    df.dropna(subset=key_columns, inplace=True)
    df = df[np.isfinite(df[key_columns]).all(axis=1)]
    print(f"Dados carregados com sucesso. Amostra contém {len(df)} objetos válidos.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{INPUT_FILE}' não encontrado. Verifique o nome e o local do arquivo.")
    exit()
except KeyError as e:
    print(f"ERRO: A coluna esperada {e} não foi encontrada no arquivo. Verifique o cabeçalho do CSV.")
    exit()

# ==============================================================================
# --- FASE 1: ANÁLISE EXPLORATÓRIA E VALIDAÇÃO DOS DADOS ---
# ==============================================================================
print("\n--- Fase 1: Análise Exploratória ---")

# Teste 1.1: Histogramas
print("Gerando histogramas...")
for column in key_columns:
    plt.figure(figsize=(10, 6))
    data_to_plot = df[column].copy()
    
    if column == 'pressure_dyn':
        data_to_plot = data_to_plot[data_to_plot > 0]
        if not data_to_plot.empty:
            plt.hist(np.log10(data_to_plot), bins=50, alpha=0.7, color='navy')
            plt.axvline(np.log10(P_crit), color='red', linestyle='--', linewidth=2, label=f'log10(P_crit) = {np.log10(P_crit):.2f}')
            plt.xlabel(f'log10({column} [Pa])')
            plt.legend()
    else:
        plt.hist(data_to_plot, bins=50, alpha=0.7, color='navy')
        plt.xlabel(column)
        
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {column}')
    plt.grid(True, linestyle=':')
    file_name = f'1_histogram_{column}.png'
    plt.savefig(file_name)
    plt.close()
    print(f" -> Histograma '{file_name}' salvo.")

# Teste 1.2: Matriz de Correlação
print("\nGerando matriz de correlação...")
corr_matrix = df[key_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Key Variables')
file_name = '2_correlation_matrix.png'
plt.savefig(file_name)
plt.close()
print(f" -> Matriz de correlação '{file_name}' salva.")

# ==============================================================================
# --- FASE 2: COMPARAÇÃO DIRETA DO PODER PREDITIVO ---
# ==============================================================================
print("\n--- Fase 2: Comparação de Modelos ---")

# Teste 2.1: Gráfico de Dispersão Principal
print("Gerando gráfico de dispersão principal...")
plt.figure(figsize=(8, 8))
plt.hist2d(df['kappa_obs'], df['kappa_cet'], bins=100, cmap='Blues', cmin=1)
plt.colorbar(label='Galaxy Count')
lims = [df[['kappa_obs', 'kappa_cet']].min().min(), df[['kappa_obs', 'kappa_cet']].max().max()]
plt.plot(lims, lims, 'r--', linewidth=2, label='Identity (y=x)')
plt.xlabel('Observed Kappa (kappa_obs)')
plt.ylabel('TEC Predicted Kappa (kappa_cet)')
plt.title('Comparison of TEC Prediction and Observation')
plt.grid(True, linestyle=':')
plt.legend()
plt.xlim(lims)
plt.ylim(lims)
plt.axis('equal')
file_name = '3_scatter_cet_vs_obs.png'
plt.savefig(file_name)
plt.close()
print(f" -> Gráfico de dispersão '{file_name}' salvo.")

# Teste 2.2: Análise de Resíduos
print("\nGerando gráficos de análise de resíduos...")
residuals_cet = df['kappa_cet'] - df['kappa_obs']
plt.figure(figsize=(10, 6))
plt.hist(residuals_cet, bins=100, alpha=0.7, color='green')
plt.axvline(0, color='red', linestyle='--')
plt.xlabel('Residuals (kappa_cet - kappa_obs)')
plt.ylabel('Frequency')
plt.title('Histogram of TEC Model Residuals')
plt.grid(True, linestyle=':')
file_name = '4_histogram_residuals.png'
plt.savefig(file_name)
plt.close()
print(f" -> Histograma de resíduos '{file_name}' salvo.")

# Teste 2.3: Métricas Quantitativas
def calculate_rmse(y_true, y_pred): return np.sqrt(np.mean((y_pred - y_true)**2))
def calculate_bias(y_true, y_pred): return np.mean(y_pred - y_true)

print("\n--- Resultados Quantitativos ---")
print(f"TEC Model (kappa_cet):")
print(f"  RMSE = {calculate_rmse(df['kappa_obs'], df['kappa_cet']):.6f}")
print(f"  Bias = {calculate_bias(df['kappa_obs'], df['kappa_cet']):.6f}")
print("\nSimple Baryonic Model (kappa_barionica):")
print(f"  RMSE = {calculate_rmse(df['kappa_obs'], df['kappa_barionica']):.6f}")
print(f"  Bias = {calculate_bias(df['kappa_obs'], df['kappa_barionica']):.6f}")
print("----------------------------")

# ==============================================================================
# --- FASE 3: TESTE DA HIPÓTESE DA TRANSIÇÃO DE FASE ---
# ==============================================================================
print("\n--- Fase 3: Teste da Hipótese da Transição de Fase ---")

# Teste 3.1: Coeficiente de Variação (CV)
print("Gerando o gráfico do Coeficiente de Variação ('smoking gun')...")
log_p_cv = np.log10(df['pressure_dyn'])
num_bins_cv = 25
bins_cv = np.linspace(log_p_cv.min(), log_p_cv.max(), num_bins_cv + 1)
df['pressure_bin_cv'] = pd.cut(log_p_cv, bins=bins_cv, include_lowest=True, labels=range(num_bins_cv))
grouped_cv = df.groupby('pressure_bin_cv')['kappa_cet'].agg(['mean', 'std', 'count'])
grouped_cv = grouped_cv[grouped_cv['count'] > 10] 
grouped_cv['cv'] = grouped_cv['std'] / grouped_cv['mean']
grouped_cv['bin_center'] = (bins_cv[grouped_cv.index.astype(int)] + bins_cv[grouped_cv.index.astype(int) + 1]) / 2

plt.figure(figsize=(12, 7))
plt.plot(grouped_cv['bin_center'], grouped_cv['cv'], 'o-', label='CV(kappa_cet) from Data')
plt.axvline(np.log10(P_crit), color='red', linestyle='--', linewidth=2, label=f'log10(P_crit) = {np.log10(P_crit):.2f}')
plt.xlabel('log10(pressure_dyn [Pa])')
plt.ylabel('Coefficient of Variation (CV) of kappa_cet')
plt.title('Phase Transition Signature: Kappa Variance vs. Pressure')
plt.grid(True, linestyle=':')
plt.legend()
file_name = '5_smoking_gun_cv_vs_pressure.png'
plt.savefig(file_name)
plt.close()
print(f" -> Gráfico do CV '{file_name}' salvo.")

# Teste 3.2: Resíduos em Função da Pressão
print("\nGerando gráfico de resíduos em função da pressão...")
df['residuos_cet_res'] = df['kappa_cet'] - df['kappa_obs']
df['residuos_barionico_res'] = df['kappa_barionica'] - df['kappa_obs']
mean_residuos_cet = df.groupby('pressure_bin_cv')['residuos_cet_res'].mean()
mean_residuos_barionico = df.groupby('pressure_bin_cv')['residuos_barionico_res'].mean()

plt.figure(figsize=(12, 7))
plt.plot(grouped_cv['bin_center'], mean_residuos_barionico.loc[grouped_cv.index], 'o--', color='orange', label='Baryonic Model Residuals (Binned Mean)')
plt.plot(grouped_cv['bin_center'], mean_residuos_cet.loc[grouped_cv.index], 'o-', color='blue', label='TEC Model Residuals (Binned Mean)')
plt.axhline(0, color='black', linestyle='-')
plt.axvline(np.log10(P_crit), color='red', linestyle='--', label='P_crit')
plt.xlabel('log10(pressure_dyn [Pa])')
plt.ylabel('Mean Error (Prediction - Observation)')
plt.title('Model Performance as a Function of Local Pressure')
plt.legend()
plt.grid(True, linestyle=':')
file_name = '6_residuals_vs_pressure.png'
plt.savefig(file_name)
plt.close()
print(f" -> Gráfico de Resíduos vs. Pressão '{file_name}' salvo.")

print("\n\n✅ BATERIA DE TESTES CONCLUÍDA! Verifique os arquivos .png no diretório.")
