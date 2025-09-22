import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Settings ---
INPUT_FILE = "PSF_final_for_analysis.csv"
P_crit = 2.3e-15  # Pa

print(f"--- Iniciando a Bateria de Testes Avançados para o arquivo '{INPUT_FILE}' ---")

# --- Load and Clean Data ---
try:
    df = pd.read_csv(INPUT_FILE)
    key_columns = ['kappa_cet', 'pressure_dyn', 'F_DM_CET', 'M_barionica_kg', 'mean_z_x', 'bulge_fraction', 'kappa_obs']
    df.dropna(subset=key_columns, inplace=True)
    df = df[np.isfinite(df[key_columns]).all(axis=1)]
    df = df[df['pressure_dyn'] > 0] # Ensure pressure is positive for log scale
    print(f"Dados carregados com sucesso. Amostra contém {len(df)} objetos válidos.")
except FileNotFoundError:
    print(f"ERRO: Arquivo '{INPUT_FILE}' não encontrado.")
    exit()
except KeyError as e:
    print(f"ERRO: A coluna esperada {e} não foi encontrada no arquivo.")
    exit()

# ==============================================================================
# --- Gráfico 1: Teste da Métrica F_DM_CET (A Transição Sigmoidal) ---
# ==============================================================================
print("\nGerando Gráfico 1: A Transição Sigmoidal de F_DM_CET...")

# Filtrar valores extremos de F_DM_CET que podem ser ruído
df_plot_fdm = df[(df['F_DM_CET'] > -1) & (df['F_DM_CET'] < 2)]
log_p_norm = np.log10(df_plot_fdm['pressure_dyn'] / P_crit)

plt.figure(figsize=(12, 7))
# Usar um scatter plot com baixa opacidade e pontos pequenos para muitos dados
plt.scatter(log_p_norm, df_plot_fdm['F_DM_CET'], alpha=0.1, s=5, c=log_p_norm, cmap='coolwarm')
plt.axvline(0, color='black', linestyle='--', linewidth=2, label=f'P = P_crit')
plt.axhline(0, color='gray', linestyle=':', label='TEC Explains 100% of DM effect')
plt.axhline(1, color='gray', linestyle=':', label='TEC Explains 0% of DM effect (recovers GR)')
plt.colorbar(label='log10(P / P_crit)')
plt.xlabel('Normalized Local Pressure (log10(P / P_crit))')
plt.ylabel('Unexplained DM Fraction (F_DM_CET)')
plt.title('TEC Performance vs. Local Pressure: The Sigmoidal Transition')
plt.ylim(-0.5, 1.5)
plt.legend()
plt.grid(True, linestyle=':')
file_name = '7_sigmoid_transition_F_DM_CET.png'
plt.savefig(file_name)
plt.close()
print(f" -> Gráfico '{file_name}' salvo.")


# ==============================================================================
# --- Gráfico 2: Teste de Robustez da Pressão (Isolando a Variável Chave) ---
# ==============================================================================
print("\nGerando Gráfico 2: Testes de Robustez da Pressão...")

def plot_cv_vs_variable(variable_name, x_label):
    """Função auxiliar para plotar CV(kappa_cet) vs. uma variável."""
    num_bins = 25
    bins = np.linspace(df[variable_name].min(), df[variable_name].max(), num_bins + 1)
    df['temp_bin'] = pd.cut(df[variable_name], bins=bins, include_lowest=True, labels=range(num_bins))
    
    grouped = df.groupby('temp_bin')['kappa_cet'].agg(['mean', 'std', 'count'])
    grouped = grouped[grouped['count'] > 10]
    grouped['cv'] = grouped['std'] / grouped['mean']
    grouped['bin_center'] = (bins[grouped.index.astype(int)] + bins[grouped.index.astype(int) + 1]) / 2

    plt.figure(figsize=(12, 7))
    plt.plot(grouped['bin_center'], grouped['cv'], 'o-')
    plt.xlabel(x_label)
    plt.ylabel('Coefficient of Variation (CV) of kappa_cet')
    plt.title(f'Robustness Check: CV(kappa_cet) vs. {variable_name}')
    plt.grid(True, linestyle=':')
    file_name = f'8_robustness_cv_vs_{variable_name}.png'
    plt.savefig(file_name)
    plt.close()
    print(f" -> Gráfico de robustez '{file_name}' salvo.")

# Gerar gráficos de robustez para Massa e Redshift
plot_cv_vs_variable('M_barionica_kg', 'Baryonic Mass (kg)')
plot_cv_vs_variable('mean_z_x', 'Redshift (z)')


# ==============================================================================
# --- Gráfico 3: Teste de Viés Morfológico ---
# ==============================================================================
print("\nGerando Gráfico 3: Teste de Viés Morfológico...")

# Dividir a amostra
df_disk = df[df['bulge_fraction'] < 0.5].copy()
df_bulge = df[df['bulge_fraction'] >= 0.5].copy()

plt.figure(figsize=(12, 7))

def plot_binned_residuals(subset_df, label, color):
    """Função auxiliar para plotar resíduos em bins de pressão."""
    if subset_df.empty:
        return
    log_p = np.log10(subset_df['pressure_dyn'])
    num_bins = 20
    bins = np.linspace(log_p.min(), log_p.max(), num_bins + 1)
    subset_df['pressure_bin'] = pd.cut(log_p, bins=bins, include_lowest=True, labels=range(num_bins))
    
    subset_df['residuals'] = subset_df['kappa_cet'] - subset_df['kappa_obs']
    binned_residuals = subset_df.groupby('pressure_bin')['residuals'].mean()
    bin_centers = (bins[binned_residuals.index.astype(int)] + bins[binned_residuals.index.astype(int) + 1]) / 2
    
    plt.plot(bin_centers, binned_residuals, 'o--', label=label, color=color)

plot_binned_residuals(df_disk, 'Disk-Dominated Galaxies', 'blue')
plot_binned_residuals(df_bulge, 'Bulge-Dominated Galaxies', 'red')

plt.axhline(0, color='black', linestyle='-')
plt.axvline(np.log10(P_crit), color='gray', linestyle=':', label='P_crit')
plt.xlabel('log10(pressure_dyn [Pa])')
plt.ylabel('Mean Error (kappa_cet - kappa_obs)')
plt.title('Morphological Bias Check: TEC Residuals vs. Pressure')
plt.legend()
plt.grid(True, linestyle=':')
file_name = '9_morphological_bias_check.png'
plt.savefig(file_name)
plt.close()
print(f" -> Gráfico de viés morfológico '{file_name}' salvo.")

print("\n\n✅ BATERIA DE TESTES AVANÇADOS CONCLUÍDA!")
