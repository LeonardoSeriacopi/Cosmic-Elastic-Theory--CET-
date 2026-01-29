import os
import glob
import numpy as np
import pandas as pd

# ============================================================
# CET — CONSOLIDAÇÃO FÍSICA POR GALÁXIA (AUTOMATIC SEARCH)
# ============================================================

# Busca no diretório onde o script está sendo executado
BASE_DIR = os.getcwd()
OUTPUT_FILE = "CET_Raw_Consolidated_By_Galaxy.csv"

# Colunas que identificam um arquivo de sensores bruto da CET
REQUIRED_COLUMNS = {
    'e1', 'e2', 'weight', 'dist_deg', 'Z_LENS', 'gamma_t'
}

rows = []

# Busca todos os CSVs na pasta raiz
csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))

print(f"--- Iniciando Busca Automática em: {BASE_DIR} ---")

for file_path in csv_files:
    filename = os.path.basename(file_path)
    
    # Pula os arquivos que sabemos que são tabelas de suporte ou o próprio output
    if filename in ["Inspire_structure_age.csv", "CET_Pico_Mpc_Results.csv", OUTPUT_FILE, "CET_Relics_Peak_Summary_Raw.csv"]:
        continue
        
    try:
        # Lê apenas o cabeçalho primeiro para validar se é um arquivo de sensores
        header = pd.read_csv(file_path, nrows=0).columns
        if not REQUIRED_COLUMNS.issubset(set(header)):
            continue # Se não tem as colunas de sensores, pula (ex: tabelas de resultados)

        df = pd.read_csv(file_path)
        galaxy_id = os.path.splitext(filename)[0]

        # ----------------------------
        # MÉTRICAS FÍSICAS (RMS & DISPERSÃO)
        # ----------------------------
        # Usando RMS para e_mod conforme discutimos para maior rigor
        e_mod = np.sqrt(df['e1']**2 + df['e2']**2)

        weights = df['weight'].values
        weights = np.where(weights <= 0, 1.0, weights)

        # Média ponderada da dispersão (Sinal de tensão no vácuo)
        mean_disp = np.average(e_mod, weights=weights)
        
        # Estatísticas de alcance do sensor
        r_med = np.median(df['dist_deg'])
        r_max = np.nanmax(df['dist_deg'])
        z_lens = df['Z_LENS'].iloc[0]
        n_sources = len(df)

        rows.append({
            'ID': galaxy_id,
            'N_sources': n_sources,
            'z_lens': z_lens,
            'mean_dispersion_rms': mean_disp,
            'r_median_deg': r_med,
            'r_max_deg': r_max
        })

        print(f"[OK] {galaxy_id}: {n_sources} sensores processados.")

    except Exception as e:
        print(f"[ERRO] Falha ao processar {filename}: {e}")

# ============================================================
# SALVAR CONSOLIDAÇÃO
# ============================================================
if rows:
    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print("\n" + "="*35)
    print(f" CONSOLIDAÇÃO FINALIZADA")
    print(f" Arquivo: {OUTPUT_FILE}")
    print(f" Galáxias detectadas: {len(df_out)}")
    print("="*35)
else:
    print("\n[AVISO] Nenhuma galáxia com dados brutos foi encontrada.")