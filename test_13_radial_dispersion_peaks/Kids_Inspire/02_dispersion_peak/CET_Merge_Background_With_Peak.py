import pandas as pd

# ============================================================
# CET — FINAL MERGE: BACKGROUND + STRUCTURAL AGE + PEAK DATA
# ============================================================

# Arquivos de entrada
FILE_BG   = "CET_Background_With_StructuralAge.csv"
FILE_PEAK = "CET_Peak_Observables.csv"

# Arquivo de saída
OUTPUT_FILE = "consolidated_relic_sample.csv"

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
df_bg = pd.read_csv(FILE_BG)
df_pk = pd.read_csv(FILE_PEAK)

# Padroniza IDs
df_bg['ID'] = df_bg['ID'].astype(str).str.strip()
df_pk['ID'] = df_pk['ID'].astype(str).str.strip()

# ------------------------------------------------------------
# MERGE (LEFT JOIN — mantém todas as galáxias base)
# ------------------------------------------------------------
df_master = pd.merge(
    df_bg,
    df_pk,
    on='ID',
    how='left',
    validate='one_to_one'
)

# ------------------------------------------------------------
# CHECAGEM DE SANIDADE
# ------------------------------------------------------------
n_total = len(df_master)
n_missing_peak = df_master['dist_pico_mpc'].isna().sum()
n_missing_trans = df_master['limite_transbordo_mpc'].isna().sum()

print("===================================")
print(" MERGE FINAL — CET PAPER 5 ")
print(" Galáxias totais:", n_total)
print(" Sem pico:", n_missing_peak)
print(" Sem transbordo:", n_missing_trans)
print("===================================")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
df_master.to_csv(OUTPUT_FILE, index=False)
print("Arquivo salvo:", OUTPUT_FILE)