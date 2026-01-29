import pandas as pd

# ============================================================
# CET — MERGE BACKGROUND CONSOLIDADO + IDADE ESTRUTURAL
# ============================================================

# Arquivos de entrada
FILE_BACKGROUND = "CET_Raw_Consolidated_By_Galaxy.csv"
FILE_STRUCTURE  = "Inspire_structure_age.csv"

# Arquivo de saída
OUTPUT_FILE = "CET_Background_With_StructuralAge.csv"

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
df_bg = pd.read_csv(FILE_BACKGROUND)
df_st = pd.read_csv(FILE_STRUCTURE)

# Padroniza coluna ID (sanidade)
df_bg['ID'] = df_bg['ID'].astype(str).str.strip()
df_st['ID'] = df_st['ID'].astype(str).str.strip()

# ------------------------------------------------------------
# MERGE (LEFT JOIN: mantém só as galáxias do CET)
# ------------------------------------------------------------
df_merged = pd.merge(
    df_bg,
    df_st,
    on='ID',
    how='left',
    validate='one_to_one'
)

# ------------------------------------------------------------
# CHECAGEM RÁPIDA
# ------------------------------------------------------------
missing = df_merged['Idade Estrutural (logρ)'].isna().sum()

print("===================================")
print(" MERGE FINALIZADO ")
print(" Galáxias:", len(df_merged))
print(" Sem idade estrutural:", missing)
print("===================================")

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------
df_merged.to_csv(OUTPUT_FILE, index=False)
print("Arquivo salvo:", OUTPUT_FILE)