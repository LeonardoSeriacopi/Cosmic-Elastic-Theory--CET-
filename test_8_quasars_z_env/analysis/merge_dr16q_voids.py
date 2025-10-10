# merge_dr16q_voids_fix.py
import pandas as pd

QSO_CSV   = "dr16q_catalog.csv"      # catálogo “limpo” de quasares
VOID_CSV  = "void_catalog.csv"      # catálogo de voids (ou outro)
OUT_CSV   = "dr16q_merged_voidflag.csv"

# ---- 1. carrega e padroniza -------------------------------------------------
qso   = pd.read_csv(QSO_CSV)
voids = pd.read_csv(VOID_CSV)

# tira b' ... ' e espaços extras em AMBOS os catálogos
for df in (qso, voids):
    df['SDSS_NAME'] = (
        df['SDSS_NAME']
          .astype(str)                 # garante string
          .str.strip()                 # remove \n, espaços
          .str.replace(r"^b'|'$", "", regex=True)
          .str.upper()                 # opcional: caixa-alta p/ garantir
    )

# ---- 2. faz merge pelo nome -------------------------------------------------
merged = (
    qso.merge(
        voids,
        on='SDSS_NAME',
        suffixes=('_QSO', '_VOID'),
        how='inner'          # ‘inner’ = só quem existir nos dois
    )
)

# ---- 3. salva & resume ------------------------------------------------------
merged.to_csv(OUT_CSV, index=False)

resumo = merged['inside_void'].value_counts().rename_axis('inside_void').to_frame('count')
print(resumo)
print(f"✔ Catálogo salvo em: {OUT_CSV}  (linhas = {len(merged)})")

