import pandas as pd

# ==============================
# ARQUIVOS
# ==============================
file_peaks   = "kids_dispersion_peaks.csv"
file_centers = "kids_10k_centers.csv"
output_file  = "kids_merged_centers_peaks.csv"

print("==============================")
print("CARREGANDO DADOS")
print("==============================")

peaks = pd.read_csv(file_peaks)
centers = pd.read_csv(file_centers)

print(f"Peaks carregados   : {len(peaks)}")
print(f"Centers carregados : {len(centers)}")

# ==============================
# NORMALIZAÇÃO DE COLUNA ID
# ==============================
peaks.columns = peaks.columns.str.strip()
centers.columns = centers.columns.str.strip()

# ==============================
# MERGE
# ==============================
print("==============================")
print("FAZENDO MERGE POR ID")
print("==============================")

merged = pd.merge(
    centers,
    peaks,
    on="ID",
    how="inner",
    suffixes=("_center", "_peak")
)

print(f"Objetos após merge: {len(merged)}")

# ==============================
# SALVAR
# ==============================
merged.to_csv(output_file, index=False)

print("==============================")
print("FINALIZADO")
print("==============================")
print(f"Arquivo salvo: {output_file}")