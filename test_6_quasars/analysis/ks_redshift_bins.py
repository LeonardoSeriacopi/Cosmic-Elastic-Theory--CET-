import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

# Configuração — arquivo e colunas
csv_file = 'dr16q_merged_voidflag.csv'
z_col = 'Z_QSO'
mi_col = 'M_I'
inside_col = 'inside_void'

# Carregar dados
df = pd.read_csv(csv_file)

# Função para rodar KS por bin
def ks_by_bin(var, bins, label):
    rows = []
    for vmin, vmax in bins:
        mask = (df[var] >= vmin) & (df[var] < vmax)
        in_void = df[mask & (df[inside_col] == 1)][z_col if var != z_col else mi_col].dropna()
        out_void = df[mask & (df[inside_col] == 0)][z_col if var != z_col else mi_col].dropna()
        if len(in_void) > 0 and len(out_void) > 0:
            ks_stat, p_value = ks_2samp(in_void, out_void)
            rows.append({'bin_min': vmin, 'bin_max': vmax, 'N_in': len(in_void), 'N_out': len(out_void),
                         'KS_stat': ks_stat, 'p_value': p_value})
            print(f"{label} {vmin}–{vmax}: N_in={len(in_void)}, N_out={len(out_void)}, KS={ks_stat:.4f}, p={p_value:.2e}")
    return pd.DataFrame(rows)

# Bins de redshift
z_bins = [(0.5, 1.0), (1.0, 2.0), (2.0, 2.6)]

# Bins de magnitude (ajuste como quiser; ex: -29 a -28, -28 a -27, ...)
mi_bins = [(-29, -28), (-28, -27), (-27, -26), (-26, -25)]

# Rodar KS por redshift
print("KS por redshift:")
ks_z = ks_by_bin(z_col, z_bins, label="z")

# Rodar KS por magnitude absoluta
print("\nKS por magnitude M_I:")
ks_mi = ks_by_bin(mi_col, mi_bins, label="M_I")

# Salvar resultados
ks_z.to_csv("ks_results_redshift_bins.csv", index=False)
ks_mi.to_csv("ks_results_mi_bins.csv", index=False)

print("\nSalvos: ks_results_redshift_bins.csv e ks_results_mi_bins.csv")


