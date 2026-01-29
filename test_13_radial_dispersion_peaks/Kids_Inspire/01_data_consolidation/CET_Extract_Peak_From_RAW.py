import os
import glob
import numpy as np
import pandas as pd

# ============================================================
# CET — PEAK & TRANSBORDO EXTRACTION FROM RAW FILES
# ============================================================

# Pasta raiz (onde estão os CSVs das galáxias)
BASE_DIR = "."

# Parâmetros
PISO_MPC = 0.31
R_MAX_DEG = 0.488        # limite angular comum
N_BINS = 30              # bins radiais
R_MAX_MPC_DEFAULT = 5.0  # fallback se algo der errado

# ------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------

def compute_dispersion(df):
    """Métrica de dispersão (não normalizada)."""
    return np.sqrt(df['e1']**2 + df['e2']**2)

def build_radial_profile(df, r_max_mpc):
    bins = np.linspace(0, r_max_mpc, N_BINS)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    profile = []
    for i in range(len(bins) - 1):
        mask = (df['dist_mpc'] >= bins[i]) & (df['dist_mpc'] < bins[i+1])
        if mask.sum() > 10:
            profile.append(df.loc[mask, 'dispersion'].mean())
        else:
            profile.append(np.nan)

    return bin_centers, np.array(profile)

def find_peak(bin_centers, profile):
    valid = ~np.isnan(profile)
    if valid.sum() < 5:
        return np.nan, np.nan

    idx = np.nanargmax(profile)
    return bin_centers[idx], profile[idx]

def find_transbordo(bin_centers, profile, peak_idx):
    """
    Transbordo = primeiro ponto após o pico onde
    o perfil entra em regime plano/decrescente.
    """
    if peak_idx >= len(profile) - 3:
        return np.nan

    tail = profile[peak_idx+1:]
    diffs = np.diff(tail)

    for i, d in enumerate(diffs):
        if d <= 0:
            return bin_centers[peak_idx + 1 + i]

    return np.nan

# ------------------------------------------------------------
# EXECUÇÃO
# ------------------------------------------------------------

results = []

csv_files = glob.glob(os.path.join(BASE_DIR, "*.csv"))

# Remove arquivos que NÃO são dados brutos
csv_files = [
    f for f in csv_files
    if not f.startswith("CET_") and not f.startswith("Inspire_")
]

print(f"[INFO] {len(csv_files)} arquivos brutos encontrados")

for file_path in csv_files:
    galaxy_id = os.path.splitext(os.path.basename(file_path))[0]

    try:
        df = pd.read_csv(file_path)

        # Checagem mínima
        required = {'e1', 'e2', 'dist_deg'}
        if not required.issubset(df.columns):
            print(f"[SKIP] {galaxy_id}: colunas ausentes")
            continue

        # Conversão angular → Mpc
        if 'limite_transbordo_mpc' in df.columns:
            scale = df['limite_transbordo_mpc'].iloc[0] / R_MAX_DEG
            r_max_mpc = df['limite_transbordo_mpc'].iloc[0]
        else:
            scale = R_MAX_MPC_DEFAULT / R_MAX_DEG
            r_max_mpc = R_MAX_MPC_DEFAULT

        df['dist_mpc'] = df['dist_deg'] * scale
        df['dispersion'] = compute_dispersion(df)

        # Perfil radial
        r_bins, profile = build_radial_profile(df, r_max_mpc)

        # Pico
        dist_pico, val_pico = find_peak(r_bins, profile)
        if np.isnan(dist_pico):
            print(f"[WARN] {galaxy_id}: pico não encontrado")
            continue

        peak_idx = np.nanargmax(profile)

        # Transbordo
        dist_transbordo = find_transbordo(r_bins, profile, peak_idx)

        results.append({
            'ID': galaxy_id,
            'dist_pico_mpc': dist_pico,
            'valor_pico_e': val_pico,
            'limite_transbordo_mpc': dist_transbordo,
            'gap_pico_transbordo': (
                dist_transbordo - dist_pico
                if not np.isnan(dist_transbordo) else np.nan
            )
        })

        print(f"[OK] {galaxy_id}: pico={dist_pico:.2f} Mpc")

    except Exception as e:
        print(f"[ERRO] {galaxy_id}: {e}")

# ------------------------------------------------------------
# SALVAR RESULTADO FINAL
# ------------------------------------------------------------

df_out = pd.DataFrame(results)
df_out.to_csv("CET_Peak_Observables.csv", index=False)

print("\n===================================")
print(" EXTRAÇÃO FINALIZADA ")
print(f" Galáxias processadas: {len(df_out)}")
print(" Arquivo: CET_Peak_Observables.csv")
print("===================================")