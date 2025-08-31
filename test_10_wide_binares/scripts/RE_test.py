# RE_test.py
import numpy as np
import pandas as pd

from sklearn.utils import resample

# ===== CONFIG =====
IN_CSV  = "pairs_merged_env.csv"
OUT_CSV = "re_summary.csv"
N_BINS  = 6         # número de bins em densidade (quantis)
N_BOOT  = 500       # bootstrap para erro de variância
N_PLACEBO = 100     # quantas vezes rodar placebo (densidade embaralhada)
# ==================

df = pd.read_csv(IN_CSV)

# checar colunas
for col in ["env_density_kNN", "sep_AU", "dpm_masyr"]:
    if col not in df.columns:
        raise ValueError(f"Coluna {col} não encontrada!")

# índice dissipativo: alongamento
df["S1"] = df["sep_AU"] * df["dpm_masyr"]

# bins de densidade
df["bin"] = pd.qcut(df["env_density_kNN"], N_BINS, labels=False)

def bootstrap_var(x, n_iter=500):
    if len(x) < 5: return np.nan, np.nan
    vars_ = []
    for _ in range(n_iter):
        s = resample(x, replace=True, n_samples=len(x))
        vars_.append(np.var(s, ddof=1))
    return np.mean(vars_), np.std(vars_)

rows = []
for b in range(N_BINS):
    sub = df.loc[df["bin"]==b]
    if sub.empty: continue
    m = sub["S1"].mean()
    v, v_se = bootstrap_var(sub["S1"].values, N_BOOT)
    tail90 = (sub["sep_AU"] > np.percentile(df["sep_AU"],90)).mean()
    rows.append({"bin": b,
                 "dens_mid": sub["env_density_kNN"].median(),
                 "mean_S1": m, "var_S1": v, "var_S1_se": v_se,
                 "tail_sep90": tail90,
                 "n": len(sub)})
summary = pd.DataFrame(rows)

# -------- Placebo: embaralhar densidade --------
def run_placebo(df, n_bins):
    tmp = df.copy()
    tmp["bin"] = np.random.permutation(tmp["bin"].values)
    rows=[]
    for b in range(n_bins):
        sub = tmp.loc[tmp["bin"]==b]
        if sub.empty: continue
        m = sub["S1"].mean()
        v = sub["S1"].var(ddof=1)
        rows.append(m)
    return np.mean(rows)

placebo_means = [run_placebo(df, N_BINS) for _ in range(N_PLACEBO)]
placebo_mean = np.mean(placebo_means)
obs_mean_lowdens = summary.iloc[0]["mean_S1"]

print("[INFO] Média S1 no bin mais baixo:", obs_mean_lowdens)
print("[INFO] Média S1 no placebo médio:", placebo_mean)

# salvar
summary.to_csv(OUT_CSV, index=False)
print("[DONE] Resumo salvo em", OUT_CSV)