# cet_env_diagnostics.py
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ============== CONFIG ==============
IN_CSV  = "pairs_merged_env.csv"     # arquivo merged
OUTDIR  = "cet_env_out"              # pasta de saída

# nomes de colunas
COL_ENV = "env_density_kNN"          # densidade local (kNN)
COL_NTOT= "ntot_r10"                 # contagem simples de vizinhos no raio

# métricas de "instabilidade" (ajuste se os nomes diferirem)
COL_SEP = "sep_AU"
COL_DPM = "dpm_masyr"
COL_DPLX= "dparallax_mas"

# opções
N_BINS           = 8                 # número de bins em env_density
BOOTSTRAP_ITERS  = 500               # bootstrap para erro em variância
MAKE_PLOTS       = True              # salvar gráficos
SAVE_SUMMARY_CSV = True              # salvar CSVs de resumo
PLOT_FORMAT      = "png"             # "png" | "pdf" etc.

# ====================================

os.makedirs(OUTDIR, exist_ok=True)

# ---------- Load ----------
df = pd.read_csv(IN_CSV)
n0 = len(df)
print(f"[INFO] Linhas: {n0}, Colunas: {len(df.columns)}")

# check columns
for c in [COL_ENV, COL_SEP, COL_DPM, COL_DPLX]:
    if c not in df.columns:
        raise ValueError(f"Coluna '{c}' não encontrada no dataset.")

# dropna básicos nas colunas usadas
use_cols = [COL_ENV, COL_SEP, COL_DPM, COL_DPLX]
df_clean = df.dropna(subset=use_cols).copy()
print(f"[INFO] Após dropna mínimos: {len(df_clean)} linhas (perda {n0-len(df_clean)})")

# ---------- Bin por densidade (quantis) ----------
quantiles = np.linspace(0, 1, N_BINS+1)
edges = df_clean[COL_ENV].quantile(quantiles).values
# garantir monotonidade estrita
edges = np.unique(np.round(edges, 6))
if len(edges) < N_BINS+1:
    # fallback para bins uniformes
    edges = np.linspace(df_clean[COL_ENV].min(), df_clean[COL_ENV].max(), N_BINS+1)

# cortar em bins
labels = [f"Q{i+1}" for i in range(len(edges)-1)]
df_clean["env_bin"] = pd.cut(df_clean[COL_ENV], bins=edges, labels=labels, include_lowest=True)
print(f"[INFO] Bins de densidade ({len(labels)}):", list(zip(labels, zip(edges[:-1], edges[1:]))))

# ---------- Funções auxiliares ----------
def bootstrap_var(x, n_iter=BOOTSTRAP_ITERS, random_state=42):
    rng = np.random.default_rng(random_state)
    x = np.asarray(x)
    x = x[~np.isnan(x)]
    if len(x) < 5:
        return np.nan, np.nan
    vars_ = []
    for _ in range(n_iter):
        s = rng.choice(x, size=len(x), replace=True)
        vars_.append(np.var(s, ddof=1))
    vars_ = np.array(vars_)
    return np.nanmean(vars_), np.nanstd(vars_)

def summarize_by_bin(df_in, value_col, agg_name):
    rows = []
    for lb in labels:
        sub = df_in.loc[df_in["env_bin"] == lb, value_col]
        n = sub.notna().sum()
        if n == 0:
            rows.append({"env_bin": lb, "n": 0, f"mean_{agg_name}": np.nan,
                         f"var_{agg_name}": np.nan, f"var_{agg_name}_se": np.nan})
            continue
        m = sub.mean()
        v, v_se = bootstrap_var(sub.values)
        rows.append({"env_bin": lb, "n": n, f"mean_{agg_name}": m,
                     f"var_{agg_name}": v, f"var_{agg_name}_se": v_se})
    out = pd.DataFrame(rows)
    # anexa centro do bin para plot contínuo
    mids = []
    for lb in out["env_bin"]:
        i = labels.index(lb)
        mids.append(0.5*(edges[i] + edges[i+1]))
    out["env_mid"] = mids
    return out.sort_values("env_mid")

# ---------- Resumos ----------
sum_sep = summarize_by_bin(df_clean, COL_SEP, "sep")
sum_dpm = summarize_by_bin(df_clean, COL_DPM, "dpm")
sum_dplx= summarize_by_bin(df_clean, COL_DPLX, "dplx")

# merge dos resumos por bin
summary = sum_sep.merge(sum_dpm, on=["env_bin","env_mid"], how="outer")\
                 .merge(sum_dplx, on=["env_bin","env_mid"], how="outer")

# ---------- Localizar picos de variância (assinatura de susceptibilidade) ----------
def locate_peak(df_sum, var_col):
    s = df_sum.dropna(subset=[var_col, "env_mid"]).sort_values("env_mid")
    if s.empty:
        return None
    idx = s[var_col].idxmax()
    return s.loc[idx, ["env_mid", var_col, "env_bin"]].to_dict()

peak_sep  = locate_peak(sum_sep,  "var_sep")
peak_dpm  = locate_peak(sum_dpm,  "var_dpm")
peak_dplx = locate_peak(sum_dplx, "var_dplx")

print("[INFO] Pico variância (sep_AU): ", peak_sep)
print("[INFO] Pico variância (dpm):    ", peak_dpm)
print("[INFO] Pico variância (dplx):   ", peak_dplx)

# ---------- Plots ----------
if MAKE_PLOTS:
    sns.set(style="whitegrid", context="talk")

    # 1) variâncias vs densidade (com erro bootstrap)
    fig, ax = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
    ax[0].errorbar(sum_sep["env_mid"], sum_sep["var_sep"], yerr=sum_sep["var_sep_se"], fmt="o-", label="Var(sep_AU)")
    ax[1].errorbar(sum_dpm["env_mid"], sum_dpm["var_dpm"], yerr=sum_dpm["var_dpm_se"], fmt="o-", color="C1", label="Var(dpm_masyr)")
    ax[2].errorbar(sum_dplx["env_mid"], sum_dplx["var_dplx"], yerr=sum_dplx["var_dplx_se"], fmt="o-", color="C2", label="Var(dparallax_mas)")

    ax[0].set_ylabel("Var(sep_AU)")
    ax[1].set_ylabel("Var(dpm_masyr)")
    ax[2].set_ylabel("Var(dparallax_mas)")
    ax[2].set_xlabel("Densidade local (env_density_kNN)")

    for a in ax: a.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUTDIR, f"variance_vs_env.{PLOT_FORMAT}"), dpi=150)

    # 2) dispersão: sep_AU vs densidade, colorido por dpm
    fig2, ax2 = plt.subplots(1, 1, figsize=(9, 7))
    sc = ax2.scatter(df_clean[COL_ENV], df_clean[COL_SEP], c=df_clean[COL_DPM],
                     s=12, cmap="viridis", alpha=0.7)
    cb = plt.colorbar(sc, ax=ax2)
    cb.set_label(COL_DPM)
    ax2.set_xlabel(COL_ENV)
    ax2.set_ylabel(COL_SEP)
    ax2.set_title("sep_AU vs densidade (cor = dpm_masyr)")
    fig2.savefig(os.path.join(OUTDIR, f"scatter_sep_vs_env_col_dpm.{PLOT_FORMAT}"), dpi=150)

    # 3) boxplots por bin para ver alargamento (instabilidade)
    fig3, ax3 = plt.subplots(1, 3, figsize=(14, 6), sharex=False)
    sns.boxplot(data=df_clean, x="env_bin", y=COL_SEP, ax=ax3[0])
    sns.boxplot(data=df_clean, x="env_bin", y=COL_DPM, ax=ax3[1])
    sns.boxplot(data=df_clean, x="env_bin", y=COL_DPLX, ax=ax3[2])
    ax3[0].set_title("sep_AU por bin")
    ax3[1].set_title("dpm_masyr por bin")
    ax3[2].set_title("dparallax_mas por bin")
    for a in ax3:
        a.set_xlabel("Bins de densidade (quantis)")
    plt.tight_layout()
    fig3.savefig(os.path.join(OUTDIR, f"boxplots_by_env_bin.{PLOT_FORMAT}"), dpi=150)

# ---------- Save summaries ----------
if SAVE_SUMMARY_CSV:
    summary.to_csv(os.path.join(OUTDIR, "summary_by_env_bin.csv"), index=False)
    sum_sep.to_csv(os.path.join(OUTDIR, "summary_var_sep.csv"), index=False)
    sum_dpm.to_csv(os.path.join(OUTDIR, "summary_var_dpm.csv"), index=False)
    sum_dplx.to_csv(os.path.join(OUTDIR, "summary_var_dplx.csv"), index=False)

print("[DONE] Saídas em:", OUTDIR)