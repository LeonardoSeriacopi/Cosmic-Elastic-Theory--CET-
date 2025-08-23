# -*- coding: utf-8 -*-
# Uso:
#   python probe_tmax.py posterior_with_density.csv results_tmax
#
# Saídas (na pasta outdir):
#   - base_ols_summary.txt           (resumo OLS base)
#   - nonlin_ols_summary.txt         (resumo OLS com termo quadrático)
#   - cap_grid_search.txt            (melhor cap e ganho de R2)
#   - metrics.csv                    (R2, ΔR2 etc.)
#   - coefs_base.csv                 (betas padronizados - modelo base)
#   - coefs_nonlin.csv               (betas padronizados - modelo com não linearidade)
#   - per_event_summary.csv          (resumo por evento)
#   - scatter_SNR_vs_density.png     (dispersão SNR × densidade/LOS)
#   - scatter_SNR_vs_pred.png        (SNR vs SNR_pred antes/depois do cap)
#   - betas_base.png / betas_nonlin.png

import sys, os, math, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

USE_SM = True
try:
    import statsmodels.api as sm
except Exception:
    USE_SM = False
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

INFILE = sys.argv[1] if len(sys.argv) > 1 else "posterior_with_density.csv"
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else "results_tmax"
os.makedirs(OUTDIR, exist_ok=True)

# ---------- Helpers ----------
def deg_or_rad(theta_series):
    m = np.nanmax(theta_series.values)
    return "deg" if m > 2*math.pi else "rad"

def pick_LOS_ratio(df):
    # Usa direto se existir:
    if "losdens_galaxies_per_seg" in df.columns:
        return pd.to_numeric(df["losdens_galaxies_per_seg"], errors="coerce")
    # Reconstrói se tiver contagem e nseg:
    gc = pd.to_numeric(df.get("losdens_galaxy_count", np.nan), errors="coerce")
    ns = pd.to_numeric(df.get("losdens_nseg", np.nan), errors="coerce")
    ratio = gc / ns
    return ratio.replace([np.inf, -np.inf], np.nan)

def pick_env_metric(df):
    """Escolhe uma métrica de ambiente/densidade (ordem de preferência)."""
    candidates = [
        "losdens_mass_density_mean",
        "losdens_lum_density_mean",
        "losdens_number_density_mean",
        "losdens_galaxies_per_seg",
    ]
    for c in candidates:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce"), c
    # fallback: ratio LOS
    r = pick_LOS_ratio(df)
    return r, "los_ratio"

def standardize(s):
    s = pd.to_numeric(s, errors="coerce")
    mu = s.mean()
    sd = s.std(ddof=0)
    return (s - mu) / sd, mu, sd

def fit_ols(y, X, add_const=True):
    if USE_SM:
        X_ = sm.add_constant(X) if add_const else X
        m = sm.OLS(y, X_).fit()
        r2_adj = float(m.rsquared_adj)
        coefs = m.params.to_dict()
        pvals = m.pvalues.to_dict()
        summ = m.summary().as_text()
        return r2_adj, coefs, pvals, summ
    else:
        # sklearn: sem p-valor, usamos R2 adj. e coeficientes
        lr = LinearRegression().fit(X, y)
        yhat = lr.predict(X)
        r2 = r2_score(y, yhat)
        n, k = X.shape
        r2_adj = 1 - (1-r2)*(n-1)/(n-k-1)
        coefs = {c: float(b) for c,b in zip(X.columns, lr.coef_)}
        pvals = {c: np.nan for c in X.columns}
        # intercepto "equivalente" (não exato por não usar add_constant)
        coefs["const"] = float(y.mean() - np.sum([coefs[c]*X[c].mean() for c in X.columns]))
        summ = f"[sklearn] R2_adj={r2_adj:.4f}"
        return r2_adj, coefs, pvals, summ

def save_barplot(coefs_dict, labels_order, outfile, title):
    vals = [coefs_dict.get(l, np.nan) for l in labels_order]
    colors = ["#ff7f0e","#2ca02c","#1f77b4","#d62728","#8c564b"]
    plt.figure(figsize=(6,3))
    plt.bar(labels_order, vals, color=colors[:len(labels_order)])
    plt.axhline(0, color="k", lw=0.7)
    plt.ylabel("Standardized beta")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

# ---------- Load & prepare ----------
print(f"[INFO] Reading {INFILE}")
df = pd.read_csv(INFILE, low_memory=False)

# Campos necessários
needed = ["snr_net","luminosity_distance","theta_jn","chirp_mass","event"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise RuntimeError(f"Missing columns: {missing}")

# SNR alvo
df = df.rename(columns={"snr_net":"SNR"})

# cos(theta_jn)
mode = deg_or_rad(df["theta_jn"])
th = np.deg2rad(df["theta_jn"]) if mode == "deg" else df["theta_jn"]
df["cos_theta_jn"] = np.cos(th)

# log10(DL)
df["log_DL"] = np.log10(pd.to_numeric(df["luminosity_distance"], errors="coerce").replace(0, np.nan))

# chirp_mass
df["chirp_mass"] = pd.to_numeric(df["chirp_mass"], errors="coerce")

# métrica de densidade/ambiente
env_raw, env_name = pick_env_metric(df)
df["_ENV"] = env_raw

# limpeza
keep = ["SNR","log_DL","cos_theta_jn","chirp_mass","_ENV","event"]
df = df[keep].replace([np.inf,-np.inf], np.nan).dropna()
print(f"[OK] valid samples: {len(df):,}  | env={env_name}")

# padronizar y e X
y_raw = pd.to_numeric(df["SNR"], errors="coerce")
X_raw = df[["log_DL","cos_theta_jn","chirp_mass","_ENV"]].copy()

# z-score
yz, mu_y, sd_y = standardize(y_raw)
Xz = pd.DataFrame(index=X_raw.index)
zs_meta = {}
for c in X_raw.columns:
    Xz[c], mu, sd = standardize(X_raw[c])
    zs_meta[c] = {"mean": float(mu), "std": float(sd)}

# ---------- Modelo linear base (padronizado) ----------
r2_base, coefs_base, p_base, summ_base = fit_ols(yz, Xz)
with open(os.path.join(OUTDIR,"base_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summ_base)
pd.DataFrame([{"beta_"+k: v for k,v in coefs_base.items()}]).to_csv(
    os.path.join(OUTDIR,"coefs_base.csv"), index=False
)

# ---------- Proxy de amplitude e não linearidade ----------
# Proxy: predição linear (em escala z-score) -> termo quadrático (amplitude²)
if USE_SM:
    yhat_base = sm.OLS(yz, sm.add_constant(Xz)).fit().predict(sm.add_constant(Xz))
else:
    lr = LinearRegression().fit(Xz, yz)
    yhat_base = lr.predict(Xz)

X_nonlin = Xz.copy()
X_nonlin["amp2"] = pd.Series(yhat_base) ** 2  # termo quadrático

r2_nonlin, coefs_nonlin, p_nonlin, summ_nonlin = fit_ols(yz, X_nonlin)
with open(os.path.join(OUTDIR,"nonlin_ols_summary.txt"), "w", encoding="utf-8") as f:
    f.write(summ_nonlin)
pd.DataFrame([{"beta_"+k: v for k,v in coefs_nonlin.items()}]).to_csv(
    os.path.join(OUTDIR,"coefs_nonlin.csv"), index=False
)

# ---------- Modelo com teto (cap) dependente do ambiente ----------
# Estratégia:
#  1) usa yhat_base (escala z) como SNR_pred
#  2) define Z = padronização da ENV já em Xz["_ENV"]
#  3) faz grid em c0 e c1 e calcula SNR_cap = min(yhat_base, c0 + c1*Z)
#  4) avalia R2(O(yz), O(SNR_cap)) com OLS "y ~ SNR_cap" (ou R2 simples)
Z = Xz["_ENV"].values
yp = np.array(yhat_base)

# grades razoáveis:
c0_grid = np.linspace(0.0, 2.0, 21)     # nível do teto em z-score
c1_grid = np.linspace(-1.0, 1.0, 21)    # inclinação vs ambiente (positivo -> teto ↑ com densidade)

best = {"c0": None, "c1": None, "r2": -np.inf}
for c0 in c0_grid:
    for c1 in c1_grid:
        cap = c0 + c1 * Z
        y_cap = np.minimum(yp, cap)
        # R2 entre yz e y_cap
        if USE_SM:
            r2c, _, _, _ = fit_ols(pd.Series(yz, index=df.index), pd.DataFrame({"ycap":y_cap}, index=df.index), add_const=False)
        else:
            r2c = r2_score(yz, y_cap)
        if r2c > best["r2"]:
            best = {"c0": float(c0), "c1": float(c1), "r2": float(r2c)}

# ganho relativo do cap vs base
delta_r2_cap = best["r2"] - r2_base
with open(os.path.join(OUTDIR,"cap_grid_search.txt"), "w", encoding="utf-8") as f:
    f.write(f"best_c0={best['c0']:.3f}  best_c1={best['c1']:.3f}  R2_adj_cap≈{best['r2']:.4f}\n")
    f.write(f"ΔR2 (cap - base) ≈ {delta_r2_cap:.4f}\n")
    f.write(f"(base R2_adj ≈ {r2_base:.4f})\n")

# ---------- Per-event summary ----------
per_event = df.groupby("event").agg(
    n=("SNR","size"),
    SNR_p50=("SNR","median"),
    ENV_p50=("_ENV","median"),
    logDL_p50=("log_DL","median"),
    cosT_p50=("cos_theta_jn","median"),
    mass_p50=("chirp_mass","median"),
).reset_index().sort_values("n", ascending=False)
per_event.to_csv(os.path.join(OUTDIR, "per_event_summary.csv"), index=False)

# ---------- Figuras ----------
# 1) SNR vs ENV (amostral)
plt.figure(figsize=(6,4))
plt.scatter(df["_ENV"], df["SNR"], s=3, alpha=0.25, color="#1f77b4")
plt.xlabel(f"Environment metric ({env_name})")
plt.ylabel("SNR (ringdown proxy)")
plt.title("SNR vs Environment")
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "scatter_SNR_vs_density.png"), dpi=160)
plt.close()

# 2) SNR vs yhat_base e SNR vs y_cap (sobreposto)
#    transformar y_cap (z) de volta pro espaço do SNR original: y = yz*sd_y + mu_y
c0 = best["c0"]; c1 = best["c1"]
cap = c0 + c1 * Z
y_cap = np.minimum(yp, cap)
SNR_pred = (pd.Series(yp, index=df.index) * sd_y + mu_y)
SNR_cap  = (pd.Series(y_cap, index=df.index) * sd_y + mu_y)

plt.figure(figsize=(6,4))
plt.scatter(SNR_pred, df["SNR"], s=3, alpha=0.2, label="Base pred", color="#888888")
plt.scatter(SNR_cap,  df["SNR"], s=3, alpha=0.2, label="Capped pred", color="#d62728")
plt.xlabel("Predicted SNR")
plt.ylabel("Observed SNR")
plt.title("Observed vs Predicted (base vs capped)")
plt.legend(frameon=False, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, "scatter_SNR_vs_pred.png"), dpi=160)
plt.close()

# 3) Barras de betas (padronizados)
labels_base = ["log_DL","cos_theta_jn","chirp_mass","_ENV"]
save_barplot(coefs_base, labels_base, os.path.join(OUTDIR,"betas_base.png"),
             "Standardized betas (base)")
labels_nonlin = ["log_DL","cos_theta_jn","chirp_mass","_ENV","amp2"]
save_barplot(coefs_nonlin, labels_nonlin, os.path.join(OUTDIR,"betas_nonlin.png"),
             "Standardized betas (nonlinear)")

# ---------- Métricas resumo ----------
metrics = {
    "R2_adj_base": float(r2_base),
    "R2_adj_nonlin": float(r2_nonlin),
    "delta_R2_nonlin_minus_base": float(r2_nonlin - r2_base),
    "R2_cap": float(best["r2"]),
    "delta_R2_cap_minus_base": float(delta_r2_cap),
    "env_metric_used": env_name,
    "n_samples": int(len(df)),
}
pd.DataFrame([metrics]).to_csv(os.path.join(OUTDIR,"metrics.csv"), index=False)

# Também salva um .txt humano
with open(os.path.join(OUTDIR,"metrics.txt"), "w", encoding="utf-8") as f:
    f.write("=== TMax probe summary ===\n")
    for k,v in metrics.items():
        f.write(f"{k}: {v}\n")

print("==== SUMMARY ====")
print(f"R²_adj (base):     {r2_base:.4f}")
print(f"R²_adj (nonlin):   {r2_nonlin:.4f}  Δ={r2_nonlin - r2_base:+.4f}")
print(f"R²_adj (cap):      {best['r2']:.4f}  Δ={delta_r2_cap:+.4f}  (best c0={c0:.2f}, c1={c1:.2f} wrt {env_name})")
print(f"[OK] outputs -> {OUTDIR}")