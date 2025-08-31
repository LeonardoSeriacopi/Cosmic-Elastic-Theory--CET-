# fig_dissipation.py
import numpy as np, pandas as pd, matplotlib.pyplot as plt

IN_CSV = "pairs_merged_env.csv"
N_BINS = 8
BOOT   = 500
PLACEBO_ITERS = 300
OUT_PNG = "fig_dissipation_en.png"

df = pd.read_csv(IN_CSV)
for c in ["env_density_kNN", "sep_AU", "dpm_masyr"]:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")
df = df.dropna(subset=["env_density_kNN","sep_AU","dpm_masyr"]).copy()
df["S1"] = df["sep_AU"] * df["dpm_masyr"]

edges = df["env_density_kNN"].quantile(np.linspace(0,1,N_BINS+1)).values
edges = np.unique(np.round(edges, 6))
if len(edges) < N_BINS+1:
    edges = np.linspace(df["env_density_kNN"].min(), df["env_density_kNN"].max(), N_BINS+1)
labels = [f"Q{i+1}" for i in range(len(edges)-1)]
df["env_bin"] = pd.cut(df["env_density_kNN"], bins=edges, labels=labels, include_lowest=True)

def boot_mean(x, n=BOOT, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    x = np.asarray(x)
    vals = []
    for _ in range(n):
        s = rng.choice(x, size=len(x), replace=True)
        vals.append(np.mean(s))
    mu = np.mean(vals)
    lo, hi = np.percentile(vals, [16, 84])
    return mu, lo, hi

def placebo_band(df, labels, iters=PLACEBO_ITERS, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    mus = []
    for _ in range(iters):
        tmp = df.copy()
        tmp["env_bin"] = rng.permutation(tmp["env_bin"].values)
        m = []
        for lb in labels:
            s = tmp.loc[tmp["env_bin"]==lb, "S1"].values
            m.append(np.nan if len(s)==0 else np.mean(s))
        mus.append(m)
    mus = np.array(mus)
    lo = np.nanpercentile(mus, 16, axis=0)
    hi = np.nanpercentile(mus, 84, axis=0)
    mu = np.nanmean(mus, axis=0)
    return mu, lo, hi

env_mid, mean_S1, lo_S1, hi_S1, n_bin, tail_p = [],[],[],[],[],[]
sep_thresh = np.percentile(df["sep_AU"], 90)
for i, lb in enumerate(labels):
    sub = df.loc[df["env_bin"]==lb]
    env_mid.append(0.5*(edges[i]+edges[i+1]))
    if len(sub)==0:
        mean_S1.append(np.nan); lo_S1.append(np.nan); hi_S1.append(np.nan)
        n_bin.append(0); tail_p.append(np.nan)
    else:
        mu, lo, hi = boot_mean(sub["S1"].values)
        mean_S1.append(mu); lo_S1.append(lo); hi_S1.append(hi)
        n_bin.append(len(sub))
        tail_p.append( (sub["sep_AU"]>sep_thresh).mean() )

placebo_mu, placebo_lo, placebo_hi = placebo_band(df, labels)

# -------- FIGURE --------
fig = plt.figure(figsize=(13,8))
gs = fig.add_gridspec(2, 2, height_ratios=[1,1], width_ratios=[1,1], hspace=0.3, wspace=0.25)

# A) schematic: tau(rho) and Geff/G
axA = fig.add_subplot(gs[0,0])
rho = np.linspace(0, 1.8, 500)
m = 8.0
tau = 1.0/(1.0 + np.exp(-m*(rho-1.0)))
axA.plot(rho, tau, 'C0', lw=2)
axA.set_xlabel(r'$\rho / \rho_{\rm crit}$')
axA.set_ylabel(r'$\tau(\rho),\; G_{\rm eff}/G$')
axA.set_title("A) Regimes: dissipative → transition → saturated")
axA.axvspan(0.0, 0.8, color='C0', alpha=0.08, label='Dissipative (RE)')
axA.axvspan(0.8, 1.2, color='C1', alpha=0.10, label='Transition')
axA.axvspan(1.2, 1.8, color='C2', alpha=0.08, label='Saturated')
axA.legend(frameon=False)

# B) mean S1 with CI and placebo band
axB = fig.add_subplot(gs[0,1])
x = np.array(env_mid, dtype=float)
axB.fill_between(x, placebo_lo, placebo_hi, color='gray', alpha=0.25, label='placebo band (16–84%)')
axB.plot(x, placebo_mu, color='gray', ls='--', lw=1)
axB.errorbar(x, mean_S1,
             yerr=[np.array(mean_S1)-np.array(lo_S1), np.array(hi_S1)-np.array(mean_S1)],
             fmt='o-', color='C3', lw=2, ms=6, capsize=4, label=r'$\langle S1\rangle$')
axB.set_xlabel('Local density (env_density_kNN, bin center)')
axB.set_ylabel(r'$\langle S1\rangle = \langle \mathrm{sep\_AU}\cdot \mathrm{dpm\_masyr}\rangle$')
axB.set_title("B) Dissipation: elevated S1 at low densities (RE)")
axB.grid(alpha=0.3); axB.legend(frameon=False)

# C) tail probability (very loose pairs)
axC = fig.add_subplot(gs[1,:])
axC.plot(x, tail_p, 'o-', color='C4', lw=2, ms=6)
axC.set_xlabel('Local density (env_density_kNN, bin center)')
axC.set_ylabel(f'P(sep_AU > p90 = {sep_thresh:.0f})')
axC.set_title("C) Tail of separations: higher probability in dissipative regime")
axC.grid(alpha=0.3)

# annotate sample size
for xi, yi, n in zip(x, mean_S1, n_bin):
    if np.isfinite(xi) and np.isfinite(yi):
        axB.annotate(f"n={n}", (xi, yi), textcoords="offset points", xytext=(0,8),
                     ha='center', fontsize=8)
for xi, yi, n in zip(x, tail_p, n_bin):
    if np.isfinite(xi) and np.isfinite(yi):
        axC.annotate(f"n={n}", (xi, yi), textcoords="offset points", xytext=(0,8),
                     ha='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=180)
print(f"[OK] saved {OUT_PNG}")