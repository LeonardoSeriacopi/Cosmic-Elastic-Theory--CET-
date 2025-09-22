# analyze_los_vs_residuals.py
import argparse, math, pandas as pd, numpy as np
from scipy.stats import spearmanr

def thiel_sen(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = len(x)
    if n < 3: return np.nan, np.nan
    # amostra de pares para robustez/velocidade
    idx = np.random.default_rng(0).choice(n, size=min(400, n), replace=False)
    xx, yy = x[idx], y[idx]
    slopes = []
    for i in range(len(xx)):
        dx = (xx[i] - xx)
        mask = np.abs(dx) > 0
        if mask.any():
            slopes.extend( (yy[i]-yy[mask]) / dx[mask] )
    if len(slopes)==0: return np.nan, np.nan
    slope = np.median(slopes)
    intercept = np.median(y - slope*x)
    return float(slope), float(intercept)

def bin_stats(x, y, q=3, label_prefix="Q"):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.nunique() < q: q = max(2, min(int(x.nunique()), q))
    bins = pd.qcut(x, q=q, duplicates="drop")
    out = (pd.DataFrame({"bin": bins, "x": x, "y": y})
             .groupby("bin")
             .agg(x_min=("x","min"), x_max=("x","max"),
                  y_mean=("y","mean"), y_med=("y","median"),
                  y_std=("y","std"), n=("y","size"))
             .reset_index(drop=True))
    out.insert(0, "label", [f"{label_prefix}{i+1}" for i in range(len(out))])
    return out

def make_scatter_png(df, xcol, ycol, png_path):
    import matplotlib.pyplot as plt
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    plt.figure(dpi=110)
    plt.scatter(x, y, s=16, alpha=0.7)
    # linha TS
    m, b = thiel_sen(x, y)
    if math.isfinite(m) and math.isfinite(b):
        xs = np.linspace(np.nanpercentile(x,2), np.nanpercentile(x,98), 50)
        plt.plot(xs, m*xs+b)
    plt.xlabel(xcol); plt.ylabel(ycol); plt.title(f"{ycol} vs {xcol}")
    plt.tight_layout(); plt.savefig(png_path); plt.close()

def make_bins_png(bins_df, png_path, ycol="y_med"):
    import matplotlib.pyplot as plt
    labs = bins_df["label"].astype(str).values
    vals = bins_df[ycol].values
    plt.figure(dpi=110)
    plt.plot(range(1,len(vals)+1), vals, marker="o")
    plt.xticks(range(1,len(vals)+1), labs)
    plt.xlabel("bins LOS"); plt.ylabel(f"{ycol}(ΔD_resid)")
    plt.title(f"{ycol} por bins de LOS")
    plt.tight_layout(); plt.savefig(png_path); plt.close()

def main():
    ap = argparse.ArgumentParser(description="Correlação ΔD_resid vs métricas LOS (DEEP2).")
    ap.add_argument("--csv", required=True, help="Arquivo merged (ceers_resid_with_los.csv)")
    ap.add_argument("--delta-col", default="DeltaD_resid")
    ap.add_argument("--env-cols", default="los_count,los_sum_w,los_sum_w2,los_mean_w,los_median_w")
    ap.add_argument("--out-prefix", default="los_resid")
    ap.add_argument("--bins", type=int, default=3)  # 3=tercis, 4=quartis
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    ycol = args.delta_col
    envs = [c.strip() for c in args.env_cols.split(",") if c.strip()]

    rows = []
    for ecol in envs:
        if ecol not in df.columns: continue
        x = pd.to_numeric(df[ecol], errors="coerce")
        y = pd.to_numeric(df[ycol], errors="coerce")
        ok = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 5: continue
        rho, p = spearmanr(x[ok], y[ok])
        m, b = thiel_sen(x[ok], y[ok])
        rows.append({"env_col": ecol, "spearman_rho": rho, "p_value": p, "thiel_sen_slope": m, "intercept": b, "N": int(ok.sum())})
        # gráficos
        make_scatter_png(df[ok], ecol, ycol, f"{args.out_prefix}_{ecol}_scatter.png")
        bins_df = bin_stats(x[ok], y[ok], q=args.bins, label_prefix="B")
        bins_df.to_csv(f"{args.out_prefix}_{ecol}_bins.csv", index=False)
        make_bins_png(bins_df, f"{args.out_prefix}_{ecol}_bins_med.png", ycol="y_med")

    pd.DataFrame(rows).to_csv(f"{args.out_prefix}_spearman.csv", index=False)
    print(f"[OK] Resultados salvos com prefixo: {args.out_prefix}")
    print(f"[OK] Tabelas: {args.out_prefix}_spearman.csv + *_bins.csv")
    print(f"[OK] Figuras: *_scatter.png e *_bins_med.png")

if __name__ == "__main__":
    main()