import numpy as np
import pandas as pd

# opcional: pra distância luminosa/abs mag
try:
    from astropy.cosmology import Planck18 as cosmo
    ASTROPY_OK = True
except Exception:
    ASTROPY_OK = False


def robust_corr(x, y):
    """Pearson + Spearman (manual simples, sem scipy)."""
    x = np.asarray(x)
    y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 5:
        return np.nan, np.nan

    # Pearson
    xp = x - x.mean()
    yp = y - y.mean()
    pearson = (xp @ yp) / (np.sqrt((xp @ xp) * (yp @ yp)) + 1e-30)

    # Spearman: rank corr
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    xrp = xr - xr.mean()
    yrp = yr - yr.mean()
    spearman = (xrp @ yrp) / (np.sqrt((xrp @ xrp) * (yrp @ yrp)) + 1e-30)

    return pearson, spearman


def make_abs_mag(df, mag_col="MAG_AUTO", z_col="Z_center"):
    """
    Converte magnitude aparente -> absoluta usando Planck18.
    M = m - 5 log10(DL/10pc)
    DL em parsec. Ignora K-correction (proxy bruto mesmo).
    """
    if not ASTROPY_OK:
        print("[WARN] astropy não disponível. Pulando abs mag.")
        return None

    z = df[z_col].to_numpy()
    m = df[mag_col].to_numpy()

    msk = np.isfinite(z) & np.isfinite(m) & (z > 0)
    M = np.full(len(df), np.nan, dtype=float)
    if msk.sum() == 0:
        return M

    dl = cosmo.luminosity_distance(z[msk]).to("pc").value  # parsec
    dist_mod = 5 * (np.log10(dl) - 1)  # 5 log10(DL/10pc)
    M[msk] = m[msk] - dist_mod
    return M


def envelope_by_bins(df, xcol, ycol, nbins=8, q=0.90):
    """
    Calcula envoltória superior: quantil q de y em bins de x.
    Isso vira o 'limite do transbordo' condicionado à massa-proxy.
    """
    x = df[xcol].to_numpy()
    y = df[ycol].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    if len(x) < 20:
        return None

    # bins por quantis do x (mais estável)
    edges = np.quantile(x, np.linspace(0, 1, nbins + 1))
    # evita edges repetidos
    edges = np.unique(edges)
    if len(edges) < 4:
        return None

    rows = []
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = (x >= lo) & (x <= hi) if i == len(edges) - 2 else (x >= lo) & (x < hi)
        if sel.sum() < 10:
            continue
        xx = x[sel]
        yy = y[sel]
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "x_med": np.median(xx),
            "n": int(sel.sum()),
            "y_med": float(np.median(yy)),
            "y_p10": float(np.quantile(yy, 0.10)),
            "y_p90": float(np.quantile(yy, 0.90)),
            f"y_q{int(q*100)}": float(np.quantile(yy, q)),
            "y_max": float(np.max(yy)),
        })

    return pd.DataFrame(rows)


def main():
    # >>> AJUSTE AQUI <<<
    merged_csv = "kids_merged_centers_peaks.csv"  # seu arquivo final (o que você mostrou)
    out_prefix = "anchor_out"

    df = pd.read_csv(merged_csv)

    # Padroniza nomes esperados (caso)
    if "Z_center" not in df.columns and "Z" in df.columns:
        df = df.rename(columns={"Z": "Z_center"})

    # Checagens mínimas
    need = ["ID", "Z_center", "r_peak_Mpc"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Coluna faltando: {c}. Colunas disponíveis: {list(df.columns)}")

    # Proxy mínimo (tosco, mas roda com o que você tem)
    df["mass_proxy_min"] = np.log10(1.0 + df["Z_center"].astype(float))

    # Proxy fotométrico (se houver mag)
    mag_candidates = ["MAG_AUTO", "MAG_GAAP_r", "MAG_GAAP_i", "MAG_GAAP_g"]
    mag_col = None
    for c in mag_candidates:
        if c in df.columns:
            mag_col = c
            break

    if mag_col is not None:
        print(f"[OK] Achei coluna de magnitude: {mag_col}")
        Mabs = make_abs_mag(df, mag_col=mag_col, z_col="Z_center")
        if Mabs is not None:
            df["Mabs_proxy"] = Mabs
            # luminosidade ~ 10^(-0.4 M) (proxy, sem zeropoint solar)
            df["lum_proxy"] = 10 ** (-0.4 * df["Mabs_proxy"])
            # usa log-lum como proxy de massa estelar (ainda tosco, mas melhor que Z)
            df["mass_proxy_lum"] = np.log10(df["lum_proxy"])
    else:
        print("[INFO] Nenhuma coluna de magnitude no merged. Vou rodar só com mass_proxy_min (log10(1+Z)).")
        print("       Se você incluir MAG_AUTO (r) na extração do KiDS, o proxy melhora MUITO.")

    # Diagnóstico rápido
    pear1, spear1 = robust_corr(df["mass_proxy_min"], df["r_peak_Mpc"])
    print("\n=== CORRELAÇÃO (proxy mínimo) ===")
    print(f"Pearson:  {pear1:.4f}")
    print(f"Spearman: {spear1:.4f}")

    # “Âncora” do transbordo: envoltória superior do r_peak condicionado ao proxy
    env_min = envelope_by_bins(df, "mass_proxy_min", "r_peak_Mpc", nbins=8, q=0.90)
    if env_min is not None:
        env_path = f"{out_prefix}_envelope_massproxy_min.csv"
        env_min.to_csv(env_path, index=False)
        print(f"\n[OK] Envoltória (proxy mínimo) salva em: {env_path}")
        print(env_min.head(10).to_string(index=False))

    # Se tiver proxy fotométrico, faz também
    if "mass_proxy_lum" in df.columns:
        pear2, spear2 = robust_corr(df["mass_proxy_lum"], df["r_peak_Mpc"])
        print("\n=== CORRELAÇÃO (proxy fotométrico via luminosidade) ===")
        print(f"Pearson:  {pear2:.4f}")
        print(f"Spearman: {spear2:.4f}")

        env_lum = envelope_by_bins(df, "mass_proxy_lum", "r_peak_Mpc", nbins=8, q=0.90)
        if env_lum is not None:
            env_path2 = f"{out_prefix}_envelope_massproxy_lum.csv"
            env_lum.to_csv(env_path2, index=False)
            print(f"\n[OK] Envoltória (proxy lum) salva em: {env_path2}")
            print(env_lum.head(10).to_string(index=False))

    # salva dataset com proxies
    out_all = f"{out_prefix}_merged_with_proxies.csv"
    df.to_csv(out_all, index=False)
    print(f"\n[OK] Dataset com proxies salvo em: {out_all}")

    print("\nPronto: isso já te dá um 'limite do transbordo' por massa-proxy (envoltória q90 e max por bin).")


if __name__ == "__main__":
    main()