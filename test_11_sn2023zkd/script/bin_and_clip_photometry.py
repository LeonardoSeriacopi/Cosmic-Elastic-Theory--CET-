#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bin temporal + sigma-clipping para a fotometria do SN 2023zkd.
Usa diretamente phot_all_clean.csv (ATLAS+ZTF limpo) e produz:
  - phot_all_clean_bin{binsize}d.csv
  - phot_all_clean_bin{binsize}d_clip.csv
Depois você pode rodar a pipeline no arquivo *_clip.csv.

Autor: você :)
"""

import argparse, numpy as np, pandas as pd
from pathlib import Path

REQ = {'source','band','mjd','mag','mag_err','effective_wavelength_angstrom'}
VALID_BANDS = {"ztf_g","ztf_r","cyan","orange"}

def mad_clip(x, nsig=2.5):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        return np.isfinite(x)
    z = (x - med) / (1.4826*mad)
    return np.isfinite(z) & (np.abs(z) <= nsig)

def timebin(df, binsize=3.0):
    """bin por banda com média ponderada por 1/err^2 (dias)."""
    out = []
    for band, g in df.groupby('band'):
        g = g.sort_values('mjd').copy()
        if g.empty: continue
        mjd0 = g['mjd'].min()
        bidx = np.floor((g['mjd']-mjd0)/binsize).astype(int)
        for _, gg in g.groupby(bidx):
            w = 1.0/np.clip(gg['mag_err'].values, 1e-6, None)**2
            w /= w.sum()
            mag = float(np.sum(gg['mag'].values*w))
            merr = float(np.sqrt(1.0/np.sum(1.0/np.clip(gg['mag_err'].values,1e-6,None)**2)))
            row = {
                'source': gg['source'].iloc[0],
                'band': band,
                'mjd': float(np.average(gg['mjd'].values, weights=w)),
                'mag': mag,
                'mag_err': merr,
                'effective_wavelength_angstrom': gg['effective_wavelength_angstrom'].iloc[0],
            }
            out.append(row)
    return pd.DataFrame(out)

def color_clip(df, window=7.0, nsig=3.0):
    """clip em cor g-r via rolling mediana no tempo (apenas impacta ZTF g/r)."""
    g = df[df.band=="ztf_g"][["mjd","mag"]].rename(columns={"mag":"g"})
    r = df[df.band=="ztf_r"][["mjd","mag"]].rename(columns={"mag":"r"})
    if g.empty or r.empty:
        df["keep_color"] = True
        return df

    # emparelha g e r por proximidade temporal (≤ 0.6*window)
    g_ = g.copy(); r_ = r.copy()
    g_["key"]=1; r_["key"]=1
    pairs = g_.merge(r_, on="key", suffixes=("_g","_r")).drop(columns="key")
    dt = np.abs(pairs["mjd_g"] - pairs["mjd_r"])
    tol = 0.6*window
    pairs = pairs[dt <= tol].copy()
    if pairs.empty:
        df["keep_color"] = True
        return df

    pairs["mjd"] = 0.5*(pairs["mjd_g"]+pairs["mjd_r"])
    pairs["color"] = pairs["g"] - pairs["r"]
    pairs = pairs.sort_values("mjd").reset_index(drop=True)

    mids = pairs["mjd"].to_numpy()
    col  = pairs["color"].to_numpy()
    med  = np.empty_like(col); med[:] = np.nan
    for i, t in enumerate(mids):
        m = (mids >= t-window/2) & (mids <= t+window/2)
        med[i] = np.nanmedian(col[m])
    resid = col - med
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid)))
    if not np.isfinite(mad) or mad == 0:
        mask_pairs = np.isfinite(resid)
    else:
        z = resid/(1.4826*mad)
        mask_pairs = np.isfinite(z) & (np.abs(z) <= nsig)

    bad_g = set(np.round(pairs.loc[~mask_pairs,"mjd_g"],6))
    bad_r = set(np.round(pairs.loc[~mask_pairs,"mjd_r"],6))

    keep = []
    for _, row in df.iterrows():
        t = round(row["mjd"],6)
        if row["band"]=="ztf_g" and t in bad_g: keep.append(False); continue
        if row["band"]=="ztf_r" and t in bad_r: keep.append(False); continue
        keep.append(True)
    df["keep_color"] = keep
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True, help='phot_all_clean.csv')
    ap.add_argument('--outdir', required=True, help='pasta de saída dos CSVs filtrados')
    ap.add_argument('--binsize', type=float, default=3.0, help='tamanho do bin (dias)')
    ap.add_argument('--sigma_mag', type=float, default=2.5, help='σ MAD por banda')
    ap.add_argument('--color_window', type=float, default=7.0, help='janela (dias) para tendência g-r')
    ap.add_argument('--sigma_color', type=float, default=3.0, help='σ no resíduo de g-r')
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    df.columns = [c.lower().strip() for c in df.columns]
    if not REQ.issubset(df.columns):
        raise SystemExit(f"CSV inválido. Esperado colunas: {sorted(REQ)}")
    df = df[df['band'].isin(VALID_BANDS)].replace([np.inf,-np.inf], np.nan).dropna(subset=['mjd','mag'])

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    # 1) BIN
    dfb = timebin(df, binsize=args.binsize).sort_values('mjd').reset_index(drop=True)
    out_bin = outdir/f'phot_all_clean_bin{int(args.binsize)}d.csv'
    dfb.to_csv(out_bin, index=False)

    # 2) CLIP por banda
    keep = []
    for b, g in dfb.groupby('band'):
        m = mad_clip(g['mag'], nsig=args.sigma_mag)
        keep.append(g.loc[m])
    dfx = pd.concat(keep).sort_values('mjd').reset_index(drop=True)

    # 3) CLIP por cor (g-r) — não afeta ATLAS
    dfx = color_clip(dfx, window=args.color_window, nsig=args.sigma_color)
    dff = dfx[dfx["keep_color"]].drop(columns=["keep_color"], errors="ignore")
    out_clip = outdir/f'phot_all_clean_bin{int(args.binsize)}d_clip.csv'
    dff.to_csv(out_clip, index=False)

    print("=== BIN + CLIP ===")
    print(f"Entrada: {args.inp} -> {len(df)} linhas")
    print(f"Após BIN {args.binsize:.1f} d: {len(dfb)}")
    print(f"Após CLIP: {len(dff)}")
    print("Por banda:")
    print(dff['band'].value_counts())
    print(f"Arquivos gerados:\n  {out_bin}\n  {out_clip}")

if __name__ == '__main__':
    main()