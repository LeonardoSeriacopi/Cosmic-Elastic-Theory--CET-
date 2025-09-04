#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limpeza de fotometria para SN 2023zkd (ATLAS + ZTF)
Entrada: CSV no formato do pipeline:
  source,band,mjd,mag,mag_err,effective_wavelength_angstrom
Saída: phot_all_clean.csv (mesmo formato)
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

VALID_BANDS = {"cyan","orange","ztf_g","ztf_r"}

def mad_sigma_clip(x, sigma=3.0):
    """Retorna máscara True para pontos mantidos após clipping robusto (MAD)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    if not np.isfinite(mad) or mad == 0:
        # sem dispersão detectável => mantém tudo
        return np.isfinite(x)
    # 1.4826 * MAD ~ std para gaussiano
    z = 0. if mad == 0 else (x - med) / (1.4826 * mad)
    return np.isfinite(z) & (np.abs(z) <= sigma)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="phot_all.csv (ATLAS+ZTF)")
    ap.add_argument("--out", required=True, help="phot_all_clean.csv")
    ap.add_argument("--err_max", type=float, default=0.20, help="corte em mag_err (default 0.20 mag)")
    ap.add_argument("--mag_max", type=float, default=21.0, help="corte em magnitude (default 21.0)")
    ap.add_argument("--mjd_min", type=float, default=None, help="(opcional) MJD mínimo")
    ap.add_argument("--mjd_max", type=float, default=None, help="(opcional) MJD máximo")
    ap.add_argument("--sigma", type=float, default=3.0, help="sigma-clipping robusto por banda (default 3.0)")
    args = ap.parse_args()

    df = pd.read_csv(args.inp)
    # normalizar colunas
    df.columns = [c.lower().strip() for c in df.columns]

    req = ['source','band','mjd','mag','mag_err','effective_wavelength_angstrom']
    if not all(c in df.columns for c in req):
        raise SystemExit(f"Colunas inválidas. Esperado: {req}. Encontrado: {df.columns.tolist()}")

    # manter linhas com números finitos
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=['mjd','mag'])

    # cortes básicos
    base_n = len(df)
    df = df[df['band'].isin(VALID_BANDS)]
    if args.err_max is not None:
        df = df[(~df['mag_err'].notna()) | (df['mag_err'] <= args.err_max) | (df['mag_err'] <= 0)]
        # mantém linhas sem mag_err (alguns ATLAS antigos), mas ideal é ter erro
    if args.mag_max is not None:
        df = df[df['mag'] <= args.mag_max]
    if args.mjd_min is not None:
        df = df[df['mjd'] >= args.mjd_min]
    if args.mjd_max is not None:
        df = df[df['mjd'] <= args.mjd_max]

    # sigma-clipping robusto por banda na magnitude (remove outliers fotométricos)
    keep_masks = []
    for band, g in df.groupby('band'):
        m = mad_sigma_clip(g['mag'].values, sigma=args.sigma)
        km = pd.Series(m, index=g.index)
        keep_masks.append(km)
    keep = pd.concat(keep_masks).reindex(df.index).fillna(False)
    df_clean = df[keep].copy()

    # ordenar e salvar
    df_clean = df_clean.sort_values('mjd').reset_index(drop=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(args.out, index=False)

    # resumo
    print("=== LIMPEZA DE FOTOMETRIA ===")
    print(f"Entrada: {args.inp}  |  Linhas: {base_n}")
    print(f"Após cortes + clipping -> {len(df_clean)} linhas")
    print("Contagem por banda (limpo):")
    print(df_clean['band'].value_counts())
    if args.mjd_min or args.mjd_max:
        print("Janela MJD:",
              args.mjd_min if args.mjd_min is not None else "-Inf",
              "→",
              args.mjd_max if args.mjd_max is not None else "+Inf")

if __name__ == "__main__":
    main()