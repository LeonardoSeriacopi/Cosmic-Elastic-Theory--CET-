#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge ATLAS + ZTF photometry for SN 2023zkd into the pipeline format:
source,band,mjd,mag,mag_err,effective_wavelength_angstrom

- ATLAS input: CSV already in pipeline format (e.g., sn2023zkd_atlas_converted.csv)
- ZTF input: raw export from IRSA (CSV/TSV). We try to autodetect columns.

USAGE:
  python merge_photometry_sn2023zkd.py --atlas sn2023zkd_atlas_converted.csv \
      --ztf ztf_sn2023zkd.csv --out all_photometry.csv

NOTE:
- Do NOT apply extinction here; your pipeline handles E(B-V) via --ebv.
- The script keeps only rows with finite mjd and mag.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

EFF = {
    'cyan': 5330.0,
    'orange': 6790.0,
    'ztf_g': 4723.0,
    'ztf_r': 6339.0,
    'w': 6231.0,   # PS1 w (se algum dia entrar)
}

def read_any_csv(path):
    """Read CSV/TSV with best-effort parsing."""
    p = Path(path)
    try:
        return pd.read_csv(p)
    except Exception:
        try:
            return pd.read_csv(p, sep=';')
        except Exception:
            return pd.read_csv(p, sep='\t')

def to_pipeline_from_ztf(df):
    # normalize columns (lowercase)
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # mjd: try common possibilities
    mjd = None
    for c in ['mjd','jd','hjd','mjdobs','jdobs']:
        if c in df.columns:
            mjd = pd.to_numeric(df[c], errors='coerce')
            break
    if mjd is None:
        raise ValueError("ZTF: coluna de tempo (MJD/JD) não encontrada. Colunas: %s" % df.columns.tolist())

    # mag / mag_err
    mag = None
    for c in ['mag','magpsf','mag_aperture','psfmag','mag_auto']:
        if c in df.columns:
            mag = pd.to_numeric(df[c], errors='coerce')
            break
    if mag is None:
        # às vezes a coluna é 'clrpsfmag' em IRSA tables
        for c in [col for col in df.columns if 'mag' in col]:
            try:
                mag = pd.to_numeric(df[c], errors='coerce')
                break
            except Exception:
                pass
    if mag is None:
        raise ValueError("ZTF: coluna de magnitude não encontrada.")

    merr = None
    for c in ['magerr','magerr_psf','magpsferr','sigmag','magerr_auto']:
        if c in df.columns:
            merr = pd.to_numeric(df[c], errors='coerce')
            break
    if merr is None:
        # procurar algo parecido
        cand = [c for c in df.columns if 'mag' in c and 'err' in c]
        if cand:
            merr = pd.to_numeric(df[cand[0]], errors='coerce')

    # band / filter
    band = None
    # IRSA costuma ter 'filtercode' (zg/zr/zi), 'fid' (1=g,2=r,3=i), ou 'filter'
    if 'filtercode' in df.columns:
        band = df['filtercode'].astype(str).str.lower().map(
            {'zg':'ztf_g','zr':'ztf_r','zi':'ztf_i'}
        )
    elif 'fid' in df.columns:
        # 1=g, 2=r, 3=i
        band = df['fid'].map({1:'ztf_g',2:'ztf_r',3:'ztf_i'})
    elif 'filter' in df.columns:
        band = df['filter'].astype(str).str.lower().map(
            {'g':'ztf_g','r':'ztf_r','i':'ztf_i','ztf_g':'ztf_g','ztf_r':'ztf_r','ztf_i':'ztf_i'}
        )
    else:
        # tenta detectar por nomes de colunas
        band = pd.Series(['']*len(df))

    out = pd.DataFrame({
        'source':'ztf',
        'band': band,
        'mjd': mjd,
        'mag': mag,
        'mag_err': merr
    })
    # effective wavelength
    out['effective_wavelength_angstrom'] = out['band'].map(EFF)
    # manter apenas g/r por enquanto
    out = out[out['band'].isin(['ztf_g','ztf_r'])]
    return out

def to_pipeline_from_atlas(df):
    # assumimos que já está no formato do pipeline (como o convertido)
    req = ['source','band','mjd','mag','mag_err','effective_wavelength_angstrom']
    if all(c in df.columns for c in req):
        return df[req].copy()
    # caso tenha vindo bruto, tenta normalizar
    dfc = df.copy()
    dfc.columns = [c.lower() for c in dfc.columns]
    # mjd
    if 'mjd' not in dfc.columns:
        raise ValueError("ATLAS: coluna 'mjd' não encontrada.")
    # band
    if 'band' not in dfc.columns and 'filter' in dfc.columns:
        dfc['band'] = dfc['filter']
    dfc['band'] = dfc['band'].astype(str).str.lower().map({'c':'cyan','cyan':'cyan','o':'orange','orange':'orange'})
    # mag/mag_err
    mag = None
    for c in ['mag','m','mag_c','mag_o']:
        if c in dfc.columns:
            mag = pd.to_numeric(dfc[c], errors='coerce'); break
    merr = None
    for c in ['mag_err','dmag','dm','dmag_c','dmag_o']:
        if c in dfc.columns:
            merr = pd.to_numeric(dfc[c], errors='coerce'); break
    out = pd.DataFrame({
        'source':'atlas',
        'band': dfc['band'],
        'mjd': pd.to_numeric(dfc['mjd'], errors='coerce'),
        'mag': mag,
        'mag_err': merr
    })
    out['effective_wavelength_angstrom'] = out['band'].map(EFF)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas', required=False, help='CSV do ATLAS (já convertido ou bruto).')
    ap.add_argument('--ztf', required=False, help='CSV/TSV da ZTF (IRSA).')
    ap.add_argument('--out', required=True, help='Arquivo de saída CSV combinado.')
    args = ap.parse_args()

    frames = []

    if args.atlas:
        df_a = read_any_csv(args.atlas)
        frames.append(to_pipeline_from_atlas(df_a))

    if args.ztf:
        df_z = read_any_csv(args.ztf)
        frames.append(to_pipeline_from_ztf(df_z))

    if not frames:
        raise SystemExit("Nenhuma entrada fornecida. Use --atlas e/ou --ztf.")

    comb = pd.concat(frames, ignore_index=True)
    # limpar
    comb = comb.replace([np.inf,-np.inf], np.nan)
    comb = comb.dropna(subset=['mjd','mag'])
    # remover duplicatas exatas
    comb = comb.drop_duplicates(subset=['source','band','mjd','mag'], keep='first')
    comb = comb.sort_values('mjd').reset_index(drop=True)

    # salvar
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    comb.to_csv(args.out, index=False)

    # resumo
    print("Salvo:", args.out)
    print("Total linhas:", len(comb))
    print("Contagem por band:\n", comb['band'].value_counts())

if __name__ == "__main__":
    main()
