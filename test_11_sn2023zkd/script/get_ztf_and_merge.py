#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baixa lightcurves do ZTF (g/r) via IRSA API, converte para o formato do pipeline,
lê o CSV do ATLAS já convertido e gera um único phot_all.csv.

Formato de saída:
source,band,mjd,mag,mag_err,effective_wavelength_angstrom
"""

import argparse
import io
import sys
import requests
import pandas as pd
import numpy as np
from pathlib import Path

EFF = {
    'cyan': 5330.0,
    'orange': 6790.0,
    'ztf_g': 4723.0,
    'ztf_r': 6339.0,
    'ztf_i': 7827.0,
}

IRSA_LC_API = "https://irsa.ipac.caltech.edu/cgi-bin/ZTF/nph_light_curves"

def fetch_ztf_csv(ra, dec, radius_deg=0.0014, bandname="g,r",
                  bad_mask=32768, fmt="csv", timeout=120):
    """Baixa CSV de lightcurves do ZTF (g/r) via API do IRSA."""
    params = {
        "CIRCLE": f"{ra} {dec} {radius_deg}",
        "BANDNAME": bandname,
        "BAD_CATFLAGS_MASK": str(bad_mask),
        "FORMAT": fmt
    }
    url = IRSA_LC_API
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    # Alguns retornos podem vir como text com comentários; pandas lê direto.
    return r.text

def to_pipeline_from_ztf(csv_text):
    """Converte o CSV da API (Lightcurve Program Interface) para o formato do pipeline."""
    # Ler o CSV robustamente (pode vir com espaços em branco)
    df = pd.read_csv(io.StringIO(csv_text))
    if df.empty:
        return pd.DataFrame(columns=['source','band','mjd','mag','mag_err','effective_wavelength_angstrom'])

    # Normalizar nomes
    df.columns = [c.lower().strip() for c in df.columns]

    # Identificar colunas
    # Tempo
    time_col = None
    for c in ['mjd','mjdobs','jd','jdobs']:
        if c in df.columns:
            time_col = c
            break
    if time_col is None:
        raise ValueError(f"Não achei coluna de tempo no ZTF CSV. Colunas: {df.columns.tolist()}")

    # Magnitude e erro
    mag_col = None
    for c in ['mag','magpsf','psfmag','clrpsfmag','apmag','mag_auto']:
        if c in df.columns:
            mag_col = c
            break
    if mag_col is None:
        # fallback: primeira coluna que contenha 'mag' no nome
        cands = [c for c in df.columns if 'mag' in c]
        if cands:
            mag_col = cands[0]
    if mag_col is None:
        raise ValueError("Não achei coluna de magnitude no ZTF CSV.")

    merr_col = None
    for c in ['magerr','magpsferr','magerr_psf','sigmag','magerr_auto']:
        if c in df.columns:
            merr_col = c
            break

    # Banda
    band_series = None
    if 'filtercode' in df.columns:
        # valores típicos: zg, zr, zi
        band_series = df['filtercode'].astype(str).str.lower().map(
            {'zg':'ztf_g','zr':'ztf_r','zi':'ztf_i'}
        )
    elif 'fid' in df.columns:
        # 1=g, 2=r, 3=i
        band_series = df['fid'].map({1:'ztf_g',2:'ztf_r',3:'ztf_i'})
    elif 'filter' in df.columns:
        band_series = df['filter'].astype(str).str.lower().map(
            {'g':'ztf_g','r':'ztf_r','i':'ztf_i','ztf_g':'ztf_g','ztf_r':'ztf_r','ztf_i':'ztf_i'}
        )
    else:
        # Sem banda identificável → vazio
        band_series = pd.Series(['']*len(df))

    out = pd.DataFrame({
        'source': 'ztf',
        'band': band_series,
        'mjd': pd.to_numeric(df[time_col], errors='coerce'),
        'mag': pd.to_numeric(df[mag_col], errors='coerce'),
        'mag_err': pd.to_numeric(df[merr_col], errors='coerce') if merr_col in df.columns else np.nan,
    })
    out['effective_wavelength_angstrom'] = out['band'].map(EFF)

    # manter apenas g/r (se quiser incluir i, é só tirar esse filtro)
    out = out[out['band'].isin(['ztf_g','ztf_r'])]

    # limpar
    out = out.replace([np.inf,-np.inf], np.nan)
    out = out.dropna(subset=['mjd','mag'])
    out = out.sort_values('mjd').reset_index(drop=True)
    return out

def read_atlas_csv(atlas_path):
    """Lê o CSV do ATLAS já no formato do pipeline."""
    req = ['source','band','mjd','mag','mag_err','effective_wavelength_angstrom']
    df = pd.read_csv(atlas_path)
    lower = [c.lower() for c in df.columns]
    if lower != list(df.columns):
        df.columns = lower
    if not all(c in df.columns for c in req):
        raise ValueError(f"ATLAS CSV não está no formato esperado. Colunas: {df.columns.tolist()}")
    return df[req].copy()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--atlas', required=True, help='CSV do ATLAS já convertido (sn2023zkd_atlas_converted.csv)')
    ap.add_argument('--out', required=True, help='Arquivo CSV de saída combinado (phot_all.csv)')
    ap.add_argument('--ra', type=float, required=True, help='RA (graus) para a SN')
    ap.add_argument('--dec', type=float, required=True, help='Dec (graus) para a SN')
    ap.add_argument('--radius', type=float, default=0.0014, help='Raio (graus), default ~5 arcsec = 0.0014')
    ap.add_argument('--no_download', action='store_true', help='Não baixar ZTF; apenas combinar ATLAS (debug).')
    args = ap.parse_args()

    # 1) ATLAS
    atlas_df = read_atlas_csv(args.atlas)

    # 2) ZTF (download + conversão)
    if args.no_download:
        ztf_df = pd.DataFrame(columns=['source','band','mjd','mag','mag_err','effective_wavelength_angstrom'])
    else:
        print("Baixando ZTF lightcurves via IRSA API...")
        txt = fetch_ztf_csv(args.ra, args.dec, args.radius)
        ztf_df = to_pipeline_from_ztf(txt)
        print(f"ZTF: {len(ztf_df)} linhas úteis (g/r).")

    # 3) Combinar
    comb = pd.concat([atlas_df, ztf_df], ignore_index=True)
    comb = comb.replace([np.inf,-np.inf], np.nan)
    comb = comb.dropna(subset=['mjd','mag'])
    comb = comb.drop_duplicates(subset=['source','band','mjd','mag'], keep='first')
    comb = comb.sort_values('mjd').reset_index(drop=True)

    # 4) Salvar
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    comb.to_csv(args.out, index=False)

    # 5) Resumo
    print("\n=== RESUMO ===")
    print("ATLAS linhas:", len(atlas_df))
    print("ZTF   linhas:", len(ztf_df))
    print("TOTAL linhas:", len(comb))
    print("Bands:\n", comb['band'].value_counts())

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERRO:", e, file=sys.stderr)
        sys.exit(1)