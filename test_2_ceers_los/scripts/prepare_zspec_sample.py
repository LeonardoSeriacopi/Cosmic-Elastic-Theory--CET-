#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, re, json
import numpy as np
import pandas as pd

C_KMS = 299_792.458

# --------- helpers ---------
def detect_units_from_values(lam):
    lam = np.asarray(lam, float)
    if lam.size == 0 or not np.isfinite(np.nanmedian(lam)):
        return "unknown"
    med = float(np.nanmedian(lam))
    # NIRSpec PRISM típico no CEERS ~ 1–5 μm => 1e4–5e4 Å
    if med < 100.0:       # e.g., 1–5 μm
        return "micron"
    elif med > 1000.0:    # e.g., ~10000–50000 Å
        return "angstrom"
    else:
        # zona cinza (nm). Trate como micron se ~1–5e3 (nm)
        if 800.0 <= med <= 6000.0:
            return "nanometer"
        return "angstrom"

def to_angstrom(lam, unit):
    lam = np.asarray(lam, float)
    if unit == "angstrom":
        return lam
    elif unit == "micron":
        return lam * 1e4
    elif unit == "nanometer":
        return lam * 10.0
    elif unit == "auto" or unit == "unknown":
        u = detect_units_from_values(lam)
        return to_angstrom(lam, u)
    else:
        return lam

def robust_mad(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

def estimate_snr(flux, err):
    f = np.asarray(flux, float)
    e = np.asarray(err, float) if err is not None else None
    if e is not None and np.isfinite(np.nanmedian(e)) and np.nanmedian(e) > 0:
        snr = f / e
        return float(np.nanmedian(np.abs(snr)))
    # fallback: usa MAD como “ruído”
    noise = robust_mad(f)
    if not np.isfinite(noise) or noise <= 0:
        noise = np.nanstd(f)
    if not np.isfinite(noise) or noise <= 0:
        return np.nan
    return float(np.nanmedian(np.abs(f)) / noise)

def read_x1d_csv(path, wave_col="wavelength", flux_col="flux", err_col="flux_err"):
    df = pd.read_csv(path)
    if wave_col not in df or flux_col not in df:
        raise ValueError(f"Colunas {wave_col}/{flux_col} não encontradas em {path}")
    lam = df[wave_col].to_numpy(dtype=float)
    flx = df[flux_col].to_numpy(dtype=float)
    err = df[err_col].to_numpy(dtype=float) if err_col in df else np.full_like(flx, np.nan)
    return lam, flx, err

# --------- main ---------
def main():
    ap = argparse.ArgumentParser(description="Prepara amostra para medição de z_spec (padroniza espectros e gera manifesto).")
    ap.add_argument("--catalog", required=True, help="CSV com coluna para caminho do x1d (ex.: ceers_xmatch_3dhst_matched_with_spec.csv)")
    ap.add_argument("--x1d-col", default="spec_csv_path", help="Nome da coluna com o caminho para o CSV do espectro x1d")
    ap.add_argument("--id-col", default="msa_id", help="Identificador único do objeto")
    ap.add_argument("--wave-col", default="wavelength")
    ap.add_argument("--flux-col", default="flux")
    ap.add_argument("--err-col", default="flux_err")
    ap.add_argument("--units", choices=["auto","angstrom","micron","nanometer"], default="auto")
    ap.add_argument("--min-len", type=int, default=20, help="Comprimento mínimo esperado do espectro (apenas flag, não exclui)")
    ap.add_argument("--out-manifest", default="zprep_manifest.csv")
    ap.add_argument("--out-npz", default="zprep_spectra.npz")
    ap.add_argument("--keep-cols", default="msa_id,ra_final,dec_final,egs_id,egs_ra,egs_dec,s2d_path,src_file",
                    help="Cols adicionais do catálogo base para copiar ao manifesto (se existirem). Separe por vírgula.")
    args = ap.parse_args()

    base = pd.read_csv(args.catalog)
    if args.id_col not in base.columns:
        raise SystemExit(f"[ERRO] id-col '{args.id_col}' não está em {args.catalog}. Colunas: {list(base.columns)}")
    if args.x1d_col not in base.columns:
        raise SystemExit(f"[ERRO] x1d-col '{args.x1d_col}' não está em {args.catalog}. Colunas: {list(base.columns)}")

    keep_cols = [c.strip() for c in args.keep_cols.split(",") if c.strip()]
    keep_cols = [c for c in keep_cols if c in base.columns]

    rows = []
    blobs = {}
    n_ok = n_err = 0

    for idx, row in base.iterrows():
        obj = row[args.id_col]
        spec_path = row[args.x1d_col]

        rec = {args.id_col: obj, "spec_path": spec_path}
        for c in keep_cols:
            rec[c] = row[c]

        try:
            if not isinstance(spec_path, str) or not os.path.exists(spec_path):
                raise FileNotFoundError(f"Espectro não encontrado: {spec_path}")

            lam, flx, err = read_x1d_csv(spec_path, args.wave_col, args.flux_col, args.err_col)
            unit = args.units if args.units != "auto" else "auto"
            lamA = to_angstrom(lam, unit)

            # Ordena por lambda
            order = np.argsort(lamA)
            lamA = lamA[order]
            flx  = flx[order]
            err  = err[order] if err is not None else np.full_like(flx, np.nan)

            n = int(lamA.size)
            lam_min = float(np.nanmin(lamA)) if n else np.nan
            lam_max = float(np.nanmax(lamA)) if n else np.nan
            snr_med = estimate_snr(flx, err)
            frac_finite = float(np.isfinite(flx).sum() / max(1, n))

            # Flags leves (sem excluir)
            flags = []
            if n < args.min_len: flags.append("short")
            if frac_finite < 0.7: flags.append("many_nans")
            if not np.isfinite(snr_med): flags.append("snr_nan")
            if np.isfinite(snr_med) and snr_med < 0.5: flags.append("snr_low")

            rec.update({
                "n_pts": n,
                "lam_min_A": lam_min,
                "lam_max_A": lam_max,
                "snr_med": snr_med,
                "frac_flux_finite": frac_finite,
                "flags": "|".join(flags) if flags else "",
            })

            # Guarda arrays no NPZ (chaves determinísticas)
            key_l = f"lam_A__{obj}"
            key_f = f"flux__{obj}"
            key_e = f"err__{obj}"
            blobs[key_l] = lamA.astype(np.float32)
            blobs[key_f] = flx.astype(np.float32)
            blobs[key_e] = err.astype(np.float32)

            rows.append(rec)
            n_ok += 1

        except Exception as e:
            rec.update({
                "n_pts": 0, "lam_min_A": np.nan, "lam_max_A": np.nan,
                "snr_med": np.nan, "frac_flux_finite": 0.0, "flags": f"error:{e}"
            })
            rows.append(rec)
            n_err += 1

    # Saídas
    man = pd.DataFrame(rows)
    man.to_csv(args.out_manifest, index=False)
    np.savez_compressed(args.out_npz, **blobs)

    print(f"[OK] Objetos processados: {n_ok}  |  com erro: {n_err}")
    print(f"[OK] Manifesto -> {args.out_manifest}")
    print(f"[OK] Pacote NPZ -> {args.out_npz}")
    print("[OBS] Nada foi filtrado; flags servem só para inspeção/depuração.")
    print("[DICA] Próximo passo: usar o NPZ/manifesto para rodar o solver de z_spec (matching multi-linha + ajuste fino).")

if __name__ == "__main__":
    main()