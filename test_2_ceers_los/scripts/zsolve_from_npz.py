#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, json
import numpy as np
import pandas as pd

# --------------------- linhas de referência (Å) ---------------------
REST_LINES = [
    # fortes óptico/NIR comuns na faixa do PRISM
    ("Lyα", 1215.67),
    ("CIV1549", 1549.0),
    ("HeII1640", 1640.4),
    ("OIII]1663", 1663.0),
    ("CIII]1907", 1906.68),
    ("CIII]1909", 1908.73),
    ("MgII2796", 2796.35),
    ("MgII2803", 2803.53),
    ("[OII]3726", 3726.03),
    ("[OII]3729", 3728.82),
    ("Hδ", 4101.74),
    ("Hγ", 4340.47),
    ("Hβ", 4861.33),
    ("[OIII]4959", 4958.91),
    ("[OIII]5007", 5006.84),
    ("HeI5876", 5875.62),
    ("[OI]6300", 6300.30),
    ("Hα", 6562.80),
    ("[NII]6548", 6548.05),
    ("[NII]6583", 6583.45),
    ("[SII]6716", 6716.44),
    ("[SII]6731", 6730.82),
    # Paβ/Paα se cobrir:
    ("Paβ", 12818.1),
    ("Paα", 18756.1),
]

# --------------------- utilidades ---------------------
def robust_mad(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    return 1.4826 * np.nanmedian(np.abs(x - med))

def rolling_median(x, win):
    x = np.asarray(x, float)
    k = int(max(3, win))
    if k % 2 == 0: k += 1
    if x.size < k: return np.full_like(x, np.nanmedian(x))
    pad = k//2
    xp = np.pad(x, (pad,pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.size):
        out[i] = np.nanmedian(xp[i:i+k])
    return out

def smooth_mean(x, win):
    if win<=1: return np.asarray(x, float)
    k = int(win); 
    if k%2==0: k+=1
    pad = k//2
    xp = np.pad(x, (pad,pad), mode="edge")
    return np.convolve(xp, np.ones(k)/k, mode="valid")

def detect_peaks_signed(y, err=None, snr_min=2.0, rel_prom=0.05, min_sep=3):
    y = np.asarray(y, float)
    n = y.size
    if n<5: 
        return np.array([], int), np.array([], float), np.array([], float)
    if err is None or np.all(~np.isfinite(err)):
        noise = robust_mad(y)
        if not np.isfinite(noise) or noise<=0: noise = np.nanstd(y)
        if noise<=0: noise = 1.0
    else:
        noise = np.nanmedian(err)
        if not np.isfinite(noise) or noise<=0:
            noise = robust_mad(y)
            if noise<=0: noise = 1.0

    max_c = np.where((y[1:-1]>y[:-2]) & (y[1:-1]>y[2:]))[0]+1
    min_c = np.where((y[1:-1]<y[:-2]) & (y[1:-1]<y[2:]))[0]+1
    cand  = np.unique(np.concatenate([max_c, min_c]))
    if cand.size==0: 
        return cand, np.array([], float), np.array([], float)

    y_med = np.nanmedian(y)
    span  = np.nanmax(np.abs(y-y_med)) + 1e-12
    prom  = np.abs(y[cand] - y_med)
    snr   = np.abs(y[cand]) / max(noise,1e-12)

    keep = (snr >= snr_min) & (prom >= rel_prom*span)
    cand = cand[keep]; snr = snr[keep]; prom = prom[keep]
    if cand.size==0: return cand, snr, prom

    if min_sep>1 and cand.size>1:
        keep_idx = [0]; last = cand[0]
        for i in range(1, cand.size):
            if cand[i]-last >= min_sep:
                keep_idx.append(i); last = cand[i]
        cand = cand[keep_idx]; snr = snr[keep_idx]; prom = prom[keep_idx]
    return cand, snr, prom

def dv_from_match(lam_obs, lam_rest, z):
    c = 299792.458
    lam_model = lam_rest * (1.0 + z)
    return c * (lam_obs - lam_model) / lam_model

def refine_scale(lam_obs, lam_rest):
    """Min quad: lam_obs = s * lam_rest  -> s = sum(xy)/sum(x^2)."""
    x = np.asarray(lam_rest, float)
    y = np.asarray(lam_obs, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if x.size < 2: 
        return None
    s = (x*y).sum() / (x*x).sum()
    return s

# --------------------- solver ---------------------
def solve_z_for_object(lamA, flx, err, args, rest_lines):
    # ordenar por lambda
    order = np.argsort(lamA)
    lamA  = lamA[order]
    flx   = flx[order]
    err   = err[order] if err is not None else np.full_like(flx, np.nan)

    # pré-processamento
    cont  = rolling_median(flx, args.cont_win)
    flx_d = flx - cont
    flx_s = smooth_mean(flx_d, args.smooth) if args.smooth>1 else flx_d

    # picos
    pk, snr, prom = detect_peaks_signed(flx_s, err, args.snr_min, args.rel_prom, args.min_sep)
    if pk.size == 0:
        return None, [], {"n_peaks": 0, "reason": "no_peaks"}

    lam_pk = lamA[pk]
    snr_pk = snr

    # limita aos N picos mais altos em SNR (evita combinatória louca)
    if lam_pk.size > args.max_peaks:
        top = np.argsort(snr_pk)[-args.max_peaks:]
        lam_pk = lam_pk[top]
        snr_pk = snr_pk[top]

    # candidatos de z por pareamento simples
    rest = np.array([r[1] for r in rest_lines], float)
    names = [r[0] for r in rest_lines]

    cand_z = []
    for lo in lam_pk:
        z_vec = lo/rest - 1.0
        # restrição de faixa de z
        m = (z_vec >= args.zmin) & (z_vec <= args.zmax)
        cand_z.extend(list(z_vec[m]))
    if len(cand_z)==0:
        return None, [], {"n_peaks": int(pk.size), "reason": "no_z_in_range"}

    cand_z = np.array(cand_z, float)

    # clusterização simples por histograma (janelas de dz)
    dz = args.cluster_dz
    bins = np.arange(args.zmin, args.zmax+dz, dz)
    hist, edges = np.histogram(cand_z, bins=bins)
    # escolhe até K bins mais povoados para avaliar
    k = min(args.max_z_trials, int(np.sum(hist>0)))
    if k==0:
        return None, [], {"n_peaks": int(pk.size), "reason": "no_clusters"}

    best = None
    best_score = (-1, -np.inf, np.inf)  # (n_match, sum_snr, rms_dv) -> max n_match, max snr, min rms
    best_used = None

    centers = 0.5*(edges[:-1]+edges[1:])
    trial_z = centers[np.argsort(hist)[-k:]]

    for z0 in trial_z:
        # para cada linha de rest, calcule posição esperada e procure pico mais próximo
        used = []
        dv_list = []
        snr_sum = 0.0
        for name, lam_rest in rest_lines:
            lam_exp = lam_rest*(1.0+z0)
            # ignore linhas fora da janela espectral:
            if lam_exp < lamA[0] or lam_exp > lamA[-1]:
                continue
            # procure pico mais próximo
            j = np.argmin(np.abs(lam_pk - lam_exp))
            lam_hit = lam_pk[j]
            dv = dv_from_match(lam_hit, lam_rest, z0)
            if np.abs(dv) <= args.dv_tol:
                used.append((name, lam_rest, lam_hit, float(dv), float(snr_pk[j])))
                dv_list.append(dv)
                snr_sum += float(snr_pk[j])

        n_match = len(used)
        if n_match >= max(1, args.min_lines):
            rms = float(np.sqrt(np.mean(np.square(dv_list)))) if n_match>0 else np.inf
            score = (n_match, snr_sum, -rms)  # último invertido para max()
            if score > best_score:
                best_score = score
                best = z0
                best_used = used

    if best is None:
        return None, [], {"n_peaks": int(pk.size), "reason": "no_trial_passed"}

    # refino linear com as linhas usadas
    if len(best_used) >= 2:
        lam_obs = np.array([u[2] for u in best_used], float)
        lam_rst = np.array([u[1] for u in best_used], float)
        s = refine_scale(lam_obs, lam_rst)
        if s is not None and np.isfinite(s) and s>0:
            z_ref = s - 1.0
            # re-avalia dv com z_ref
            used2 = []
            dv2 = []
            snr_sum2 = 0.0
            for name, lam_rest, lam_hit, _, snrhit in best_used:
                dv = dv_from_match(lam_hit, lam_rest, z_ref)
                used2.append((name, lam_rest, lam_hit, float(dv), float(snrhit)))
                dv2.append(dv); snr_sum2 += snrhit
            rms2 = float(np.sqrt(np.mean(np.square(dv2)))) if len(dv2)>0 else np.inf
            # aceita refino se melhora RMS
            rms1 = float(np.sqrt(np.mean(np.square([u[3] for u in best_used]))))
            if rms2 <= rms1:
                best = float(z_ref)
                best_used = used2

    # métricas finais
    rms = float(np.sqrt(np.mean(np.square([u[3] for u in best_used])))) if len(best_used)>0 else np.inf
    return float(best), best_used, {
        "n_peaks": int(pk.size),
        "n_match": int(len(best_used)),
        "rms_dv": rms,
        "snr_sum": float(np.sum([u[4] for u in best_used])) if best_used else 0.0
    }

def main():
    ap = argparse.ArgumentParser(description="Solver de z_spec a partir de zprep_spectra.npz (matching multi-linha).")
    ap.add_argument("--manifest", default="zprep_manifest.csv")
    ap.add_argument("--npz",      default="zprep_spectra.npz")
    ap.add_argument("--catalog",  default=None, help="(opcional) CSV para juntar z de referência")
    ap.add_argument("--id-col",   default="msa_id")
    ap.add_argument("--zref-col", default="z_best")

    # pico & pré-processamento
    ap.add_argument("--cont-win", type=int, default=101)
    ap.add_argument("--smooth",   type=int, default=7)
    ap.add_argument("--snr-min",  type=float, default=2.0)
    ap.add_argument("--rel-prom", type=float, default=0.05)
    ap.add_argument("--min-sep",  type=int, default=3)
    ap.add_argument("--max-peaks", type=int, default=20)

    # busca em z
    ap.add_argument("--zmin", type=float, default=0.0)
    ap.add_argument("--zmax", type=float, default=8.0)
    ap.add_argument("--cluster-dz", type=float, default=0.05)
    ap.add_argument("--dv-tol", type=float, default=600.0, help="tolerância de velocidade [km/s]")
    ap.add_argument("--min-lines", type=int, default=2)
    ap.add_argument("--max-z-trials", type=int, default=25)

    ap.add_argument("--out-solve", default="zsolve_results.csv")
    ap.add_argument("--out-matches", default="zsolve_line_matches.csv")
    args = ap.parse_args()

    mani = pd.read_csv(args.manifest)
    blobs = np.load(args.npz, allow_pickle=False)

    zref_map = {}
    if args.catalog:
        cat = pd.read_csv(args.catalog)
        if args.id_col in cat.columns and args.zref_col in cat.columns:
            zref_map = dict(zip(cat[args.id_col], cat[args.zref_col]))

    rows_res = []
    rows_lin = []

    for obj in mani[args.id_col].tolist():
        lam_key = f"lam_A__{obj}"
        flx_key = f"flux__{obj}"
        err_key = f"err__{obj}"
        if lam_key not in blobs or flx_key not in blobs:
            rows_res.append({args.id_col: obj, "z_spec": np.nan, "n_match": 0, "rms_dv": np.nan,
                             "snr_sum": np.nan, "quality": "no_data", "z_ref": zref_map.get(obj, np.nan),
                             "dz_vs_ref": np.nan})
            continue

        lamA = blobs[lam_key].astype(float)
        flx  = blobs[flx_key].astype(float)
        err  = blobs[err_key].astype(float) if err_key in blobs else np.full_like(flx, np.nan)

        z, used, meta = solve_z_for_object(lamA, flx, err, args, REST_LINES)

        quality = "fail"
        if z is not None:
            if   meta["n_match"] >= 3 and meta["rms_dv"] <= 300: quality = "A"
            elif meta["n_match"] >= 2 and meta["rms_dv"] <= 500: quality = "B"
            else: quality = "C"

        zref = zref_map.get(obj, np.nan)
        dzr  = (z - zref) if (z is not None and np.isfinite(zref)) else np.nan

        rows_res.append({
            args.id_col: obj,
            "z_spec": float(z) if z is not None else np.nan,
            "n_match": int(meta.get("n_match", 0)),
            "rms_dv": float(meta.get("rms_dv", np.nan)),
            "snr_sum": float(meta.get("snr_sum", np.nan)),
            "quality": quality,
            "z_ref": zref,
            "dz_vs_ref": dzr
        })

        for name, lam_rest, lam_hit, dv, snr in used:
            rows_lin.append({
                args.id_col: obj,
                "line": name,
                "lambda_rest_A": float(lam_rest),
                "lambda_obs_A":  float(lam_hit),
                "dv_kms": float(dv),
                "peak_snr": float(snr),
                "z_adopted": float(z) if z is not None else np.nan
            })

    pd.DataFrame(rows_res).to_csv(args.out_solve, index=False)
    pd.DataFrame(rows_lin).to_csv(args.out_matches, index=False)

    n_good = sum(pd.Series([r["quality"] for r in rows_res]).isin(["A","B"]))
    print(f"[OK] z_spec resolvidos: {n_good} (A/B) de {len(rows_res)}")
    print(f"[OK] Resultados  -> {args.out_solve}")
    print(f"[OK] Matches    -> {args.out_matches}")
    if args.catalog:
        print("[OBS] Coluna dz_vs_ref preenchida quando z_ref disponível.")

if __name__ == "__main__":
    main()