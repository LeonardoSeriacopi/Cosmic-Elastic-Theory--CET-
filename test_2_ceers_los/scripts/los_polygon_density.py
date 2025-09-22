#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LOS com POLÍGONOS (AEGIS) + z-cut + raio por alvo (opcional)

- Lê polígonos do AEGIS em CSV com colunas ra1,dec1,ra2,dec2,... (mínimo 4 pontos).
- Para cada alvo (base), encontra o polígono que o contém; conta apenas traçadores (ext)
  dentro do MESMO polígono, além do corte direcional em z (0 <= z_ext <= z_base)
  e do filtro de velocidade |dv| >= exclude_dv_kms.
- Opcional: limita por uma abertura angular por alvo (--sky-aperture-arcmin).
- Saída: adiciona los_count, los_sum_w (+ opcional los_shells_json) ao CSV base.

Uso (exemplo no fim).
"""

import argparse, sys, json, math
import numpy as np
import pandas as pd

C_KM_S = 299792.458

# --------------------------
# Utilidades geométricas
# --------------------------

def point_in_polygon(ra: float, dec: float, poly_pts):
    """
    Teste ponto-em-polígono (ray casting) no plano RA/Dec.
    poly_pts: lista [(ra1,dec1), (ra2,dec2), ...] (fechado ou não).
    """
    inside = False
    n = len(poly_pts)
    if n < 3:
        return False
    # garanta fechamento
    x1, y1 = poly_pts[0]
    for i in range(1, n + 1):
        x2, y2 = poly_pts[i % n]
        # cruza linha horizontal em 'dec'?
        if (dec > min(y1, y2)) and (dec <= max(y1, y2)) and (ra <= max(x1, x2)):
            if y1 != y2:
                xints = (dec - y1) * (x2 - x1) / (y2 - y1) + x1
            else:
                xints = ra  # segmento horizontal; tratar como passando por x=ra
            if (x1 == x2) or (ra <= xints):
                inside = not inside
        x1, y1 = x2, y2
    return inside

def load_polygons_csv(path: str):
    """
    Lê um CSV com colunas ra1,dec1,ra2,dec2,... (mínimo 4 pontos).
    Também aceita mais de 4 vértices (ra5/dec5, ra6/dec6, ...).
    Cada linha define UM polígono.
    Retorna: lista de polígonos, onde cada polígono é lista [(ra,dec),...].
    """
    df = pd.read_csv(path)
    # Detectar pares (raK, decK) presentes
    cols = df.columns.tolist()
    idxs = []
    k = 1
    while True:
        ra_col = f"ra{k}"
        dec_col = f"dec{k}"
        if (ra_col in cols) and (dec_col in cols):
            idxs.append((ra_col, dec_col))
            k += 1
        else:
            break
    if len(idxs) < 4:
        raise SystemExit("[ERR] O CSV de polígonos precisa de ao menos ra1..ra4 e dec1..dec4")

    polys = []
    for _, row in df.iterrows():
        pts = []
        for ra_col, dec_col in idxs:
            try:
                ra = float(row[ra_col]); dec = float(row[dec_col])
                pts.append((ra, dec))
            except Exception:
                pass
        # garantir fechamento explícito (opcional)
        if len(pts) >= 3 and pts[0] != pts[-1]:
            pts.append(pts[0])
        polys.append(pts)
    return polys

# --------------------------
# Utilidades astronômicas
# --------------------------

def approx_dv_kms(z_base: float, z_ext: np.ndarray) -> np.ndarray:
    return C_KM_S * np.abs(z_base - z_ext) / (1.0 + max(z_base, 1e-9))

def ang_sep_deg(ra0, dec0, ra, dec):
    # aproximação planar (ok p/ campo pequeno)
    fac = math.cos(math.radians(dec0))
    dRA  = (ra - ra0) * fac
    dDec = (dec - dec0)
    return np.sqrt(dRA*dRA + dDec*dDec)

# --------------------------
# Núcleo LOS (pesos 1/(1+Δz)^alpha)
# --------------------------

def compute_los_for_target(
    ra_b, dec_b, z_b,
    ext_ra, ext_dec, ext_z,
    polygons, sky_aperture_arcmin,
    alpha, exclude_dv_kms,
    shell_dz
):
    # Casos degenerados
    if not np.isfinite(z_b) or z_b <= 0:
        return dict(los_count=0, los_sum_w=0.0,
                    shells_json=None)

    # Descobrir qual polígono contém o alvo; se nenhum, retorna 0 (conservador)
    target_poly = None
    if polygons is not None and len(polygons) > 0:
        for poly in polygons:
            if point_in_polygon(ra_b, dec_b, poly):
                target_poly = poly
                break
        if target_poly is None:
            return dict(los_count=0, los_sum_w=0.0, shells_json=None)

    # Corte direcional em z (0 <= z_ext <= z_b)
    mask = (ext_z >= 0.0) & (ext_z <= z_b)

    # Aplica polígono aos traçadores: só quem está no MESMO polígono do alvo
    if target_poly is not None:
        # (loop simples; num catálogo ~10^4–10^5 ainda é ok)
        mask_poly = np.zeros_like(mask, dtype=bool)
        for i in range(mask.size):
            if mask[i]:
                if point_in_polygon(float(ext_ra[i]), float(ext_dec[i]), target_poly):
                    mask_poly[i] = True
        mask &= mask_poly

    # Raio angular por alvo (opcional)
    if sky_aperture_arcmin and sky_aperture_arcmin > 0:
        R_deg = sky_aperture_arcmin / 60.0
        sep = ang_sep_deg(ra_b, dec_b, ext_ra, ext_dec)
        mask &= (sep <= R_deg)

    # Excluir companheiras por dv
    if exclude_dv_kms and exclude_dv_kms > 0:
        dv = approx_dv_kms(z_b, ext_z)
        mask &= (dv >= exclude_dv_kms)

    z_use = ext_z[mask]
    if z_use.size == 0:
        return dict(los_count=0, los_sum_w=0.0, shells_json=None)

    # Pesos: 1/(1+Δz)^alpha
    dzf = (z_b - z_use)  # >= 0
    w = 1.0 / np.power(1.0 + dzf, max(alpha, 0.0))

    los_count = int(z_use.size)
    los_sum_w = float(np.sum(w))

    # Shells truncadas em z_b (diagnóstico)
    shells_json = None
    if shell_dz and shell_dz > 0:
        edges = np.arange(0.0, z_b + 1e-9, shell_dz)
        if edges.size >= 2:
            hist, _ = np.histogram(z_use, bins=edges)
            shells_json = json.dumps(dict(
                z_edges=edges.tolist(),
                counts=[int(x) for x in hist]
            ))

    return dict(los_count=los_count, los_sum_w=los_sum_w, shells_json=shells_json)

# --------------------------
# CLI
# --------------------------

def main():
    ap = argparse.ArgumentParser(description="LOS com polígonos (AEGIS) + z-cut + raio por alvo")
    ap.add_argument("--base", required=True, help="CSV base (alvos)")
    ap.add_argument("--ext",  required=True, help="CSV traçadores")
    ap.add_argument("--regions", required=True, help="CSV de polígonos (ra1,dec1,ra2,dec2,...)")
    ap.add_argument("--base-ra", required=True)
    ap.add_argument("--base-dec", required=True)
    ap.add_argument("--base-z", required=True)
    ap.add_argument("--ext-ra", required=True)
    ap.add_argument("--ext-dec", required=True)
    ap.add_argument("--ext-z", required=True)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--exclude-dv-kms", type=float, default=1500.0)
    ap.add_argument("--sky-aperture-arcmin", type=float, default=0.0)
    ap.add_argument("--shell-dz", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    ext  = pd.read_csv(args.ext)

    # validação de colunas
    for col in [args.base_ra, args.base_dec, args.base_z]:
        if col not in base.columns:
            raise SystemExit(f"[ERR] faltou coluna base: {col}")
    for col in [args.ext_ra, args.ext_dec, args.ext_z]:
        if col not in ext.columns:
            raise SystemExit(f"[ERR] faltou coluna ext: {col}")

    # carregar polígonos
    polygons = load_polygons_csv(args.regions)

    # arrays ext
    ext_ra  = ext[args.ext_ra].to_numpy(float)
    ext_dec = ext[args.ext_dec].to_numpy(float)
    ext_z   = ext[args.ext_z].to_numpy(float)

    # outputs
    los_count_list = []
    los_sumw_list  = []
    shells_list    = []

    # loop por alvo
    for _, row in base.iterrows():
        ra_b  = float(row[args.base_ra])
        dec_b = float(row[args.base_dec])
        z_b   = float(row[args.base_z])

        res = compute_los_for_target(
            ra_b, dec_b, z_b,
            ext_ra, ext_dec, ext_z,
            polygons, args.sky_aperture_arcmin,
            args.alpha, args.exclude_dv_kms,
            args.shell_dz
        )
        los_count_list.append(res["los_count"])
        los_sumw_list.append(res["los_sum_w"])
        shells_list.append(res["shells_json"])

    out = base.copy()
    out["los_count"] = los_count_list
    out["los_sum_w"] = los_sumw_list
    if args.shell_dz and args.shell_dz > 0:
        out["los_shells_json"] = shells_list

    out.to_csv(args.out, index=False)
    print(f"[OK] Wrote: {args.out}")
    print("Columns added: los_count, los_sum_w" + (", los_shells_json" if args.shell_dz and args.shell_dz > 0 else ""))

if __name__ == "__main__":
    main()