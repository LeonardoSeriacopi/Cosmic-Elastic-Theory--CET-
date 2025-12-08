#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
check_coldspots_wmap_planck_nohealpy.py

Verifica o sinal de temperatura CMB nos dois "cold spots" usando
mapas HEALPix do WMAP (K, Ka, Q, V, W) e Planck (LFI 30, 44, 70 GHz),
SEM usar healpy.

Requer:
    - numpy
    - astropy
    - astropy-healpix  (pip install astropy-healpix)

Supõe que os mapas estejam em coordenadas GALÁCTICAS (padrão WMAP/Planck).
Para cada mapa e cada centro (CS1, CS2), imprime:

    - temperatura no pixel central
    - média em discos de 1°, 2° e 5° de raio
"""

import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from astropy_healpix import HEALPix

# ---------------------------------------------------------------------
# Definição dos dois centros (em coordenadas galácticas)
# ---------------------------------------------------------------------

COLDSPOTS = [
    ("CS1_main", 209.0,      -57.0),      # cold spot "principal"
    ("CS2_second", 208.4349, -55.8545),   # nosso segundo centro
]

# ---------------------------------------------------------------------
# Arquivos de entrada (ajuste os nomes se forem diferentes)
# ---------------------------------------------------------------------

WMAP_FILES = {
    "WMAP_K"  : "wmap_band_imap_r9_9yr_K_v5.fits",
    "WMAP_Ka" : "wmap_band_imap_r9_9yr_Ka_v5.fits",
    "WMAP_Q"  : "wmap_band_imap_r9_9yr_Q_v5.fits",
    "WMAP_V"  : "wmap_band_imap_r9_9yr_V_v5.fits",
    "WMAP_W"  : "wmap_band_imap_r9_9yr_W_v5.fits",
}

PLANCK_FILES = {
    "Planck_30" : "LFI_SkyMap_030_1024_R3.00_full.fits",
    "Planck_44" : "LFI_SkyMap_044_1024_R3.00_full.fits",
    "Planck_70" : "LFI_SkyMap_070_1024_R3.00_full.fits",
}

# raios (em graus) para médias em disco ao redor do centro
DISC_RADII_DEG = [1.0, 2.0, 5.0]


# ---------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------

def nside_from_npix(npix):
    """Retorna NSIDE a partir de npix, ou None se não bater com 12*nside^2."""
    nside = int(np.sqrt(npix / 12.0) + 0.5)
    if 12 * nside * nside == npix:
        return nside
    return None


def find_healpix_map_in_hdus(hdul):
    """
    Procura um HDU dentro do FITS que contenha um mapa 1D com npix = 12*nside^2.
    Retorna (map_array, header) ou (None, None) se não achar.
    """
    for hdu in hdul:
        data = hdu.data
        if data is None:
            continue

        # se for tabela (recarray), tentamos achar uma coluna com tamanho HEALPix
        if hasattr(data, "dtype") and data.dtype.fields is not None:
            # procurar coluna apropriada
            colnames = list(data.dtype.fields.keys())
            candidate = None
            for name in ("TEMPERATURE", "I_STOKES", "I", "SIGNAL", "T"):
                if name in colnames:
                    candidate = name
                    break
            if candidate is None and len(colnames) > 0:
                candidate = colnames[0]

            arr = np.array(data[candidate])
        else:
            # array normal
            arr = np.array(data)

        # se for 2D, tentar reduzir para 1D (pega componente I/temperatura)
        if arr.ndim == 2:
            if arr.shape[0] in (1, 3):
                arr = arr[0]
            elif arr.shape[1] in (1, 3):
                arr = arr[:, 0]
            else:
                # formato estranho, pula
                continue

        if arr.ndim != 1:
            continue

        npix = arr.size
        nside = nside_from_npix(npix)
        if nside is None:
            continue

        # achamos um mapa HEALPix plausível
        return arr.astype(float), hdu.header

    return None, None


def load_healpix_map(filename):
    """
    Abre um FITS HEALPix (WMAP/Planck) e retorna:
        map_vals, nside, ordering
    onde ordering ∈ {"ring", "nested"}.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Arquivo não encontrado: {filename}")

    hdul = fits.open(filename, memmap=True)
    try:
        m, hdr = find_healpix_map_in_hdus(hdul)
        if m is None:
            raise RuntimeError(f"Não foi possível encontrar um mapa HEALPix 1D em {filename}")

        nside = hdr.get("NSIDE", None)
        if nside is None:
            # tenta inferir de npix
            nside = nside_from_npix(m.size)
        if nside is None:
            raise RuntimeError(f"NSIDE não encontrado/inconsistente em {filename}")

        ordering = hdr.get("ORDERING", "RING").strip().upper()
        if ordering not in ("RING", "NESTED"):
            print(f"[AVISO] ORDERING não usual em {filename}: {ordering}. Assumindo RING.")
            ordering = "RING"

        return m, int(nside), ordering
    finally:
        hdul.close()


def disc_mean_temperature(hpx, temperature_map, lon_center_deg, lat_center_deg,
                          disc_radius_deg, lon_all=None, lat_all=None):
    """
    Calcula a média da temperatura em um disco de raio 'disc_radius_deg'
    ao redor de (lon_center_deg, lat_center_deg) em coordenadas galácticas.

    Se lon_all/lat_all forem fornecidos, são arrays de centros de pixel
    já pré-computados para o HEALPix 'hpx'; isso evita recomputar.
    """
    center = SkyCoord(l=lon_center_deg * u.deg,
                      b=lat_center_deg * u.deg,
                      frame="galactic")

    # centros de pixels
    if lon_all is None or lat_all is None:
        npix = temperature_map.size
        pix = np.arange(npix, dtype=int)
        lon_all, lat_all = hpx.healpix_to_lonlat(pix)

    coords_pix = SkyCoord(l=lon_all, b=lat_all, frame="galactic")
    sep = center.separation(coords_pix).deg
    mask = sep <= disc_radius_deg

    vals = temperature_map[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, 0

    return float(np.mean(vals)), int(vals.size)


def analyse_one_map(label, filename, coldspots):
    """
    Carrega um mapa HEALPix, monta um objeto HEALPix e imprime
    temperaturas no centro e em discos para cada cold spot.
    """
    print("=" * 72)
    print(f"[INFO] Mapa: {label}  ({filename})")

    m, nside, ordering = load_healpix_map(filename)
    npix = m.size
    print(f"[INFO] npix={npix}, NSIDE={nside}, ORDERING={ordering}")

    # Cria objeto HEALPix (assume coordenadas galácticas)
    hpx = HEALPix(nside=nside, order=ordering.lower(), frame="galactic")

    # Pré-calcula lon/lat de TODOS os pixels (para médias em disco)
    pix_indices = np.arange(npix, dtype=int)
    lon_all, lat_all = hpx.healpix_to_lonlat(pix_indices)

    # Loop sobre os centros
    for name, l_cs, b_cs in coldspots:
        # pixel central
        pix0 = hpx.lonlat_to_healpix(l_cs * u.deg, b_cs * u.deg)
        T0 = float(m[pix0])

        print(f"\n[COLD SPOT: {name}]  (l={l_cs:.4f} deg, b={b_cs:.4f} deg)")
        print(f"  Pixel central: index={pix0},  T = {T0:.6g}  (unidades do mapa)")

        # médias em discos
        for Rdeg in DISC_RADII_DEG:
            Tmean, Npix_disc = disc_mean_temperature(
                hpx, m, l_cs, b_cs, Rdeg,
                lon_all=lon_all, lat_all=lat_all
            )
            print(f"  Média em disco R={Rdeg:.1f}° :  T = {Tmean:.6g}   (Npix={Npix_disc})")


def main():
    print("=" * 72)
    print("[INFO] Checando sinal de temperatura nos dois cold spots")
    print("       usando mapas WMAP e Planck (HEALPix, sem healpy).")
    print("=" * 72)

    # WMAP
    for label, fname in WMAP_FILES.items():
        if os.path.exists(fname):
            analyse_one_map(label, fname, COLDSPOTS)
        else:
            print("=" * 72)
            print(f"[AVISO] Arquivo WMAP não encontrado: {fname} (pulando)")

    # Planck
    for label, fname in PLANCK_FILES.items():
        if os.path.exists(fname):
            analyse_one_map(label, fname, COLDSPOTS)
        else:
            print("=" * 72)
            print(f"[AVISO] Arquivo Planck não encontrado: {fname} (pulando)")

    print("=" * 72)
    print("[OK] Fim da verificação WMAP + Planck para os dois cold spots.")


if __name__ == "__main__":
    main()