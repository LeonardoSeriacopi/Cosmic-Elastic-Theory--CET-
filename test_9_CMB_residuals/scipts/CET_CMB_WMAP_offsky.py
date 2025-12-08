# ===============================================================
# CET Cosmic Elastic Theory — Combined WMAP + Planck (resampled)
# v7.1 — Corrige diferença de Nside (Planck -> WMAP)
# ===============================================================

import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix
import astropy.units as u
import matplotlib.pyplot as plt
import os

datasets = [
    ("wmap_band_imap_r9_9yr_K_v5.fits", "WMAP K (23 GHz)"),
    ("wmap_band_imap_r9_9yr_Ka_v5.fits", "WMAP Ka (33 GHz)"),
    ("wmap_band_imap_r9_9yr_Q_v5.fits", "WMAP Q (41 GHz)"),
    ("wmap_band_imap_r9_9yr_V_v5.fits", "WMAP V (61 GHz)"),
    ("wmap_band_imap_r9_9yr_W_v5.fits", "WMAP W (94 GHz)"),
    ("COM_CMB_IQU-smica_2048_R3.00_full.fits", "Planck SMICA (143 GHz)"),
]

output_dir = "resultados_3D_offsky"
os.makedirs(output_dir, exist_ok=True)

def load_healpix_data_auto(filename):
    with fits.open(filename) as hdul:
        for hdu in hdul:
            if hasattr(hdu, "data") and hdu.data is not None:
                arr = hdu.data
                try:
                    cols = arr.columns.names
                    for c in cols:
                        if "TEMP" in c.upper() or c.upper() in ["I", "T", "SIGNAL"]:
                            data = arr.field(c)
                            break
                    else:
                        data = arr.field(0)
                    break
                except Exception:
                    data = np.array(arr)
                    break
        else:
            raise ValueError(f"⚠️ Nenhum HDU com dados em {filename}.")
    return np.nan_to_num(data)

def detect_nside_from_data(data):
    return int(np.sqrt(len(data) / 12))

def normalize_data(data):
    data -= np.median(data)
    data /= (np.std(data) + 1e-8)
    vmax = np.percentile(np.abs(data), 99)
    return np.clip(data / vmax, -1, 1)

def downsample_healpix(data, factor):
    """Agrupa pixels do mapa Planck para reduzir de Nside=2048→512."""
    n = len(data)
    new_size = n // (factor**2)
    return data[:new_size * (factor**2)].reshape(new_size, factor**2).mean(axis=1)

# ---------------------------------------------------------------
# Combina todos os mapas em uma média ponderada
# ---------------------------------------------------------------
combined_map = None
weights = []

for filename, label in datasets:
    if not os.path.exists(filename):
        print(f"⚠️  Arquivo não encontrado: {filename}")
        continue

    print(f"🛰️  Lendo {label}")
    data = load_healpix_data_auto(filename)
    nside = detect_nside_from_data(data)

    # Se o mapa for Planck (Nside 2048), reduz para Nside 512
    if nside == 2048:
        print("   🔧 Reduzindo resolução Planck 2048 → 512")
        data = downsample_healpix(data, factor=4)
        nside = 512

    data = normalize_data(data)

    if combined_map is None:
        combined_map = np.zeros_like(data)
    combined_map += data
    weights.append(1.0)

combined_map /= max(1, len(weights))

print(f"\n✅ Mapas combinados ({len(weights)} bandas)")
print(f"   Intervalo final: min={combined_map.min():.3f}, max={combined_map.max():.3f}")

# ---------------------------------------------------------------
# Projeção Mollweide 2D
# ---------------------------------------------------------------
hp = HEALPix(nside=nside, order="nested", frame="galactic")
skycoord = hp.healpix_to_skycoord(np.arange(hp.npix))
lon = skycoord.l.to(u.rad).value
lat = skycoord.b.to(u.rad).value

lon = np.remainder(lon + np.pi, 2*np.pi) - np.pi

fig = plt.figure(figsize=(14, 7))
ax = fig.add_subplot(111, projection="mollweide")
im = ax.scatter(lon, lat, c=combined_map, cmap="turbo", s=1, lw=0)
ax.grid(True, color="white", alpha=0.3)
ax.set_title("CET Unified CMB Map — Combined WMAP + Planck (Resampled)", fontsize=12, color="white", pad=20)
fig.patch.set_facecolor("black")
ax.set_facecolor("black")

cb = plt.colorbar(im, orientation="horizontal", pad=0.05, aspect=50)
cb.set_label("Normalized Intensity (ΔI/I)", color="white")
cb.ax.xaxis.set_tick_params(color="white")
plt.setp(cb.ax.get_xticklabels(), color="white")

output_file = os.path.join(output_dir, "CET_CMB_Combined_v7.1_resampled.png")
plt.savefig(output_file, dpi=400, bbox_inches="tight", facecolor="black")
plt.close()
print(f"\n📸 Imagem salva: {output_file}")
print("✅ Mapa unificado CET (WMAP + Planck, resoluções compatíveis) concluído.")