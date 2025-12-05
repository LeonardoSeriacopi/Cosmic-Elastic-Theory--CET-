# ===============================================================
# CET Cosmic Elastic Theory – 3D Visualization of CMB (WMAP)
# v5.7_balance  →  múltiplas bandas + contraste dinâmico leve
# ===============================================================

import numpy as np
from astropy.io import fits
from astropy_healpix import HEALPix
import astropy.units as u
import pyvista as pv
import os

# ---------------------------------------------------------------
# CONFIGURAÇÕES
# ---------------------------------------------------------------
bands = [
    ("K", "wmap_band_imap_r9_9yr_K_v5.fits", 23),
    ("Ka", "wmap_band_imap_r9_9yr_Ka_v5.fits", 33),
    ("Q", "wmap_band_imap_r9_9yr_Q_v5.fits", 41),
    ("V", "wmap_band_imap_r9_9yr_V_v5.fits", 61),
    ("W", "wmap_band_imap_r9_9yr_W_v5.fits", 94),
]

output_dir = "resultados_3D_balance"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------------
# FUNÇÕES
# ---------------------------------------------------------------
def detect_nside_from_data(data):
    """Detecta automaticamente o nside real do mapa."""
    npix = len(data)
    nside = int(np.sqrt(npix / 12))
    return nside

def load_healpix_data(filename):
    """Lê mapa HEALPix e aplica normalização CET visual."""
    with fits.open(filename) as hdul:
        data = hdul[1].data.field(0)
    data = np.nan_to_num(data)
    data = data / 1e3  # mK → K
    data /= np.mean(data[data != 0])

    # ---- Reforço dinâmico de contraste ----
    data = data - np.median(data)
    data = np.sign(data) * np.sqrt(np.abs(data))
    data /= np.percentile(np.abs(data), 99.5)  # corte de outliers 0.5%
    data = np.clip(data, -1, 1)
    return data

def create_healpix_points(data, nside):
    """Cria coordenadas HEALPix simples (sem triangulação)."""
    hp = HEALPix(nside=nside, order="nested", frame="galactic")
    npix = hp.npix
    skycoord = hp.healpix_to_skycoord(np.arange(npix))
    lon = skycoord.l.to(u.rad).value
    lat = skycoord.b.to(u.rad).value

    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.column_stack([x, y, z]), data

# ---------------------------------------------------------------
# RENDERIZAÇÃO MULTI-BANDA
# ---------------------------------------------------------------
plotter = pv.Plotter(shape=(1, 5), off_screen=True, window_size=[5120, 1080])
plotter.background_color = "black"

for i, (band, filename, freq) in enumerate(bands):
    print(f"🛰️  Banda {band} ({freq} GHz) → carregando {filename}")
    if not os.path.exists(filename):
        print(f"⚠️  Arquivo {filename} não encontrado. Pulando...")
        continue

    data = load_healpix_data(filename)
    nside = detect_nside_from_data(data)
    points, values = create_healpix_points(data, nside)

    cloud = pv.PolyData(points)
    cloud["intensity"] = values

    plotter.subplot(0, i)
    plotter.add_mesh(
        cloud,
        scalars="intensity",
        cmap="turbo",
        point_size=2.0,
        render_points_as_spheres=True,
        clim=[-1, 1],
        show_scalar_bar=False,
    )
    plotter.add_text(f"{band}-band\n({freq} GHz)", font_size=12, color="white")
    plotter.camera_position = "yz"

# ---------------------------------------------------------------
# EXPORTAR FIGURA
# ---------------------------------------------------------------
output_file = os.path.join(output_dir, "CET_CMB_WMAP_v5.7_balance_5bands.png")
plotter.screenshot(output_file, window_size=[5120, 1080], scale=1)
print(f"\n📸 Imagem final salva em: {output_file}")
print("✅ Renderização múltipla concluída com sucesso!\n")