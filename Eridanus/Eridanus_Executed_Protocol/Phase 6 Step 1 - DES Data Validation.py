from astropy.io import fits
import healpy as hp

# Carregar mapa de convergência DES Y6
kappa_map = hp.read_map("DESY6kappa.fits")
mask = hp.read_map("DESY6mask_eridanus.fits")

# Coordenadas do Eridanus Supervoid (65.0°, -10.0°, z=0.006)
center_pix = hp.ang2pix(2048, 65.0, -10.0, lonlat=True)
radius = np.radians(5.0)  # Raio de 5 graus

# Extrair região do vazio
theta, phi = hp.pix2ang(2048, center_pix)
pix_void = hp.query_disc(2048, (theta, phi), radius)
kappa_void = kappa_map[pix_void][mask[pix_void] > 0.8]