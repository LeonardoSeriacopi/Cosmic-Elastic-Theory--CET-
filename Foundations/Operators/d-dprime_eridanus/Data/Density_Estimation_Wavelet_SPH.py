# Adaptação de: Voids Framework + Tensor de Deformação.py
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata

def wavelet_density_estimator(coords, masses, resolution=50):
    # Criar grid 3D
    grid_x, grid_y, grid_z = np.mgrid[min_x:max_x:resolution*1j, 
                                      min_y:max_y:resolution*1j,
                                      min_z:max_z:resolution*1j]
    
    # Kernel Wavelet-SPH (adaptativo)
    r = np.linalg.norm(coords - coords[:, None], axis=-1)
    h = np.percentile(r, 10)  # Suavização adaptativa
    kernel = np.exp(-(r/h)**2) / (h**3 * np.pi**1.5)
    
    # Densidade por partícula
    return np.sum(masses * kernel, axis=1)

# Calcular densidade voxel a voxel
ρ_voxel = wavelet_density_estimator(void_coords, galaxy_masses)