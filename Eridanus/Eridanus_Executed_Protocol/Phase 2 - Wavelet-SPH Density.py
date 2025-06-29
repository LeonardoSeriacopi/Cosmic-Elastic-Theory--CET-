# wavelet_density.py (novo)
import pywt
from scipy.spatial import cKDTree

def wavelet_density(coords, scales=[5, 10, 20]):
    tree = cKDTree(coords)
    densities = np.zeros(len(coords))
    
    for scale in scales:
        # Wavelet Mexicano (derivada de Gaussiana)
        dists = tree.query(coords, k=50)[0]
        wavelet = pywt.ContinuousWavelet('mexh')
        
        for i, pt in enumerate(coords):
            dists_pt = dists[i]
            coeffs = pywt.cwt(dists_pt, [scale], wavelet)[0]
            densities[i] += np.sum(np.abs(coeffs))
    
    return densities / len(scales)

# Calcular densidade na região do vazio
void_coords = coords[void_points]
ρ_voxel = wavelet_density(void_coords, scales=[3, 7, 12])

# Salvar campo de densidade
np.save("eridanus_density.npy", ρ_voxel)
print(f"Densidade média no vazio: {np.mean(ρ_voxel):.2e} kg/m³")