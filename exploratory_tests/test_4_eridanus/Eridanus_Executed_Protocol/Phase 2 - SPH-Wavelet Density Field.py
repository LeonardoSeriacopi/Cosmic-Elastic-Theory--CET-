# 3D density reconstruction
from pywt import cwt

def wavelet_density(points, scales=[3,7,12]):
    tree = cKDTree(points)
    return np.mean([np.sum(np.abs(cwt(tree.query(p,50)[0],s)) for s in scales], axis=0)

ρ = wavelet_density(coords[void_boundary])  # kg/m³