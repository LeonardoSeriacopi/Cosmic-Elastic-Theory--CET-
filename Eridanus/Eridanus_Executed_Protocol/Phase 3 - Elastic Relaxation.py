# elastic_relaxation.py (novo)
from scipy import integrate

# Parâmetros CET
P_crit = 1.6e-9 * (3e8)**4 / 6.6743e-11  # 1.6e-9 c⁴/G
K = 2.1e92  # Módulo volumétrico

def ε_field(ρ):
    P = ρ * (3e8)**2  # Pressão relativística
    return np.where(P > P_crit, (P - P_crit) / K, 0)

# Carregar densidade
ρ_voxel = np.load("eridanus_density.npy")
ε_voxel = ε_field(ρ_voxel)

# Integração numérica (método Monte Carlo)
def integrand(x, y, z):
    idx = np.argmin(np.linalg.norm(void_coords - [x,y,z], axis=1))
    return ε_voxel[idx] / P_crit

bounds = [[void_coords[:,i].min(), void_coords[:,i].max()] for i in range(3)]
R_void, _ = integrate.nquad(integrand, bounds, opts={'limit': 100})
print(f"Relaxamento integral R = {R_void:.4f}")