# tda_void_eridanus_script_debug.py (modificado)
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt

# Dados de exemplo (estrutura do vazio)
coords = np.loadtxt("Void_186_31_Leonardo_Seriacopi_0.csv", delimiter=',', skiprows=1)[:,:3]

# Complexo de Vietoris-Rips
rips = gd.RipsComplex(points=coords, max_edge_length=50)
simplex_tree = rips.create_simplex_tree(max_dimension=3)

# Homologia persistente
persistence = simplex_tree.persistence()
gd.plot_persistence_diagram(persistence)
plt.savefig("eridanus_persistence.png")

# Identificação automática do vazio (H2 persistente)
void_points = []
for interval in persistence:
    if interval[0] == 2 and interval[1][1] - interval[1][0] > 15:  # Filtro de persistência
        void_points.extend(simplex_tree.get_simplex_vertices(interval[0]))
        
void_points = np.unique(void_points)
np.savetxt("void_boundary_indices.txt", void_points, fmt='%d')
print(f"Fronteira do vazio identificada: {len(void_points)} pontos")