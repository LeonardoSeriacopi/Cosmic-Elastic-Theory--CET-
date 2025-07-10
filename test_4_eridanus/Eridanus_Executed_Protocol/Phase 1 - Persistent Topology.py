# Identifies void boundaries using persistent homology
import gudhi

coords = np.loadtxt("Void_186_31_Leonardo_Seriacopi_0.csv")[:,:3]  # [x,y,z] in Mpc
rips = gudhi.RipsComplex(points=coords, max_edge_length=50)
persistence = rips.create_simplex_tree(max_dimension=3).persistence()

# Save topological boundary (H2 cycles with persistence >15 Mpc)
void_boundary = [simplex[0] for simplex in persistence if simplex[0]==2 and (simplex[1][1]-simplex[1][0])>15]
np.savetxt("void_boundary.txt", void_boundary)