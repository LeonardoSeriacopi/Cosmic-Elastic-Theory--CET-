# Exemplo de figura de validação para vazios
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plotar densidade em grid
x, y, z, rho = calc.void_density(void_coords)
sc = ax.scatter(x, y, z, c=rho, cmap='viridis', alpha=0.3)

# Plotar galáxias traçadoras
ax.scatter(void_coords[:,0], void_coords[:,1], void_coords[:,2], 
           c='red', s=10)

plt.colorbar(sc, label='Densidade Relativa')
plt.savefig('void_density_validation.pdf')