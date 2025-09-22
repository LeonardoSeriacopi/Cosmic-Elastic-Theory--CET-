import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt

# Carregar os dados com coordenadas comóveis
df = pd.read_csv("2_processed_eridanus_comoving.csv")

# Usar massa constante por galáxia (aproximação)
df['galaxy_mass'] = 1e10  # em massas solares (M☉)

# Coordenadas espaciais
coords = df[['x', 'y', 'z']].values
masses = df['galaxy_mass'].values

# Resolução da grade 3D
resolution = 30
grid_x, grid_y, grid_z = np.mgrid[
    coords[:, 0].min():coords[:, 0].max():complex(resolution),
    coords[:, 1].min():coords[:, 1].max():complex(resolution),
    coords[:, 2].min():coords[:, 2].max():complex(resolution)
]

# Calcular distâncias entre pares
r = np.linalg.norm(coords[:, None] - coords[None, :], axis=2)
h = np.percentile(r[r > 0], 10)  # parâmetro SPH

# Kernel Gaussiano
kernel = np.exp(-(r / h) ** 2) / (h ** 3 * np.pi ** 1.5)
density = np.sum(masses * kernel, axis=1)

# Interpolação para grade 3D
density_grid = griddata(coords, density, (grid_x, grid_y, grid_z), method='linear', fill_value=0)

# Salvar resultado
np.save("density_grid_eridanus.npy", density_grid)

# Plot exemplo: fatiamento central em Z
plt.figure(figsize=(7, 5))
slice_z = resolution // 2
plt.imshow(
    density_grid[:, :, slice_z].T,
    origin='lower',
    cmap='inferno',
    extent=[grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
)
plt.colorbar(label="Local Density [a.u.]")
plt.title("Density Slice (Z = middle plane)")
plt.xlabel("X [Mpc]")
plt.ylabel("Y [Mpc]")
plt.tight_layout()
plt.savefig("density_slice_xy.png", dpi=300)
plt.close()

