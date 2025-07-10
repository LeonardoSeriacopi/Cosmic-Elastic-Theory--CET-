
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("CEERS_catalog_processed.csv")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['ra'], df['dec'], df['z_cet'], c=df['density_rhocrit'], cmap='viridis', s=10)
ax.set_xlabel('RA')
ax.set_ylabel('Dec')
ax.set_zlabel('z_CET')
ax.set_title('3D Distribution of Galaxies (CEERS)')
plt.tight_layout()
plt.savefig("figura2_ceers_3d_distribution.png", dpi=300)
