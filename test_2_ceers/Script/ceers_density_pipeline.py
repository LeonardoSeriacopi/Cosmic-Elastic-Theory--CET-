import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
from astropy.cosmology import FlatLambdaCDM

# Configurações cosmológicas
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
RADIUS_MPC = 2.0

# CET redshift correction (exemplo simplificado)
def z_cet(rho, a=0.02, b=0.0005):
    return a * np.exp(-b * rho)

# Cálculo da distância comoving
def compute_comoving_distance(z):
    return cosmo.comoving_distance(z).value

# Carregamento da amostra
df = pd.read_csv("CEERS_catalog_.csv")

# Conversão para coordenadas comoving 3D
df = df.dropna(subset=["redshift", "RA", "Dec"])
df["R"] = compute_comoving_distance(df["redshift"])
df["x"] = df["R"] * np.cos(np.radians(df["Dec"])) * np.cos(np.radians(df["RA"]))
df["y"] = df["R"] * np.cos(np.radians(df["Dec"])) * np.sin(np.radians(df["RA"]))
df["z"] = df["R"] * np.sin(np.radians(df["Dec"]))

coords = df[["x", "y", "z"]].values
tree = cKDTree(coords)

# Cálculo da densidade local
vol = (4/3) * np.pi * RADIUS_MPC**3
counts = tree.query_ball_point(coords, r=RADIUS_MPC)
rho = np.array([len(c) / vol for c in counts])
df["rho"] = rho

# Correção do redshift e delta z
df["z_cet"] = z_cet(df["rho"])
df["delta_z"] = df["redshift"] - df["z_cet"]

# Correlação
corr, pval = spearmanr(df["rho"], df["delta_z"])
print(f"Spearman correlation: rho = {corr:.4f}, p = {pval:.4g}")

# Gráfico
plt.figure(figsize=(8, 6))
plt.scatter(df["rho"], df["delta_z"], s=10, alpha=0.5)
plt.xlabel("Local Density $\rho_i$ (Mpc$^{-3}$)")
plt.ylabel("Redshift Residual $\Delta z$")
plt.title("Density vs Redshift Residual (CEERS)")
plt.grid(True)
plt.tight_layout()
plt.savefig("density_vs_dz_ceers.png", dpi=300)
plt.close()
