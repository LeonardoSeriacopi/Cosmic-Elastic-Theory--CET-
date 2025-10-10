import pandas as pd
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt

# Carregar catálogo
df = pd.read_csv("dr16q_merged_voidflag.csv")

# Supondo que existe a coluna 'completeness_weight', se não, crie uma coluna com 1 (sem peso)
df["weight"] = 1.0  # Substitua pelo cálculo correto se disponível

z_in = df[df["inside_void"] == True]["Z_QSO"].values
w_in = df[df["inside_void"] == True]["weight"].values

z_out = df[df["inside_void"] == False]["Z_QSO"].values
w_out = df[df["inside_void"] == False]["weight"].values

# KDE ponderado
kde_in = gaussian_kde(z_in, weights=w_in)
kde_out = gaussian_kde(z_out, weights=w_out)

x = np.linspace(0.5, 2.6, 300)
plt.plot(x, kde_in(x), label="Inside voids (weighted)")
plt.plot(x, kde_out(x), label="Outside voids (weighted)")
plt.xlabel("Redshift")
plt.ylabel("Weighted Density")
plt.legend()
plt.title("Weighted KDE of Quasar Redshift (SDSS Completeness)")
plt.show()

