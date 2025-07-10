import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

df = pd.read_csv('dr16q_merged_voidflag.csv')

mi_in = df[df['inside_void'] == 1]['M_I'].dropna()
mi_out = df[df['inside_void'] == 0]['M_I'].dropna()

kde_in = gaussian_kde(mi_in)
kde_out = gaussian_kde(mi_out)
xx = np.linspace(mi_in.min(), mi_out.max(), 300)

plt.figure(figsize=(8,5))
plt.plot(xx, kde_in(xx), label=f'Inside voids (N={len(mi_in)})')
plt.plot(xx, kde_out(xx), label=f'Outside voids (N={len(mi_out)})')
plt.xlabel('M_I')
plt.ylabel('Density')
plt.title('KDE — Absolute Magnitude (M_I) of Quasars')
plt.legend()
plt.tight_layout()
plt.show()
