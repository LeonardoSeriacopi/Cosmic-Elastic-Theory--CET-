
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_excel("Resultado_CET_Corrigido.xlsx")

# Categorize stability
df['Stable'] = df['R_CET'] > 1

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(
    df['z'],
    np.log10(df['rho_CET'].replace(0, np.nan)),
    c=df['Stable'].map({True: 'green', False: 'red'}),
    alpha=0.5
)
plt.xlabel('Redshift (z)')
plt.ylabel('log10(CET Density) [kg/m³]')
plt.title('Stability vs CET Density and Redshift')
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig("figure2_stability_vs_density_redshift.png", dpi=300)
plt.show()
