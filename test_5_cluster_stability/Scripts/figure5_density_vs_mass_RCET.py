
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_excel("Resultado_CET_Corrigido.xlsx")

# Compute logs
log_mass = np.log10(df['M_halo'].replace(0, np.nan))
log_density = np.log10(df['rho_CET'].replace(0, np.nan))
log_R_CET = np.log10(df['R_CET'].replace(0, np.nan))

# Plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    log_mass,
    log_density,
    c=log_R_CET,
    cmap='plasma',
    alpha=0.7
)
plt.colorbar(sc, label='log10(R_CET)')
plt.xlabel('log10(Mass) [M☉]')
plt.ylabel('log10(CET Density) [kg/m³]')
plt.title('Density vs Mass (colored by R_CET)')
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig("figure5_density_vs_mass_RCET.png", dpi=300)
plt.show()
