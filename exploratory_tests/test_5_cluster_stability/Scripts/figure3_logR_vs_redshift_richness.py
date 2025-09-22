
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_excel("Resultado_CET_Corrigido.xlsx")

# Compute log values
log_R_CET = np.log10(df['R_CET'].replace({0: np.nan}))
log_richness = np.log10(df['n_galaxies'].replace(0, np.nan))

# Plot
plt.figure(figsize=(10, 6))
sc = plt.scatter(
    df['z'],
    log_R_CET,
    c=log_richness,
    cmap='viridis',
    alpha=0.7
)
plt.colorbar(sc, label='log10(Richness)')
plt.axhline(0, color='red', linestyle='--', label='R_CET = 1 (log10 = 0)')
plt.xlabel('Redshift (z)')
plt.ylabel('log10(R_CET)')
plt.title('log10(R_CET) vs Redshift (colored by Richness)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig("figure3_logR_vs_redshift_richness.png", dpi=300)
plt.show()
