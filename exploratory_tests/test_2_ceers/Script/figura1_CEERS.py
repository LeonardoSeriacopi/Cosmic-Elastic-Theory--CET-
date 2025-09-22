
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("CEERS_catalog_processed.csv")

plt.figure(figsize=(8, 6))
plt.scatter(df['density_rhocrit'], df['deltaz'], c='blue', alpha=0.6, s=20)
plt.xlabel('Local Density (ρ / ρ_crit)')
plt.ylabel('Δz = z_obs - z_CET')
plt.title('Elastic Correction vs Local Density')
plt.grid(True)
plt.tight_layout()
plt.savefig("figura1_rho_vs_zcet_CEERS.png", dpi=300)
