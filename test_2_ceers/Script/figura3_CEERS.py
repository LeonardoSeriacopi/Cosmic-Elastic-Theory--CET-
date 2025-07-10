
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("CEERS_catalog_processed.csv")

plt.figure(figsize=(8, 6))
plt.hist(df['deltaz'], bins=25, color='darkred', alpha=0.7)
plt.xlabel('Δz = z_obs - z_CET')
plt.ylabel('Number of Galaxies')
plt.title('Histogram of Elastic Redshift Correction (Δz)')
plt.grid(True)
plt.tight_layout()
plt.savefig("figura3_deltaz_histograma.png", dpi=300)
