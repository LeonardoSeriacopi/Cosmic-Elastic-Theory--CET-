
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_excel("Resultado_CET_Corrigido.xlsx")

# Create bins of richness (n_galaxies)
bins = [0, 10, 20, 30, 40, 60, 80, 100, 150, 200, 300]
df['richness_bin'] = pd.cut(df['n_galaxies'], bins)

# Calculate stability fraction per bin
stability_by_bin = df.groupby('richness_bin')['R_CET'].apply(lambda x: (x > 1).mean()).reset_index()
stability_by_bin.columns = ['Richness Bin', 'Stability Fraction']

# Plot
plt.figure(figsize=(10, 6))
plt.bar(stability_by_bin['Richness Bin'].astype(str), stability_by_bin['Stability Fraction'], color='seagreen', edgecolor='black')
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.xlabel('Richness (Number of Galaxies)')
plt.ylabel('Fraction of Stable Clusters (R_CET > 1)')
plt.title('Cluster Stability by Richness Bin')
plt.grid(True, axis='y')
plt.tight_layout()

# Save figure
plt.savefig("figure6_stability_by_richness.png", dpi=300)
plt.show()
