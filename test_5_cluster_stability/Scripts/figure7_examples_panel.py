
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load processed data
df = pd.read_excel("Resultado_CET_Corrigido.xlsx")

# Select examples
stable_examples = df[df['R_CET'] > 1].sort_values(by='R_CET', ascending=False).head(3)
unstable_examples = df[df['R_CET'] <= 1].sort_values(by='R_CET']).head(3)
examples = pd.concat([stable_examples, unstable_examples])

# Plotting panel
fig, axs = plt.subplots(2, 3, figsize=(15, 8))
axs = axs.flatten()

for i, (_, row) in enumerate(examples.iterrows()):
    text = (
        f"ID: {row['cluster_id']}
"
        f"z = {row['z']:.3f}
"
        f"n_gal = {int(row['n_galaxies'])}
"
        f"M_halo = {row['M_halo']:.2e} M☉
"
        f"ρ = {row['rho_CET']:.2e} kg/m³
"
        f"P_CET = {row['P_CET']:.2e}
"
        f"P_crit = {row['P_crit_CET']:.2e}
"
        f"R_CET = {row['R_CET']:.2f}
"
        f"{'Stable ✅' if row['R_CET'] > 1 else 'Unstable ❌'}"
    )
    axs[i].text(0.1, 0.5, text, fontsize=11, va='center')
    axs[i].axis('off')

plt.suptitle("Examples of Cluster Stability (CET)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("figure7_examples_panel.png", dpi=300)
plt.show()
