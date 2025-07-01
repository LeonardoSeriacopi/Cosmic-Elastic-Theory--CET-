
import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_excel("Resultado_CET_Corrigido.xlsx")

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(df['z'], df['R_CET'], alpha=0.6, edgecolors='k')
plt.axhline(1, color='red', linestyle='--', label='Stability threshold (R = 1)')
plt.xlabel('Redshift (z)')
plt.ylabel('R_CET')
plt.title('Cluster Stability: R_CET vs Redshift')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig("figure1_R_CET_vs_redshift.png", dpi=300)
plt.show()
