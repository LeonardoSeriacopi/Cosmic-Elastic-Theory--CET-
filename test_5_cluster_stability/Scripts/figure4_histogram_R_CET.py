
import pandas as pd
import matplotlib.pyplot as plt

# Load processed data
df = pd.read_excel("Resultado_CET_Corrigido.xlsx")

# Plot
plt.figure(figsize=(10, 6))
plt.hist(df['R_CET'], bins=50, color='steelblue', edgecolor='black')
plt.axvline(1, color='red', linestyle='--', label='Stability threshold (R = 1)')
plt.xlabel('R_CET')
plt.ylabel('Number of Clusters')
plt.title('Distribution of R_CET Across the Sample')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save figure
plt.savefig("figure4_histogram_R_CET.png", dpi=300)
plt.show()
