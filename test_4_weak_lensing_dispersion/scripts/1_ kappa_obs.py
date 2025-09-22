import pandas as pd
import numpy as np

# 1. Carregar o CSV com os dados PSF-corrigidos
df = pd.read_csv("PSF_Corrected_Galaxy_Sample.csv")

# 2. Definir o fator de resposta ao cisalhamento
R = 1.0  # Se tiver valor específico do DES, substitua aqui

# 3. Calcular kappa_obs
df["kappa_obs"] = np.sqrt(df["e1"]**2 + df["e2"]**2) / (2 * R)

# 4. Exportar o resultado
df.to_csv("PSF_with_KappaObs.csv", index=False)
print("✅ kappa_obs calculado e salvo em PSF_with_KappaObs.csv")

