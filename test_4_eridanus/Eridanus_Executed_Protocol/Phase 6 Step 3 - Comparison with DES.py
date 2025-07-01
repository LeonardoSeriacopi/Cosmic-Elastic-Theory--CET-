import numpy as np
from scipy import stats

# Estatísticas observadas
κ_obs = np.mean(kappa_void)
κ_err = np.std(kappa_void) / np.sqrt(len(kappa_void))

# Teste de compatibilidade
z_score = (κ_obs - κ_CET) / κ_err
p_value = 2 * (1 - stats.norm.cdf(np.abs(z_score)))

print("\n=== RESULTADOS DES ===")
print(f"κ observado: {κ_obs:.4f} ± {κ_err:.4f}")
print(f"κ predito (CET): {κ_CET:.4f}")
print(f"Diferença: {z_score:.2f}σ (p = {p_value:.4f})")