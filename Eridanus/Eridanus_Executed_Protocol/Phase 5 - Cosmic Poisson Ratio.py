# poisson_ratio.py (modificado)
from sklearn.decomposition import PCA

# PCA para deformações
pca = PCA(n_components=3)
pca.fit(void_coords)
variance_ratio = pca.explained_variance_ratio_

# Cálculo de ν
deformacao_longitudinal = np.ptp(principal_axis[:,1])  # Variação total em z
deformacao_transversal = np.mean(variance_ratio[1:])   # Média das componentes transversais

ν_eridanus = - deformacao_transversal / deformacao_longitudinal
print(f"Razão de Poisson do vazio: ν = {ν_eridanus:.3f}")

# Resultado final
print("\n=== RESULTADOS ERI DANUS ===")
print(f"1. Tamanho do vazio: {void_coords.ptp(axis=0).max():.1f} Mpc")
print(f"2. Relaxamento integral (R): {R_void:.4f}")
print(f"3. Distorção de redshift (Δz): {Δz_pred:.4f}")
print(f"4. Razão de Poisson (ν): {ν_eridanus:.3f}")
print(f"5. Elasticidade CET: {ν_eridanus * 100:.1f}% do limite teórico")