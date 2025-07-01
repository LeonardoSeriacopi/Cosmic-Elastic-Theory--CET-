# PCA correta com eixos espaciais
from sklearn.decomposition import PCA  
pca = PCA().fit(df[['x','y','z_coord']])  
principal_axis = pca.components_[0]  

# Projeção dos pontos ao longo do eixo principal
projected = df[['x','y','z_coord']].values @ principal_axis
vcr = np.polyfit(projected, df['z'], 1)[0]  # Gradiente longitudinal
deformacao_transversal = np.mean(pca.explained_variance_ratio_[1:])  

ν_eridanus = - deformacao_transversal / vcr
print(f"Razão de Poisson do Vazio: ν = {ν_eridanus:.3f}")
