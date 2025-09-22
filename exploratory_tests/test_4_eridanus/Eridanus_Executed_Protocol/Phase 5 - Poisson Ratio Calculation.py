pca = PCA(n_components=3).fit(void_coords)
ν = -pca.explained_variance_ratio_[1:].mean() / pca.explained_variance_ratio_[0]