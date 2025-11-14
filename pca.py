from sklearn.decomposition import PCA
# Implement PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Visualize the effect of dimensionality reduction using scatter plots
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.colorbar()
plt.show()