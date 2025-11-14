from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
# Implement SVM with different kernels
svm_model_linear = SVC(kernel='linear').fit(X_train[:, :2], y_train)
svm_model_rbf = SVC(kernel='rbf').fit(X_train[:, :2], y_train)
# Visualize decision boundaries
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plot_decision_regions(X_train[:, :2], y_train, clf=svm_model_linear, legend=2)
plt.title('SVM with Linear Kernel')
plt.subplot(1, 2, 2)
plot_decision_regions(X_train[:, :2], y_train, clf=svm_model_rbf, legend=2)
plt.title('SVM with RBF Kernel')
plt.show()