from sklearn.tree import DecisionTreeClassifier, plot_tree
# Implement a decision tree classifier

dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_model.fit(X_train, y_train)
# Tune hyperparameters (example: max_depth)
# (This can be done using GridSearchCV or manually adjusting parameters and checking performance)
# Visualize the constructed decision tree
plt.figure(figsize=(12, 8))
plot_tree(dt_model, filled=True, feature_names=data.feature_names,
class_names=data.target_names)
plt.title('Decision Tree')
plt.show()