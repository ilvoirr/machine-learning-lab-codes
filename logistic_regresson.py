from sklearn.linear_model import LogisticRegression
# Experiment with different regularization parameters
log_reg_model = LogisticRegression(C=0.1, solver='liblinear')
log_reg_model.fit(X_train, y_train)
y_pred = log_reg_model.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)